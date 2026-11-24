"""Точка входа: рабочий режим (API), проверочный режим (локальный JSON), расписание.

Запуск:
  python -m src.main fetch   [--config config.yaml] [--output report.json] [--no-ai]
  python -m src.main check --input data/example_input.json [--output report.json] [--no-ai]
  python -m src.main schedule [--cron "0 9 * * *"] [--no-ai]
  python -m src.main resume --vacancy-id ID --resume data/example_resume.json

Файл/команда, с которой начинается обработка: src/main.py (см. README).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .analyzer import analyze_batch
from .config import load_config
from .fetcher import FetchError, fetch_vacancies
from .normalize import extract_list, normalize_vacancy
from .report import build_report, save_error, save_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kvota13.main")


def _apply_overrides(cfg: dict, args) -> dict:
    if getattr(args, "no_ai", False):
        cfg["ai"]["enabled"] = False
    if getattr(args, "ai", False):
        cfg["ai"]["enabled"] = True
    if getattr(args, "provider", None):
        cfg["ai"]["provider"] = args.provider
    if getattr(args, "model", None):
        cfg["ai"]["model"] = args.model
    if getattr(args, "temperature", None) is not None:
        cfg["ai"]["temperature"] = args.temperature
    if getattr(args, "ai_debug_dir", None):
        cfg["ai"]["debug_dir"] = args.ai_debug_dir
    if getattr(args, "max_vacancies", None):
        cfg["max_vacancies"] = args.max_vacancies
    if getattr(args, "pages", None):
        cfg["pages"] = args.pages
    if getattr(args, "source_url", None):
        cfg["source_url"] = args.source_url
    return cfg


def run_fetch(cfg: dict, output: str) -> int:
    """Рабочий режим: скачать с API -> анализ -> отчёт. При ошибке — сохранить старый."""
    try:
        vacancies = fetch_vacancies(cfg["source_url"], int(cfg["pages"]),
                                    int(cfg["page_size"]), int(cfg["max_vacancies"]),
                                    int(cfg["timeout_sec"]))
    except FetchError as e:
        log.error("%s", e)
        error_path = cfg.get("error_report", "report_error.json")
        if Path(error_path).resolve() == Path(output).resolve():
            error_path = str(output) + ".error.json"
        save_error(str(e), error_path, cfg["source_url"])
        old = Path(output)
        if old.exists():
            log.info("Сохранён последний успешный отчёт: %s", old)
        else:
            log.info("Предыдущего успешного отчёта нет.")
        return 2
    results, ai_status = analyze_batch(vacancies, cfg["ai"])
    save_report(build_report(results, "fetch", cfg["source_url"], ai_status), output)
    return 0


def run_check(cfg: dict, input_path: str, output: str) -> int:
    """Проверочный режим: локальный JSON в формате API, без обращения к порталу."""
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        vacancies = [normalize_vacancy(r) for r in extract_list(payload)]
    except (OSError, ValueError) as exc:
        log.error("Не удалось прочитать вход: %s", exc)
        return 1
    results, ai_status = analyze_batch(vacancies, cfg["ai"])
    save_report(build_report(results, "check", f"local:{input_path}", ai_status), output)
    return 0


def _parse_cron(cron: str) -> tuple[int, int]:
    """Поддерживаем 'M H * * *' и 'H:MM'. Возвращает (hour, minute) МСК."""
    cron = cron.strip()
    if ":" in cron:
        h, m = cron.split(":", 1)
        hour, minute = int(h), int(m)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
        raise ValueError(f"Неверное время: {cron!r}")
    parts = cron.split()
    if len(parts) == 5 and parts[2:] == ["*", "*", "*"]:
        hour, minute = int(parts[1]), int(parts[0])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    raise ValueError(f"Не понял расписание: {cron!r} (пример: '0 9 * * *' или '09:00')")


def run_schedule(cfg: dict, output: str, cron: str) -> int:
    """Цикл по расписанию: каждый день в заданное время — fetch."""
    hour, minute = _parse_cron(cron)
    log.info("Расписание: ежедневно в %02d:%02d (локальное время сервера). Ctrl+C — остановить.", hour, minute)
    while True:
        now = time.localtime()
        if now.tm_hour == hour and now.tm_min == minute and now.tm_sec < 30:
            log.info("Запуск по расписанию...")
            run_fetch(cfg, output)
            time.sleep(60)
        else:
            time.sleep(20)


def _temperature_arg(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("температура должна быть числом от 0 до 2") from exc
    if not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("температура должна быть числом от 0 до 2")
    return number


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--config",)
    common.add_argument("--output",)
    common.add_argument("--no-ai", action="store_true", help="без модели: правила + fallback")
    common.add_argument("--ai", action="store_true", help="включить ИИ (по умолчанию включён)")
    common.add_argument("--provider", choices=["local", "api", "off"],)
    common.add_argument("--model",)
    common.add_argument("--temperature", type=_temperature_arg,
                        help="температура модели от 0 до 2; 0 отключает случайную выборку")
    common.add_argument("--ai-debug-dir", help="сохранить сырые ответы ИИ и поля вакансий; могут содержать личные данные")
    common.add_argument("--max-vacancies", type=int,)
    common.add_argument("--pages", type=int,)
    common.add_argument("--source-url",)
    p = argparse.ArgumentParser(description="Квота-13: анализ понятности вакансий (регион 39)",
                                parents=[common])
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("fetch", parents=[common], help="рабочий режим: скачать с API и сформировать отчёт")
    pc = sub.add_parser("check", parents=[common], help="проверочный режим: локальный JSON без портала")
    pc.add_argument("--input", required=True)
    ps = sub.add_parser("schedule", parents=[common], help="цикл по расписанию (fetch)")
    ps.add_argument("--cron",)
    pr = sub.add_parser("resume", parents=[common], help="доп. сценарий: черновик резюме под вакансию")
    pr.add_argument("--vacancy-id", required=True)
    pr.add_argument("--resume", required=True)
    args = p.parse_args(argv)

    cfg = _apply_overrides(load_config(getattr(args, "config", None)), args)
    output = getattr(args, "output", None) or cfg.get("output_report", "report.json")
    cmd = args.cmd or "fetch"

    if cmd == "fetch":
        return run_fetch(cfg, output)
    if cmd == "check":
        return run_check(cfg, args.input, output)
    if cmd == "schedule":
        return run_schedule(cfg, output, getattr(args, "cron", None) or cfg.get("schedule_cron", "0 9 * * *"))
    if cmd == "resume":
        from .extra_resume import draft_resume
        payload = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        # Вакансию берём из последнего отчёта или example_input по ID
        vacancy = {"id": args.vacancy_id, "title": "", "company": "", "requirements": ""}
        for cand in [Path(output), Path("data/example_input.json")]:
            if cand.exists():
                data = json.loads(cand.read_text(encoding="utf-8"))
                items = data.get("vacancies", data) if isinstance(data, dict) else data
                for it in extract_list(data) if isinstance(data, dict) and "vacancies" not in data else items:
                    nv = normalize_vacancy(it) if isinstance(it, dict) else it
                    if isinstance(nv, dict) and str(nv.get("id")) == str(args.vacancy_id):
                        vacancy = nv
                        break
        print(json.dumps(draft_resume(vacancy, payload, cfg["ai"]), ensure_ascii=False, indent=2))
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
