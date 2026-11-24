"""Небольшой A/B эксперимент с локальной моделью на вымышленных данных.

Этот модуль не меняет основной анализатор и не формирует замечания от имени ИИ.
Он сохраняет ответы двух более коротких запросов для ручной проверки.
Сетевую загрузку Hugging Face отключаем до загрузки Transformers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from .ai import GemmaLocalAnalyzer, _accepted_remarks, _vacancy_text
from .normalize import extract_list, normalize_vacancy

log = logging.getLogger("kvota13.prompt_probe")

_SYSTEM = (
    "Ты редактор вакансий. Ищи реальные неясности в условиях работы. "
    "Используй все поля вакансии, не выдумывай факты. "
    "Если информация уже указана в другом поле, учитывай её."
)
_REQUESTS = {
    "short_json": (
        "Найди формулировку, по которой соискателю нужно уточнить условия работы. "
        "Если она есть, верни JSON-массив с одним объектом: "
        '{"topic":"Описание","field":"description","quote":"точная цитата",'
        '"problem":"что именно неясно","recommendation":"что конкретно уточнить",'
        '"severity":"средняя"}. '
        "Поле и цитату скопируй из вакансии. Если неясности нет, верни []. Без пояснений."
    ),
    "yes_no": (
        "Есть ли в условиях работы неясная формулировка, для которой соискателю "
        "нужно задать уточняющий вопрос? Учитывай все поля. "
        "Ответь одним словом: ДА или НЕТ."
    ),
}


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        try:
            return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        except Exception:
            return merged[0]["content"]


def _decode_yes_no(text: str) -> str | None:
    match = re.match(r"^\s*(ДА|НЕТ)(?=$|[^\w])", text, re.I)
    return match.group(1).upper() if match else None


def run_probe(vacancies: list[dict], pipe) -> list[dict]:
    results = []
    total = len(vacancies) * len(_REQUESTS)
    step = 0
    for v in vacancies:
        for strategy, request in _REQUESTS.items():
            step += 1
            log.info("Запрос %d/%d: вакансия %s, вариант %s", step, total, v.get("id", ""), strategy)
            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Вакансия:\n{_vacancy_text(v)}\n\n{request}"},
            ]
            tokenizer = getattr(pipe, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("У модели нет токенизатора для эксперимента")
            prompt = _render_prompt(tokenizer, messages)
            output = pipe(prompt, max_new_tokens=384, do_sample=False, return_full_text=False)
            text = output[0]["generated_text"] if output else ""
            item = {"vacancy_id": v.get("id", ""), "strategy": strategy, "raw_response": text}
            if strategy == "yes_no":
                item["answer"] = _decode_yes_no(text)
            else:
                try:
                    item["accepted_remarks"] = _accepted_remarks(text, v)
                    item["parse_error"] = None
                except ValueError as exc:
                    item["accepted_remarks"] = []
                    item["parse_error"] = str(exc)
            results.append(item)
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Офлайн-сравнение коротких запросов к ИИ на вымышленных данных")
    p.add_argument("--input", required=True, help="локальный JSON вакансий")
    p.add_argument("--model", required=True, help="модель, уже сохранённая в локальном кэше")
    p.add_argument("--output", required=True, help="куда записать диагностический JSON")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Выставляем до импорта Transformers/Hugging Face внутри _load().
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        vacancies = [normalize_vacancy(x) for x in extract_list(payload)]
        if not vacancies:
            raise ValueError("Входной файл не содержит вакансий")
        model = GemmaLocalAnalyzer(args.model, temperature=0)
        pipe = model._load()
        results = run_probe(vacancies, pipe)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"model": args.model, "offline": True,
                                      "note": "Эксперимент; ответы требуют ручной проверки и не являются отчётом.",
                                      "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Эксперимент сохранён: %s", output)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        log.error("Эксперимент не выполнен: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
