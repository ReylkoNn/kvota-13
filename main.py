import sys
import logging
import argparse
import time
import os

from src.config import load_config
from src.report import save_error
from pathlib import Path
from app.integration import analyze_raw_vacancies

# Импорты для Льва
from app.fetch import fetch_vacancies, load_local, ApiError

from app.report import save_integrated_report

# Настройка логов
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(raw_data, mode, cfg):
    # Анализатор получает исходные вакансии и сохраняет точные поля для цитат.
    return analyze_raw_vacancies(raw_data, cfg['ai'])


def write_report(results, ai_status, mode, cfg):
    ai = cfg['ai']
    requested_model = None
    if ai.get('enabled', True) and ai.get('provider') != 'off':
        requested_model = (os.getenv('OPENAI_MODEL', 'google/gemma-2-9b-it:free')
                           if ai.get('provider') == 'api' else ai.get('model'))
    save_integrated_report(results, mode, requested_model, cfg['output']['report_path'], ai_status)


def record_fetch_error(exc, cfg):
    output = cfg['output']['report_path']
    error_path = cfg.get('error_report', 'report_error.json')
    if Path(error_path).resolve() == Path(output).resolve():
        error_path = str(output) + '.error.json'
    try:
        save_error(str(exc), error_path, cfg['source']['url'])
    except OSError as error:
        logger.error("Не удалось сохранить файл ошибки: %s", error)


def fetch_with_retries(cfg):
    max_retries = cfg['source'].get('max_retries', 3)
    delay = cfg['source'].get('retry_delay_seconds', 5)

    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError('max_retries должен быть целым числом не меньше 1')
    if not isinstance(delay, (int, float)) or isinstance(delay, bool) or delay < 0:
        raise ValueError('retry_delay_seconds должен быть неотрицательным числом')

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Попытка загрузки данных ({attempt}/{max_retries})...")
            return fetch_vacancies(
                cfg['source']['url'],
                cfg['source']['page_size'],
                cfg['source']['max_vacancies'],
                cfg['source'].get('max_pages', 10)
            )
        except ApiError as e:
            logger.warning(f"Сбой API (попытка {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                #  пробрасываем ошибку выше чтобы внешний except не дал перезаписать старый отчет
                logger.error("Все попытки исчерпаны.")
                raise e
            logger.info(f"Ждем {delay} секунд перед следующей попыткой...")
            time.sleep(delay)

def main():
    # Общие параметры принимаются до или после команды.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument('--config', help='Путь к конфигурации YAML')
    common.add_argument('--output', help='Путь к отчёту')
    common.add_argument('--no-ai', action='store_true', help='Только правила и fallback')
    common.add_argument('--provider', choices=['local', 'api', 'off'])
    common.add_argument('--model', help='Модель для локального запуска')
    parser = argparse.ArgumentParser(description="Vacancy Checker", parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", parents=[common], help="Запуск через API")
    check_parser = subparsers.add_parser("check", parents=[common], help="Проверка локального JSON")
    check_parser.add_argument("--input", required=True, help="Путь к файлу")
    subparsers.add_parser("schedule", parents=[common], help="Работа по расписанию")
    args = parser.parse_args()
    cfg = load_config(getattr(args, 'config', None))
    if getattr(args, 'no_ai', False):
        cfg['ai']['enabled'] = False
    if getattr(args, 'provider', None):
        cfg['ai']['provider'] = args.provider
    if getattr(args, 'model', None):
        cfg['ai']['model'] = args.model
    if getattr(args, 'output', None):
        cfg['output']['report_path'] = args.output

    try:
        if args.command == "run":
            logger.info("Запуск через API...")
            raw = fetch_with_retries(cfg)
            results, ai_status = run_pipeline(raw, mode="api", cfg=cfg)
            write_report(results, ai_status, mode="api", cfg=cfg)
            logger.info(f"Обработано вакансий: {len(results)}")

        elif args.command == "check":
            logger.info(f"Запуск проверки файла: {args.input}")
            raw = load_local(args.input)
            results, ai_status = run_pipeline(raw, mode="check", cfg=cfg)
            write_report(results, ai_status, mode="check", cfg=cfg)
            logger.info(f"Обработано вакансий: {len(results)}")

        elif args.command == "schedule":
            logger.info("Запуск по расписанию...")
            hours = cfg['schedule']['every_hours']
            if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0:
                raise ValueError("schedule.every_hours должен быть положительным числом")
            logger.info(f"Интервал: каждые {hours} часов. (Нажми Ctrl+C для выхода)")
            while True:
                try:
                    logger.info("Плановая загрузка данных...")
                    raw = fetch_with_retries(cfg)
                    results, ai_status = run_pipeline(raw, mode="api", cfg=cfg)
                    write_report(results, ai_status, mode="api", cfg=cfg)
                    logger.info(f"Плановый отчет сохранен. Ждем {hours} ч. ...")
                    time.sleep(hours * 3600)
                except KeyboardInterrupt:
                    logger.info("\nРабота остановлена пользователем (Ctrl+C). Выход.")
                    sys.exit(0)
                except Exception as e:
                    # если тут, то все ретраи провалились
                    logger.error(f"Ошибка планового запуска: {e}. Ждем следующего планового запуска.")
                    if isinstance(e, ApiError):
                        record_fetch_error(e, cfg)
                    time.sleep(hours * 3600) # если API совсем лег, ждем до следующего часа
    except KeyboardInterrupt:
        logger.info("\nРабота остановлена пользователем (Ctrl+C)")
        sys.exit(0)
    except ApiError as e:
        logger.error(f"Ошибка API: {e}. Старый отчет не тронут.")
        record_fetch_error(e, cfg)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
