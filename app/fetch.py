import json
import logging

import requests

from src.normalize import extract_list


log = logging.getLogger(__name__)


class ApiError(Exception):
    pass


def unwrap(data):
    # Не принимаем сообщение об ошибке за пустую страницу: иначе неполная
    # загрузка может заменить последний успешный отчёт.
    try:
        items = extract_list(data)
    except ValueError as exc:
        raise ApiError(f'Некорректный формат данных: {exc}') from exc
    if not all(isinstance(item, dict) for item in items):
        raise ApiError('В списке вакансий обнаружен не объект')
    return items


def fetch_page(url, offset, limit):
    try:
        resp = requests.get(url, params={'offset': offset, 'limit': limit}, timeout=50)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ApiError(f'API недоступен: {e}') from e
    except ValueError as e:
        raise ApiError('API вернул не JSON') from e
    return unwrap(data)


def fetch_vacancies(url, page_size, max_vacancies, max_pages=10):
    if page_size < 1 or max_vacancies < 1 or max_pages < 1:
        raise ValueError('page_size, max_vacancies и max_pages должны быть положительными')
    # Если возможно, запрашиваем хотя бы две страницы.
    page_size = min(page_size, 100, max(1, (max_vacancies + 1) // 2))
    vacancies = []
    ids = set()
    for page in range(max_pages):
        batch = fetch_page(url, page * page_size, page_size)
        log.info("Страница %d: получено %d вакансий", page + 1, len(batch))
        if not batch:
            break
        for vacancy in batch:
            raw_id = vacancy.get('id', vacancy.get('vacancyId', vacancy.get('vacancy_id', vacancy.get('ID'))))
            key = str(raw_id).strip() if raw_id is not None else ''
            if key and key in ids:
                continue
            if key:
                ids.add(key)
            vacancies.append(vacancy)
            if len(vacancies) >= max_vacancies:
                break
        if len(vacancies) >= max_vacancies or len(batch) < page_size:
            break
    if not vacancies:
        raise ApiError('API вернул 0 вакансий - проверь URL и регион')
    return vacancies


def load_local(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return unwrap(data)
