"""Загрузка вакансий с API «Работа в России» (регион 39).

- Пагинация limit/offset, минимум 2 страницы (config.pages), лимит max_vacancies.
- Дедупликация по ID: повторный запуск не плодит дубли в текущем отчёте.
- При недоступности API бросает FetchError — вызывающий код сохраняет
  последний успешный отчёт и пишет ошибку отдельно.
"""
from __future__ import annotations

import json
import http.client
import logging
import urllib.error
import urllib.parse
import urllib.request

from .normalize import extract_list, normalize_vacancy

log = logging.getLogger("kvota13.fetcher")


class FetchError(RuntimeError):
    pass


def _get_json(url: str, timeout: int) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "kvota13/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except json.JSONDecodeError as e:
        raise FetchError(f"API вернуло не JSON: {e}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException, UnicodeError) as e:  # сеть/HTTP/обрыв соединения
        raise FetchError(f"API недоступно: {e}") from e


def fetch_vacancies(source_url: str, pages: int = 2, page_size: int = 20,
                    max_vacancies: int = 40, timeout: int = 30) -> list[dict]:
    """Скачать страницы и вернуть нормализованные вакансии (без дублей по ID)."""
    seen: dict[str, dict] = {}
    for page in range(pages):
        offset = page * page_size
        qs = urllib.parse.urlencode({"limit": page_size, "offset": offset})
        sep = "&" if "?" in source_url else "?"
        payload = _get_json(f"{source_url}{sep}{qs}", timeout)
        try:
            items = extract_list(payload)
        except ValueError as e:
            raise FetchError(f"Некорректный ответ API: {e}") from e
        log.info("Страница %d: получено %d вакансий", page + 1, len(items))
        if not items:
            break
        for raw in items:
            v = normalize_vacancy(raw)
            key = v["id"] or f"noid-{len(seen)}"
            seen[key] = v  # повторный запуск/страница: один ID — один объект
            if len(seen) >= max_vacancies:
                break
        if len(seen) >= max_vacancies or len(items) < page_size:
            break
    return list(seen.values())
