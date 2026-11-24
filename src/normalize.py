"""Приведение сырой вакансии API «Работа в России» к каноническому виду.

API в разных версиях отдаёт разные имена полей, поэтому маппим несколько
вариантов. Пустые/отсутствующие значения -> "" / None, а не исключение:
анализатор сам решит, что это пропуск.
"""
from __future__ import annotations

import math
from typing import Any

# каноническое поле -> список возможных имён в сыром JSON
_ALIASES = {
    "id": ["id", "vacancyId", "vacancy_id", "ID"],
    "title": ["vacancyName", "job-name", "name", "title", "position", "profession", "должность"],
    "company": ["companyName", "company", "employerName", "organization", "работодатель"],
    "description": ["description", "vacancyDescription", "text", "comment", "описание"],
    "duties": ["duties", "duty", "responsibilities", "jobDuties", "обязанности", "функции"],
    "requirements": ["requirements", "requirement", "skills", "требования", "навыки"],
    "salary_text": ["salary", "salaryText", "salary_text", "pay", "оплата", "зарплата"],
    "schedule_type": ["scheduleType", "schedule", "workSchedule", "график"],
    "schedule_text": ["scheduleText", "workTime", "режимРаботы"],
    "employment_type": ["employmentType", "employment", "typeOfEmployment", "busyType", "занятость"],
    "workplace": ["workPlace", "workplace", "place", "местоРаботы"],
    "address": ["address", "location", "адрес"],
    "city": ["city", "город"],
    "url": ["url", "vac_url", "link", "vacancyUrl", "cardUrl", "ссылка"],
}


def _pick(raw: dict, keys: list[str]) -> Any:
    for k in keys:
        if k in raw and raw[k] not in (None, ""):
            return raw[k]
    return None


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        value = float(str(v).replace("\u00a0", "").replace(" ", ""))
        # В API нулевые и нечисловые значения зарплаты не означают указанную сумму.
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_vacancy(raw: dict) -> dict:
    """Вернуть канонический словарь вакансии."""
    if not isinstance(raw, dict):
        raw = {"description": str(raw)}
    if isinstance(raw.get("vacancy"), dict):
        raw = raw["vacancy"]
    v: dict[str, Any] = {}
    for canon, aliases in _ALIASES.items():
        # Структурированное requirement не должно перекрывать текстовый skills.
        val = _pick(raw, [key for key in aliases if not (canon == "requirements" and isinstance(raw.get(key), dict))])
        if canon == "company" and isinstance(val, dict):
            val = val.get("name")
        if isinstance(val, dict):
            # requirement.education/experience обрабатываются отдельно ниже.
            val = None
        elif isinstance(val, list):
            val = "; ".join(str(x).strip() for x in val if isinstance(x, (str, int, float)) and str(x).strip())
        v[canon] = "" if val is None else str(val).strip()
    details = raw.get("requirement") if isinstance(raw.get("requirement"), dict) else {}
    for field in ("education", "experience"):
        value = details.get(field)
        # Числовое 0 в поле опыта сохраняем как значение источника, а не трактуем как факт.
        v[f"requirement_{field}"] = "" if value is None or isinstance(value, (dict, list, bool)) else str(value).strip()
    v["id"] = v["id"] or ""
    v["salary_min"] = _num(_pick(raw, ["salaryMin", "salary_min", "salary_from", "salaryFrom", "minSalary", "зарплатаОт"]))
    v["salary_max"] = _num(_pick(raw, ["salaryMax", "salary_max", "salary_to", "salaryTo", "maxSalary", "зарплатаДо"]))
    # Не выдумываем ссылку: API может использовать другой формат ID.
    addresses = raw.get("addresses", {})
    if not v["address"] and isinstance(addresses, dict):
        entries = addresses.get("address", [])
        if isinstance(entries, dict):
            entries = [entries]
        v["address"] = "; ".join(str(x.get("location", "")) for x in entries if isinstance(x, dict) and x.get("location"))
    # Объединённый текст для проверок «есть ли сведения хоть где-то»
    v["all_text"] = " ".join(
        x for x in [v["title"], v["description"], v["duties"], v["requirements"],
                    v["requirement_education"], v["requirement_experience"],
                    v["salary_text"], v["schedule_type"], v["schedule_text"],
                    v["employment_type"], v["workplace"], v["address"], v["city"]] if x
    )
    v["raw"] = raw
    return v


def extract_list(payload: Any) -> list[dict]:
    """Извлечь вакансии, включая results.vacancies[].vacancy API региона.

    Неизвестный ответ — ошибка, а не пустая страница: иначе ошибка API
    может перезаписать последний успешный отчёт пустым.
    """
    if isinstance(payload, list):
        if not all(isinstance(x, dict) for x in payload):
            raise ValueError("Список вакансий содержит не объекты")
        return [x.get("vacancy", x) for x in payload]
    if isinstance(payload, dict):
        for key in ("results", "data", "vacancies", "items", "vacancy"):
            if key in payload and isinstance(payload[key], (list, dict)):
                return extract_list(payload[key])
    raise ValueError("Неизвестный формат ответа API: список вакансий не найден")
