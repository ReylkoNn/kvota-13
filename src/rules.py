"""Rule-based анализ понятности вакансии (роль №2, часть 1).

Проверяет: название, обязанности, требования, оплату, график, место работы,
тип занятости — по описанию И структурированным полям. Если сведения есть
в структурированном поле, отсутствие их повтора в описании не считается
проблемой (проверяем объединённый текст).

Каждое замечание: topic / field / quote / problem / recommendation / severity.
Для пропуска quote="" ; для противоречия quote содержит оба значения.
"""
from __future__ import annotations

import re

from .normalize import _ALIASES

TOPIC_ORDER = ["Название", "Обязанности", "Требования", "Оплата", "График",
               "Место работы", "Тип занятости", "Описание"]

_GENERIC_TITLES = {"специалист", "работник", "сотрудник", "помощник", "менеджер",
                   "оператор", "служащий", "рабочий"}
_VAGUE_DUTIES = ["выполнение должностных обязанностей", "работа по специальности",
                 "иные обязанности", "исполнение обязанностей", "выполнение обязанностей"]
_SOFT_SKILLS = ["ответственность", "коммуникабельность", "стрессоустойчивость",
                "пунктуальность", "вежливость", "исполнительность", "трудолюбие"]
_VAGUE_SCHEDULE = ["гибкий график", "плавающий график", "свободный график",
                   "ненормированный рабочий день", "ненормированный день",
                   "удобный график", "разные смены"]
_VAGUE_PAY = ["по договорённости", "конкурентная зарплата", "конкурентная оплата",
              "достойная оплата", "стабильная зарплата", "стабильная оплата"]
_VAGUE_PLACE = ["в офисе", "на предприятии", "на производстве", "в организации"]


def _r(topic, field, quote, problem, recommendation, severity="средняя"):
    return {"topic": topic, "field": field, "quote": quote or "",
            "problem": problem, "recommendation": recommendation,
            "severity": severity, "source": "rule"}


def _has(patterns, text: str) -> str | None:
    low = text.lower()
    for p in patterns:
        if p in low:
            return p
    return None


def _numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d[\d\s\u00a0]*", text or "") if x.strip()]


def _source_name(v: dict, aliases: list[str], default: str) -> str:
    raw = v.get("raw", {})
    return next((name for name in aliases if name in raw and raw[name] not in (None, "")), default)


def _source_for_quote(v: dict, quote: str, default: str) -> str:
    for name, value in v.get("raw", {}).items():
        if isinstance(value, str) and quote in value:
            return name
    return default


def analyze_rules(v: dict) -> list[dict]:
    out: list[dict] = []
    title = v.get("title", "")
    desc = v.get("description", "")
    duties = v.get("duties", "")
    req = v.get("requirements", "")
    all_text = v.get("all_text", "")
    low = all_text.lower()
    duties_text = f"{duties} {desc}"
    req_text = f"{req} {desc}"

    # --- Название ---
    if not title:
        out.append(_r("Название", "vacancyName", "", "Не указано название должности.",
                      "Укажите название должности (профессия + специализация, если есть).", "высокая"))
    elif title.strip().lower() in _GENERIC_TITLES:
        out.append(_r("Название", "vacancyName", title, "Название слишком общее и не говорит о специализации.",
                      "Уточните специализацию в названии, например «Специалист по закупкам».", "средняя"))

    # --- Обязанности ---
    if not duties.strip() and len(desc.strip()) < 40:
        out.append(_r("Обязанности", "duties", "", "Не описаны обязанности: ни в поле обязанностей, ни в описании.",
                      "Перечислите 3–7 конкретных задач (что именно делает сотрудник).", "высокая"))
    else:
        hit = _has(_VAGUE_DUTIES, duties_text)
        if hit:
            out.append(_r("Обязанности", "duties", hit,
                          "Формулировка обязанностей расплывчатая и не показывает реальный объём работы.",
                          "Замените общую фразу списком конкретных задач и зон ответственности.", "средняя"))

    # --- Требования ---
    if not req.strip() and len(desc.strip()) < 30:
        out.append(_r("Требования", "requirements", "", "Не указаны требования к соискателю.",
                      "Укажите обязательные навыки, опыт и образование (если требуются).", "высокая"))
    else:
        soft = [s for s in _SOFT_SKILLS if s in req_text.lower()]
        concrete = bool(re.search(r"опыт|стаж|образован|знание|навык|умение|владение|категор|разряд|лет\b|\d+\s*лет", req_text, re.I))
        if soft and not concrete:
            out.append(_r("Требования", "requirements", ", ".join(soft),
                          "Перечислены только личные качества без конкретных профессиональных требований.",
                          "Добавьте конкретные навыки/опыт/образование, а личные качества оставьте дополнением.", "средняя"))
        elif not re.search(r"опыт|стаж|образован|лет\b|\d+\s*лет", req_text, re.I):
            out.append(_r("Требования", "requirements", "",
                          "Не понятно, нужен ли опыт и какое образование требуется.",
                          "Явно укажите требуемый опыт (в годах) и уровень образования или «без опыта».", "средняя"))

    # --- Оплата ---
    smin, smax = v.get("salary_min"), v.get("salary_max")
    stext = v.get("salary_text", "")
    pay_text = f"{stext} {desc}"
    has_pay = smin is not None or smax is not None or bool(stext.strip()) or bool(re.search(r"\d+\s*(руб|₽|тыс)", pay_text, re.I))
    if not has_pay:
        out.append(_r("Оплата", "salary", "", "Не указана оплата ни в полях, ни в описании.",
                      "Укажите вилку оплаты (от–до) и что входит в сумму (оклад/премии).", "высокая"))
    else:
        hit = _has(_VAGUE_PAY, pay_text)
        if hit and smin is None and smax is None:
            out.append(_r("Оплата", "salary", hit,
                          "Оплата описана расплывчато, соискатель не может оценить доход.",
                          "Укажите конкретную сумму или вилку, а не «по договорённости».", "высокая"))
        elif re.search(r"\bот\s+\d+", pay_text, re.I) and smax is None and not re.search(r"до\s+\d+", pay_text, re.I):
            out.append(_r("Оплата", "salary", stext or "«от …»",
                          "Указан только нижний порог оплаты, верхний не понятен.",
                          "Укажите вилку «от–до» или условия, от которых зависит сумма.", "средняя"))
    if smin is not None and smax is not None and smin > smax:
        min_field = _source_name(v, ["salaryMin", "salary_min", "salary_from", "salaryFrom", "minSalary", "зарплатаОт"], "salaryMin")
        max_field = _source_name(v, ["salaryMax", "salary_max", "salary_to", "salaryTo", "maxSalary", "зарплатаДо"], "salaryMax")
        raw = v.get("raw", {})
        min_value = str(raw.get(min_field, f"{smin:g}"))
        max_value = str(raw.get(max_field, f"{smax:g}"))
        out.append(_r("Оплата", f"{min_field} | {max_field}", f"{min_value} | {max_value}",
                      "Противоречие: минимальная оплата больше максимальной.",
                      "Проверьте и исправьте значения минимальной и максимальной оплаты.", "высокая"))

    # --- График ---
    sched = f"{v.get('schedule_type','')} {v.get('schedule_text','')}"
    sched_all = f"{sched} {desc}"
    has_sched = bool(sched.strip()) or bool(re.search(r"смен|график|рабоч\w+ (день|недел)|пятиднев|2/2|5/2|1/1|пн|будн", sched_all, re.I))
    if not has_sched:
        out.append(_r("График", "scheduleType", "", "Не указан график работы.",
                      "Укажите дни и часы работы (например, 5/2, 09:00–18:00).", "высокая"))
    else:
        hit = _has(_VAGUE_SCHEDULE, sched_all)
        concrete = bool(re.search(r"\d\s*/\s*\d|\d{1,2}:\d{2}|пн|будн|смен[аыу]?\s+\d|через\s+\d|выходн", sched_all, re.I))
        if hit and not concrete:
            out.append(_r("График", "scheduleType", hit,
                          f"Формулировка «{hit}» не даёт конкретного расписания: дни и смены не понятны.",
                          "Уточните дни недели, часы и смену (например, «5/2, с 09:00 до 18:00, смена 1»).", "высокая"))

    # --- Место работы ---
    place = f"{v.get('workplace','')} {v.get('address','')} {v.get('city','')}"
    place_all = f"{place} {desc}"
    has_place = bool(place.strip()) or bool(re.search(r"калининград|удалённо|дистанционно|на дому|адрес|ул\.|улица", place_all, re.I))
    if not has_place:
        out.append(_r("Место работы", "workPlace", "", "Не указано место работы.",
                      "Укажите город/адрес или возможность удалённой работы.", "высокая"))
    else:
        hit = _has(_VAGUE_PLACE, place_all)
        if hit and not re.search(r"калининград|ул\.|улица|адрес|удалённо|дистанционно", place_all, re.I):
            out.append(_r("Место работы", "workPlace", hit,
                          "Место указано общо («в офисе») без города/адреса.",
                          "Укажите город и адрес (или район) места работы.", "средняя"))
        if "удалённо" in place_all.lower() and re.search(r"в офисе|на предприятии", place_all, re.I) \
                and "удалённо" not in place.lower():
            remote = re.search(r"удалённо", place_all, re.I).group()
            office = re.search(r"в офисе|на предприятии", place_all, re.I).group()
            remote_field = _source_for_quote(v, remote, "description")
            office_field = _source_for_quote(v, office, "workPlace")
            fields = remote_field if remote_field == office_field else f"{remote_field} | {office_field}"
            out.append(_r("Место работы", fields, f"{remote} | {office}",
                          "Указаны удалённая работа и работа в офисе, но порядок их сочетания неясен.",
                          "Уточните формат: удалённо, гибридно или в офисе (и в каком).", "средняя"))

    # --- Тип занятости ---
    emp = v.get("employment_type", "")
    desc_employment = re.search(r"(?:полная|частичная|временная) занятость", desc, re.I)
    if not emp.strip() and not desc_employment:
        out.append(_r("Тип занятости", "employmentType", "", "Не указан тип занятости.",
                      "Укажите: полная/частичная занятость, срочный договор, стажировка и т.п.", "высокая"))
    else:
        el = emp.lower()
        workload = re.search(r"40\s*час(?:ов|а)?|полн\w+ (?:день|недел\w*)", all_text, re.I)
        if "частичн" in el and workload:
            emp_field = _source_name(v, _ALIASES["employment_type"], "employmentType")
            workload_field = _source_for_quote(v, workload.group(), "description")
            out.append(_r("Тип занятости", f"{emp_field} | {workload_field}", emp + " | " + workload.group(),
                          "Указана частичная занятость, но также приведена полная нагрузка; соотношение неясно.",
                          "Проверьте тип занятости и количество часов в неделю.", "средняя"))

    # --- Описание ---
    if desc and len(desc.strip()) < 80 and len(duties_text.strip()) < 80 and len(req_text.strip()) < 80:
        out.append(_r("Описание", "description", desc[:80],
                      "Описание слишком краткое и не раскрывает условия работы.",
                      "Добавьте описание задач, требований и условий (или заполните поля API).", "низкая"))

    raw = v.get("raw", {})
    for remark in out:
        quote = remark["quote"]
        if quote and " | " not in quote:
            for field, value in raw.items():
                if isinstance(value, str):
                    match = re.search(re.escape(quote), value, re.I)
                    if match:
                        remark["field"], remark["quote"] = field, match.group()
                        break
    out.sort(key=lambda r: TOPIC_ORDER.index(r["topic"]) if r["topic"] in TOPIC_ORDER else 99)
    return out
