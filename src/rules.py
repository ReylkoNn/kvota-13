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

from .normalize import _ALIASES, _num

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


def _has_pay_amount(v: dict) -> bool:
    """Число в поле оплаты или сумма с валютой рядом с упоминанием оплаты.

    Это эвристика, а не финансовый парсер: числа в произвольном описании
    (например, адрес или стаж) не считаются зарплатой.
    """
    if v.get("salary_min") is not None or v.get("salary_max") is not None:
        return True
    amount = r"(?<!\d)[1-9]\d*(?:[ \u00a0]\d{3})*(?:[.,]\d+)?"
    salary = v.get("salary_text", "")
    if re.search(amount, salary):
        return True
    desc = v.get("description", "")
    pay = r"зарплат\w*|заработн\w+\s+плат\w*|оплат\w*|оклад\w*|доход\w*"
    currency = r"руб\w*|₽|тыс\w*"
    return bool(re.search(rf"(?:{pay}).{{0,60}}?{amount}\s*(?:{currency})|"
                          rf"{amount}\s*(?:{currency}).{{0,60}}?(?:{pay})", desc, re.I))


def _source_name(v: dict, aliases: list[str], default: str) -> str:
    raw = v.get("raw", {})
    return next((name for name in aliases if name in raw and raw[name] not in (None, "")), default)


def _source_strings(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _source_strings(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _source_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _source_for_quote(v: dict, quote: str, default: str) -> str:
    for name, value in _source_strings(v.get("raw", {})):
        if quote in value:
            return name
    return default


def analyze_rules(v: dict) -> list[dict]:
    out: list[dict] = []
    title = v.get("title", "")
    desc = v.get("description", "")
    duties = v.get("duties", "")
    req = v.get("requirements", "")
    qualification = v.get("requirement_qualification", "")
    all_text = v.get("all_text", "")
    low = all_text.lower()
    duties_text = f"{duties} {desc}"
    req_text = f"{req} {qualification} {desc}"

    # --- Название ---
    if not title:
        out.append(_r("Название", "vacancyName", "", "Не указано название должности.",
                      "Укажите название должности (профессия + специализация, если есть).", "высокая"))
    elif title.strip().lower() in _GENERIC_TITLES:
        out.append(_r("Название", "vacancyName", title, "Название слишком общее и не говорит о специализации.",
                      "Уточните специализацию в названии, например «Специалист по закупкам».", "средняя"))

    # --- Обязанности ---
    duty_hint = re.search(r"обязанност\w*|задач\w*|функци\w*|"
                          r"(?:готовить|приготовлени\w*|обслуживани\w*|продаж\w*|"
                          r"контрол\w*|управлени\w*|разработк\w*|ведение|обеспечени\w*)", desc, re.I)
    if not duties.strip() and not duty_hint:
        out.append(_r("Обязанности", "duties", "", "Не удалось найти описание обязанностей в полях вакансии.",
                      "Проверьте описание и перечислите конкретные задачи сотрудника.", "высокая"))
    else:
        hit = _has(_VAGUE_DUTIES, duties_text)
        if hit:
            out.append(_r("Обязанности", "duties", hit,
                          "Формулировка обязанностей расплывчатая и не показывает реальный объём работы.",
                          "Замените общую фразу списком конкретных задач и зон ответственности.", "средняя"))

    # --- Требования ---
    experience = v.get("requirement_experience", "")
    # Значение 0 в данных источника может быть служебным, не утверждаем «без опыта».
    nested = bool(v.get("requirement_education") or (experience and experience != "0"))
    req_hint = re.search(r"требован\w*|опыт|стаж|образован|знани\w*|навык|умени\w*|владени\w*|категор|разряд", req_text, re.I)
    soft = _has(_SOFT_SKILLS, req_text)
    if not (req.strip() or qualification.strip()) and not req_hint and not soft and not nested:
        out.append(_r("Требования", "requirements", "", "Не удалось найти требования к соискателю в полях вакансии.",
                      "Проверьте описание и укажите обязательные навыки, опыт и образование, если требуются.", "высокая"))
    elif soft and not req_hint and not nested:
        out.append(_r("Требования", "requirements", soft,
                      "Указаны личные качества; профессиональные требования не удалось определить автоматически.",
                      "Проверьте и при необходимости добавьте профессиональные требования.", "средняя"))
    elif not nested and not re.search(r"опыт|стаж|образован|лет\b|\d+\s*лет", req_text, re.I):
        out.append(_r("Требования", "requirements", "",
                      "Не понятно, нужен ли опыт и какое образование требуется.",
                      "Явно укажите требуемый опыт (в годах) и уровень образования или «без опыта».", "средняя"))

    # Сопоставляем два конкретных поля API, не приписывая числу опыта единицы
    # измерения и не объявляя это противоречием: смысл кода может меняться.
    preferred = re.search(r"опыт\s+работы[^.!?;]{0,100}?приветствуется", qualification, re.I)
    requirement_raw = v.get("raw", {}).get("requirement", {})
    raw_experience = requirement_raw.get("experience") if isinstance(requirement_raw, dict) else None
    if (preferred and not re.search(r"не\s+приветствуется$", preferred.group(), re.I)
            and _num(experience) is not None and isinstance(raw_experience, (int, float, str))
            and not isinstance(raw_experience, bool)):
        out.append(_r("Требования", "requirement.qualification | requirement.experience",
                      f"{preferred.group()} | {raw_experience}",
                      "В тексте опыт назван желательным, а в отдельном поле опыта задано значение; их связь неясна.",
                      "Проверьте поле опыта и уточните, является ли опыт обязательным или желательным.", "средняя"))

    # --- Оплата ---
    smin, smax = v.get("salary_min"), v.get("salary_max")
    stext = v.get("salary_text", "")
    pay_text = f"{stext} {desc}"
    has_pay = _has_pay_amount(v)
    if not has_pay:
        hit = _has(_VAGUE_PAY, pay_text)
        if hit:
            out.append(_r("Оплата", "salary", hit,
                          "Оплата описана расплывчато, сумма не найдена автоматически.",
                          "Укажите конкретную сумму или вилку оплаты.", "высокая"))
        else:
            out.append(_r("Оплата", "salary", "", "Не удалось найти сумму оплаты в полях вакансии.",
                          "Проверьте и укажите сумму или вилку оплаты и что входит в неё.", "высокая"))
    elif smin is None and smax is None:
        lower = re.search(r"\bот\s+\d[\d \u00a0]*", stext, re.I)
        if lower and not re.search(r"\bдо\s+\d+", stext, re.I):
            out.append(_r("Оплата", "salary", lower.group().rstrip(),
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
    # Требования и обязанности иногда содержат дни/часы, а не только описание.
    sched_all = f"{sched} {desc} {duties} {req} {qualification}"
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
        elif hit and concrete:
            unspecified = re.search(r"гибк\w*\s+график[^.!?;]{0,60}?без\s+уточнен\w*\s+правил", sched_all, re.I)
            if unspecified:
                out.append(_r("График", "scheduleType", unspecified.group(),
                              "Указаны часы или дни, но правила гибкого графика не уточнены.",
                              "Уточните, какие часы или дни можно менять и как согласуется гибкий график.", "средняя"))

    # --- Место работы ---
    place = f"{v.get('workplace','')} {v.get('address','')} {v.get('city','')}"
    place_all = f"{place} {desc}"
    has_place = bool(place.strip()) or bool(re.search(r"калининград|удал[её]нн\w*|дистанционн\w*|на дому|адрес|ул\.|улица",
                                                      f"{place_all} {duties} {req} {qualification}", re.I))
    if not has_place:
        out.append(_r("Место работы", "workPlace", "", "Не указано место работы.",
                      "Укажите город/адрес или возможность удалённой работы.", "высокая"))
    else:
        hit = _has(_VAGUE_PLACE, place_all)
        if hit and not (v.get("city") or v.get("address")) and not re.search(r"калининград|ул\.|улица|адрес|удал[её]нн\w*|дистанционн\w*", place_all, re.I):
            out.append(_r("Место работы", "workPlace", hit,
                          "Место указано общо («в офисе») без города/адреса.",
                          "Укажите город и адрес (или район) места работы.", "средняя"))
        # Узкое сопоставление явной полностью удалённой работы с обязательным
        # присутствием: обе цитаты оставляем в исходных полях, без вывода о нарушениях.
        format_text = f"{desc} {duties} {req} {qualification}"
        full_remote = re.search(r"полностью\s+(?:дистанционн\w*|удал[её]нн\w*)", format_text, re.I)
        required_office = re.search(r"обязательн\w*\s+(?:ежедневн\w*\s+)?присутстви\w*\s+в\s+офисе", format_text, re.I)
        office_negated = required_office and (
            re.search(r"(?:не|без)\s+$", format_text[max(0, required_office.start()-15):required_office.start()], re.I)
            or re.match(r"\s+не\s+(?:требуется|нужно|обязательно)", format_text[required_office.end():], re.I))
        remote_negated = full_remote and (
            re.search(r"(?:не|без)\s+$", format_text[max(0, full_remote.start()-15):full_remote.start()], re.I)
            or re.match(r"\s+(?:работ\w*\s+)?не\s+(?:предусмотрен\w*|возможн\w*|допускается)",
                        format_text[full_remote.end():], re.I))
        if full_remote and required_office and not remote_negated and not office_negated:
            remote_quote, office_quote = full_remote.group(), required_office.group()
            remote_field = _source_for_quote(v, remote_quote, "description")
            office_field = _source_for_quote(v, office_quote, "description")
            fields = remote_field if remote_field == office_field else f"{remote_field} | {office_field}"
            out.append(_r("Место работы", fields, f"{remote_quote} | {office_quote}",
                          "Полностью дистанционная работа и обязательное присутствие в офисе могут не сочетаться.",
                          "Проверьте оба условия и уточните формат работы.", "средняя"))
        remote_match = re.search(r"удал[её]нн\w*|дистанционн\w*", place_all, re.I)
        office_match = re.search(r"в офисе|на предприятии", place_all, re.I)
        negated_remote = remote_match and (
            re.search(r"(?:не|без)\s+(?:возможн\w*\s+)?$", place_all[max(0, remote_match.start() - 25):remote_match.start()], re.I)
            or re.match(r"\s+работ\w*\s+не\s+(?:предусмотрен\w*|возможн\w*)", place_all[remote_match.end():], re.I))
        if not (full_remote and required_office and not remote_negated and not office_negated) and remote_match and office_match and not negated_remote and not re.search(r"удал[её]нн\w*|дистанционн\w*", place, re.I):
            remote = remote_match.group()
            office = office_match.group()
            remote_field = _source_for_quote(v, remote, "description")
            office_field = _source_for_quote(v, office, "workPlace")
            fields = remote_field if remote_field == office_field else f"{remote_field} | {office_field}"
            out.append(_r("Место работы", fields, f"{remote} | {office}",
                          "Указаны удалённая работа и работа в офисе, но порядок их сочетания неясен.",
                          "Уточните формат: удалённо, гибридно или в офисе (и в каком).", "средняя"))

    # --- Тип занятости ---
    emp = v.get("employment_type", "")
    text_employment = re.search(r"(?:полная|частичная|временная)\s+занятость", all_text, re.I)
    if not emp.strip() and not text_employment:
        out.append(_r("Тип занятости", "employmentType", "", "Не удалось найти тип занятости в проверенных полях.",
                      "Проверьте и уточните тип занятости: полная, частичная или другой вариант.", "средняя"))
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

    for remark in out:
        quote = remark["quote"]
        if quote and " | " not in quote:
            for field, value in _source_strings(v.get("raw", {})):
                match = re.search(re.escape(quote), value, re.I)
                if match:
                    remark["field"], remark["quote"] = field, match.group()
                    break
            else:
                # Синтетическая нормализованная строка не является цитатой источника.
                remark["quote"] = ""
    out.sort(key=lambda r: TOPIC_ORDER.index(r["topic"]) if r["topic"] in TOPIC_ORDER else 99)
    return out
