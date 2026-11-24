"""Оркестратор анализа: правила + ИИ, слияние и дедупликация.

ИИ влияет на отчёт: его замечания добавляются к rule-based (и наоборот,
дубли по теме+полю+цитате убираются). Если модель упала — fallback,
а не пустой результат.
"""
from __future__ import annotations

import logging
import re

from .ai import FallbackAnalyzer, build_ai_analyzer
from .rules import TOPIC_ORDER, analyze_rules

log = logging.getLogger("kvota13.analyzer")


def _key(r: dict) -> tuple:
    """Объединяем точные повторы по теме, полю и цитате; пропуски различаем по проблеме."""
    clean = lambda value: " ".join(str(value or "").casefold().split())
    quote = clean(r.get("quote"))
    return (clean(r.get("topic")), clean(r.get("field")), quote,
            "" if quote else clean(r.get("problem")))


def _ai_content_key(r: dict) -> tuple:
    """Полный повтор ответа ИИ по одному фрагменту, даже при разных темах."""
    clean = lambda value: " ".join(str(value or "").casefold().split())
    return tuple(clean(r.get(name)) for name in ("field", "quote", "problem", "recommendation"))


def _generic_missing_duplicate(r: dict, rules: list[dict]) -> bool:
    """Узкий случай: общая просьба об обязанностях и требованиях уже есть в правилах.

    Не предполагает общую семантическую дедупликацию и не отбрасывает конкретные
    уточнения об опыте, навыках или отдельных задачах.
    """
    if r.get("source") != "ai":
        return False
    missing = {x.get("topic") for x in rules if x.get("source") == "rule" and not x.get("quote")
               and str(x.get("problem", "")).startswith(("Не удалось найти описание обязанностей",
                                                          "Не удалось найти требования"))}
    if not {"Обязанности", "Требования"} <= missing:
        return False
    problem = str(r.get("problem", ""))
    recommendation = str(r.get("recommendation", ""))
    problem_pattern = (r"\s*(?:необходимо|нужно|следует)\s+уточнить,?\s+какие\s+именно\s+"
                       r"обязанности\s+и\s+требования(?:\s+(?:от|к|для)\s+[^.,;!?]{1,80})?\s*[.!?]*\s*")
    recommendation_pattern = (r"\s*уточните\s+обязанности\s+и\s+требования"
                              r"(?:\s+(?:от|к|для)\s+[^.,;!?]{1,80})?\s*[.!?]*\s*")
    return bool(re.fullmatch(problem_pattern, problem, re.I)
                and re.fullmatch(recommendation_pattern, recommendation, re.I))


def _repeats_experience_question(r: dict, rules: list[dict]) -> bool:
    """Не дублировать уже обоснованное двумя полями общее уточнение об опыте."""
    if (r.get("source") != "ai" or r.get("topic") != "Требования"
            or r.get("field") != "requirement.qualification"):
        return False
    rec = str(r.get("recommendation", ""))
    if not re.fullmatch(r"\s*уточните,?\s+требуется\s+ли\s+опыт\s+работы(?:\s+по\s+[^.!?]{1,100})?\s*[.!?]*\s*", rec, re.I):
        return False
    quote = str(r.get("quote", "")).casefold()
    for x in rules:
        if (x.get("source") == "rule" and x.get("topic") == "Требования"
                and x.get("field") == "requirement.qualification | requirement.experience"
                and str(x.get("problem", "")).startswith("В тексте опыт назван желательным")):
            preferred = str(x.get("quote", "")).split(" | ", 1)[0].casefold()
            if preferred and preferred in quote:
                return True
    return False


def _repeats_vague_duties(r: dict, rules: list[dict]) -> bool:
    """Не повторяем общий вопрос о должностных обязанностях из правила.

    Отбрасываем только замечание о том же поле и фрагменте. Это намеренно
    консервативно: если в одной фразе есть ещё один вопрос, он тоже может исчезнуть.
    """
    if r.get("source") != "ai" or r.get("topic") != "Обязанности":
        return False
    text = f"{r.get('problem', '')} {r.get('recommendation', '')}"
    if not re.search(r"должностн\w*\s+обязанност\w*", text, re.I):
        return False
    quote = str(r.get("quote", "")).casefold()
    return any(x.get("source") == "rule" and x.get("topic") == "Обязанности"
               and x.get("field") == r.get("field") and x.get("quote")
               and str(x.get("quote")).casefold() in quote
               and str(x.get("problem", "")).startswith("Формулировка обязанностей")
               for x in rules)


def _merge_fallback(remarks: list[dict], v: dict) -> None:
    """Добавить fallback, не повторяя вопрос о сумме, уже заданный правилом."""
    seen = {_key(r) for r in remarks}
    pay_covered = any(r.get("source") == "rule" and r.get("topic") == "Оплата" and
                      str(r.get("problem", "")).startswith(("Не указана оплата", "Не удалось найти сумму оплаты",
                                                            "Оплата описана расплывчато"))
                      for r in remarks)
    for r in FallbackAnalyzer().analyze(v):
        if pay_covered and r.get("topic") == "Оплата":
            continue
        key = _key(r)
        if key not in seen:
            remarks.append(r)
            seen.add(key)


def _question_options(r: dict) -> dict[str, str] | None:
    """Два редакторских вопроса только для известных срабатываний правил.

    Оба варианта - уточнения, не факты о вакансии. Противоречия намеренно
    не меняем: для них нужно сохранить рекомендации о проверке двух значений.
    """
    if r.get("source") != "rule":
        return None
    topic, problem = r.get("topic", ""), r.get("problem", "")
    if "Противоречие" in problem:
        return None
    choices = None
    if topic == "Обязанности" and (problem.startswith("Не описаны обязанности") or
                                     problem.startswith("Не удалось найти описание обязанностей") or
                                     problem.startswith("Формулировка обязанностей")):
        choices = ("Перечислите основные задачи сотрудника.",
                   "Уточните основные задачи сотрудника и зоны его ответственности.")
    elif topic == "Требования" and (problem.startswith("Не указаны требования") or
                                       problem.startswith("Не удалось найти требования") or
                                       problem.startswith("Указаны личные качества") or
                                       problem.startswith("Перечислены только личные качества") or
                                       problem.startswith("Не понятно, нужен ли опыт")):
        choices = ("Уточните обязательные навыки и требуемый опыт работы.",
                   "Уточните обязательные навыки и требования к образованию.")
    elif topic == "Оплата" and (problem.startswith("Не указана оплата") or
                                 problem.startswith("Не удалось найти сумму оплаты") or
                                 problem.startswith("Оплата описана расплывчато")):
        choices = ("Укажите конкретную сумму или вилку оплаты.",
                   "Укажите сумму оплаты и условия, от которых зависит её размер.")
    elif topic == "Оплата" and problem.startswith("Указан только нижний порог"):
        choices = ("Уточните верхнюю границу оплаты или вилку дохода.",
                   "Уточните, от каких условий зависит оплата выше указанного минимума.")
    elif topic == "График" and (problem.startswith("Не указан график") or
                                 problem.startswith("Формулировка «")):
        choices = ("Укажите дни недели, часы работы и продолжительность смен.",
                   "Уточните дни и часы смен и порядок формирования графика.")
    elif topic == "Место работы" and (problem.startswith("Не указано место") or
                                       problem.startswith("Место указано общо")):
        choices = ("Укажите город и адрес или район места работы.",
                   "Уточните город, место и формат работы: в офисе или удалённо.")
    elif topic == "Тип занятости" and (problem.startswith("Не указан тип занятости") or
                                        problem.startswith("Не удалось найти тип занятости")):
        choices = ("Проверьте и укажите тип занятости: полная или частичная.",
                   "Проверьте и уточните тип занятости и предполагаемую длительность работы.")
    return dict(zip(("A", "B"), choices)) if choices else None


def _apply_ai_question_choice(v: dict, remarks: list[dict], analyzer) -> bool:
    """Модель выбирает один шаблон; это не представляется как сгенерированное замечание."""
    choose = getattr(analyzer, "choose_option", None)
    if not callable(choose):
        return False
    for r in remarks:
        options = _question_options(r)
        if not options:
            continue
        try:
            choice = choose(v, r, options)
        except Exception as exc:
            log.warning("ИИ не смог выбрать уточнение (%s); оставлена рекомендация правила", exc)
            return False
        if not isinstance(choice, str) or choice not in options:
            log.warning("ИИ не вернул однозначный выбор A/B; оставлена рекомендация правила")
            return False
        if options[choice] == r["recommendation"]:
            return False
        r["recommendation_original"] = r["recommendation"]
        r["recommendation"] = options[choice]
        r["recommendation_source"] = "ai-template-choice"
        r["recommendation_choice"] = choice
        model_name = getattr(analyzer, "model_name", None) or getattr(analyzer, "model", None)
        if model_name:
            r["recommendation_model"] = model_name
        log.info("ИИ выбрал вариант уточнения %s для вакансии %s (тема: %s)",
                 choice, v.get("id", ""), r.get("topic", ""))
        return True
    return False

def analyze_vacancy(v: dict, ai_cfg: dict, ai_analyzer=None) -> tuple[list[dict], str]:
    """Вернуть (замечания, ai_status). ai_status: gemma-local|api|fallback|off."""
    remarks = analyze_rules(v)
    status = "off"
    if ai_cfg.get("enabled", True) and ai_cfg.get("provider") != "off":
        analyzer = ai_analyzer or build_ai_analyzer(ai_cfg)[0]
        # Выбор A/B независим от свободной генерации JSON: если вторая часть
        # не удалась, он остаётся отражён в отчёте, а статус становится mixed.
        selected = (getattr(analyzer, "name", "") != "fallback" and
                    _apply_ai_question_choice(v, remarks, analyzer))
        try:
            ai_remarks = analyzer.analyze(v)
            status = getattr(analyzer, "name", "ai")
            seen = {_key(r) for r in remarks}
            seen_ai_content = set()
            for r in ai_remarks:
                content_key = _ai_content_key(r) if r.get("source") == "ai" else None
                if (_key(r) in seen or (content_key and content_key in seen_ai_content)
                        or _generic_missing_duplicate(r, remarks) or _repeats_vague_duties(r, remarks)
                        or _repeats_experience_question(r, remarks)):
                    log.info("Повтор замечания ИИ пропущен для вакансии %s", v.get("id", ""))
                    continue
                remarks.append(r)
                seen.add(_key(r))
                if content_key:
                    seen_ai_content.add(content_key)
        except Exception as exc:  # модель недоступна -> fallback, отчёт не теряем
            log.warning("ИИ недоступен (%s) — fallback", exc)
            try:
                _merge_fallback(remarks, v)
                status = "mixed" if selected else "fallback"
            except Exception:
                status = "fallback-error"
    else:
        # Даже без модели fallback-лексикон дополняет правила
        try:
            _merge_fallback(remarks, v)
            status = "fallback"
        except Exception:
            status = "off"
    remarks.sort(key=lambda r: TOPIC_ORDER.index(r["topic"]) if r["topic"] in TOPIC_ORDER else 99)
    return remarks, status


def analyze_batch(vacancies: list[dict], ai_cfg: dict) -> tuple[list[dict], str]:
    """Анализ списка вакансий. Возвращает (результаты для отчёта, общий ai_status)."""
    analyzer, _ = build_ai_analyzer(ai_cfg) if ai_cfg.get("enabled", True) else (None, "off")
    max_ai = int(ai_cfg.get("max_vacancies_for_ai", 40))
    results, statuses = [], []
    for i, v in enumerate(vacancies):
        log.info("Анализ вакансии %d/%d (ID: %s)", i + 1, len(vacancies), v.get("id", ""))
        cfg = dict(ai_cfg)
        if i >= max_ai:  # защита от долгого ожидания на CPU
            cfg = {**ai_cfg, "enabled": False}
        remarks, st = analyze_vacancy(v, cfg, ai_analyzer=analyzer)
        statuses.append(st)
        results.append({"id": v.get("id", ""), "url": v.get("url", "") or None,
                        "title": v.get("title", ""), "ai_status": st, "remarks": remarks})
    # Общий статус: если хоть где-то сработала модель — показываем её
    overall = statuses[0] if statuses and len(set(statuses)) == 1 else ("mixed" if statuses else "off")
    return results, overall
