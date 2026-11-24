"""Оркестратор анализа: правила + ИИ, слияние и дедупликация.

ИИ влияет на отчёт: его замечания добавляются к rule-based (и наоборот,
дубли по теме+полю+цитате убираются). Если модель упала — fallback,
а не пустой результат.
"""
from __future__ import annotations

import logging

from .ai import FallbackAnalyzer, build_ai_analyzer
from .rules import TOPIC_ORDER, analyze_rules

log = logging.getLogger("kvota13.analyzer")


def _key(r: dict) -> tuple:
    return (r.get("topic", "").lower(), r.get("field", "").lower(),
            r.get("quote", "").lower()[:60], r.get("problem", "").lower()[:60])



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
                                     problem.startswith("Формулировка обязанностей")):
        choices = ("Перечислите основные задачи сотрудника.",
                   "Уточните основные задачи сотрудника и зоны его ответственности.")
    elif topic == "Требования" and (problem.startswith("Не указаны требования") or
                                       problem.startswith("Перечислены только личные качества") or
                                       problem.startswith("Не понятно, нужен ли опыт")):
        choices = ("Уточните обязательные навыки и требуемый опыт работы.",
                   "Уточните обязательные навыки и требования к образованию.")
    elif topic == "Оплата" and (problem.startswith("Не указана оплата") or
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
    elif topic == "Тип занятости" and problem.startswith("Не указан тип занятости"):
        choices = ("Укажите тип занятости: полная или частичная.",
                   "Уточните тип занятости и предполагаемую длительность работы.")
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
            for r in ai_remarks:
                if _key(r) not in seen:
                    remarks.append(r)
                    seen.add(_key(r))
        except Exception as exc:  # модель недоступна -> fallback, отчёт не теряем
            log.warning("ИИ недоступен (%s) — fallback", exc)
            try:
                fb = FallbackAnalyzer().analyze(v)
                seen = {_key(r) for r in remarks}
                for r in fb:
                    if _key(r) not in seen:
                        remarks.append(r)
                status = "mixed" if selected else "fallback"
            except Exception:
                status = "fallback-error"
    else:
        # Даже без модели fallback-лексикон дополняет правила
        try:
            fb = FallbackAnalyzer().analyze(v)
            seen = {_key(r) for r in remarks}
            for r in fb:
                if _key(r) not in seen:
                    remarks.append(r)
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
