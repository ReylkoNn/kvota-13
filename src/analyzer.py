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


def analyze_vacancy(v: dict, ai_cfg: dict, ai_analyzer=None) -> tuple[list[dict], str]:
    """Вернуть (замечания, ai_status). ai_status: gemma-local|api|fallback|off."""
    remarks = analyze_rules(v)
    status = "off"
    if ai_cfg.get("enabled", True) and ai_cfg.get("provider") != "off":
        analyzer = ai_analyzer or build_ai_analyzer(ai_cfg)[0]
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
                status = "fallback"
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
