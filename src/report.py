"""Сборка и сохранение JSON-отчёта.

Формат (см. README): {"meta": {...}, "vacancies": [{"id","url","title","remarks":[...]}]}.
Запись атомарная (tmp + rename). При ошибке загрузки отчёт НЕ трогаем.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("kvota13.report")


def build_report(results: list[dict], mode: str, source: str, ai_status: str) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "meta": {
            "generated_at": now,
            "mode": mode,
            "source": source,
            "ai_status": ai_status,
            "total": len(results),
            "with_remarks": sum(1 for r in results if r["remarks"]),
            "without_remarks": sum(1 for r in results if not r["remarks"]),
            "total_remarks": sum(len(r["remarks"]) for r in results),
            "ai_generated_remarks": sum(1 for v in results for r in v["remarks"]
                                        if r.get("source") == "ai"),
            "ai_selected_recommendations": sum(1 for v in results for r in v["remarks"]
                                               if r.get("recommendation_source") == "ai-template-choice"),
        },
        "vacancies": results,
    }


def save_report(report: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    log.info("Отчёт сохранён: %s (%d вакансий)", p, report["meta"]["total"])


def save_error(error: str, path: str | Path, source: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source, "error": error,
        "note": "Последний успешный отчёт не перезаписан.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
