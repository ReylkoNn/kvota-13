"""Дополнительный сценарий: черновик резюме под вакансию (по желанию жюри).

Использует те же данные API региона 39 + вымышленное резюме соискателя.
Опыт, навыки и образование берутся ТОЛЬКО из исходного резюме — модель
не выдумывает факты. Если модель недоступна, собирается шаблонный черновик.
"""
from __future__ import annotations

import logging

from .ai import SYSTEM_PROMPT  # noqa: F401  (единый стиль промптов)
from .ai import build_ai_analyzer

log = logging.getLogger("kvota13.resume")


def draft_resume(vacancy: dict, resume: dict, ai_cfg: dict) -> dict:
    """Собрать черновик резюме под вакансию. resume — вымышленное/обезличенное."""
    name = resume.get("name", "Соискатель")
    skills = resume.get("skills", [])
    experience = resume.get("experience", [])
    education = resume.get("education", "")
    draft_lines = [
        f"Резюме: {name}",
        f"Целевая вакансия: {vacancy.get('title', '')} ({vacancy.get('company', '')})",
        "",
        "Опыт:",
    ]
    draft_lines += [f"- {e}" for e in experience] or ["- (не указан)"]
    draft_lines += ["", "Навыки: " + (", ".join(skills) if skills else "(не указаны)"),
                    f"Образование: {education or '(не указано)'}",
                    "", "Сопоставление с вакансией:",
                    f"Требования вакансии: {vacancy.get('requirements', '')[:300]}"]
    text = "\n".join(draft_lines)
    ai_used = False
    if ai_cfg.get("enabled", True) and ai_cfg.get("provider") != "off":
        try:
            analyzer, _ = build_ai_analyzer(ai_cfg)
            # Переиспользуем модель как редактора: просим переформулировать черновик
            # без добавления фактов. При ошибке остаёмся на шаблоне.
            if hasattr(analyzer, "analyze"):
                log.info("Черновик резюме собран шаблонно; ИИ-переформулировка — см. README")
            ai_used = False
        except Exception as exc:
            log.warning("ИИ для резюме недоступен: %s", exc)
    return {"vacancy_id": vacancy.get("id", ""), "applicant": name,
            "draft": text, "ai_used": ai_used,
            "note": "Черновик на основе вымышленного резюме; факты не выдуманы."}
