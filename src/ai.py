"""ИИ-анализ вакансий (роль №2: бэкенд / ИИ — Gemma).

Шаг, на котором используется ИИ: после rule-based проверки модель Gemma
перечитывает вакансию и ищет неоднозначные формулировки, которые правила
не ловят, и предлагает конкретные уточнения по полям. Замечания ИИ
объединяются с правилами в analyzer.py и влияют на итоговый отчёт.

Режимы (config.yaml -> ai.provider / env AI_PROVIDER):
  local — Gemma локально через Transformers (бесплатно, без аккаунта);
  api   — OpenAI-совместимый API (бесплатные OpenRouter/Groq/AI Studio);
  off   — без модели, только fallback-шаблоны.

Fallback: если модель недоступна (нет transformers, не скачалась, таймаут,
невалидный JSON) — эвристический анализатор на лексиконе неоднозначностей.
Он тоже влияет на отчёт, а meta.ai_status честно показывает, что сработало.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

log = logging.getLogger("kvota13.ai")

SYSTEM_PROMPT = (
    "Ты редактор вакансий. Прочитай вакансию и найди формулировки, из-за которых "
    "соискатель не понимает условия работы: пропуски, неоднозначности, противоречия. "
    "Проверь название, обязанности, требования, оплату, график, место работы, тип занятости. "
    "Правила: не выдумывай факты; не делай выводов о нарушении закона; цитата quote должна быть "
    "точной подстрокой из текста вакансии, а для пропуска quote пустой; для противоречия quote "
    "содержит оба значения через ' | '. "
    "Верни ТОЛЬКО JSON-массив объектов: "
    '{"topic": "...", "field": "...", "quote": "...", "problem": "...", "recommendation": "...", "severity": "высокая|средняя|низкая"}. '
    "topic — одно из: Название, Обязанности, Требования, Оплата, График, Место работы, Тип занятости, Описание. "
    "field — имя поля источника (например vacancyName, duties, requirements, salary, scheduleType, workPlace, employmentType, description). "
    "Если проблем нет, верни []."
)

# Лексикон неоднозначных формулировок для fallback (и как подсказка модели не нужен)
_AMBIGUOUS = [
    ("стабильная зарплата", "Оплата", "salary", "«Стабильная зарплата» не говорит о сумме.",
     "Укажите сумму или вилку оплаты."),
    ("дружный коллектив", "Описание", "description", "«Дружный коллектив» не раскрывает условия работы.",
     "Опишите задачи, график и условия вместо общих похвал."),
    ("интересные задачи", "Обязанности", "duties", "«Интересные задачи» не показывает реальный объём работы.",
     "Перечислите конкретные задачи."),
    ("возможность роста", "Описание", "description", "«Возможность роста» без условий вводит в заблуждение.",
     "Укажите конкретные условия роста или уберите фразу."),
    ("своевременная оплата", "Оплата", "salary", "«Своевременная оплата» не говорит о сумме.",
     "Укажите сумму или вилку оплаты."),
    ("полный соцпакет", "Описание", "description", "«Полный соцпакет» не раскрывает, что именно входит.",
     "Перечислите, что входит в соцпакет."),
    ("официальное трудоустройство", "Тип занятости", "employmentType", "Не указан тип занятости и форма договора.",
     "Укажите тип занятости и форму оформления."),
]


def _vacancy_text(v: dict) -> str:
    parts = []
    for label, key in [("Название", "title"), ("Компания", "company"), ("Описание", "description"),
                       ("Обязанности", "duties"), ("Требования", "requirements"),
                       ("Оплата (текст)", "salary_text"), ("График", "schedule_type"),
                       ("Тип занятости", "employment_type"), ("Место работы", "workplace"),
                       ("Адрес", "address"), ("Город", "city")]:
        if v.get(key):
            parts.append(f"{label}: {v[key]}")
    if v.get("salary_min") is not None or v.get("salary_max") is not None:
        parts.append(f"Оплата (поля): от {v.get('salary_min')} до {v.get('salary_max')}")
    return "\n".join(parts)


def _valid_remark(r: dict, v: dict) -> dict | None:
    if not isinstance(r, dict):
        return None
    topic = str(r.get("topic", "")).strip()
    field = str(r.get("field", "")).strip()
    quote = str(r.get("quote", "")).strip()
    problem = str(r.get("problem", "")).strip()
    rec = str(r.get("recommendation", "")).strip()
    if not topic or not problem or not rec:
        return None
    # Цитата обязана быть из источника (кроме пропуска и противоречия с разделителем)
    if quote and any(part not in v.get("all_text", "") for part in quote.split(" | ")):
        return None
    sev = str(r.get("severity", "средняя")).lower()
    if sev not in ("высокая", "средняя", "низкая"):
        sev = "средняя"
    return {"topic": topic, "field": field or "description", "quote": quote,
            "problem": problem, "recommendation": rec, "severity": sev, "source": "ai"}


def _parse_model_json(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("Модель не вернула JSON-массив")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("Модель вернула некорректный JSON") from exc
    if not isinstance(data, list):
        raise ValueError("Модель вернула не массив")
    return data


class FallbackAnalyzer:
    """Эвристический fallback: лексикон + шаблоны уточнений. Работает без модели."""

    name = "fallback"

    def analyze(self, v: dict) -> list[dict]:
        out = []
        raw = v.get("raw", {})
        for phrase, topic, field, problem, rec in _AMBIGUOUS:
            for actual_field, value in raw.items():
                if not isinstance(value, str):
                    continue
                match = re.search(re.escape(phrase), value, re.I)
                if not match:
                    continue
                if topic == "Оплата" and (v.get("salary_min") or v.get("salary_max") or re.search(r"\d", v.get("salary_text", ""))):
                    continue
                out.append({"topic": topic, "field": actual_field, "quote": match.group(),
                            "problem": problem, "recommendation": rec,
                            "severity": "средняя", "source": "fallback"})
        return out


class GemmaLocalAnalyzer:
    """Gemma через Transformers (локально, без аккаунта и ключей)."""

    name = "gemma-local"

    def __init__(self, model: str, max_new_tokens: int = 512, temperature: float = 0.3):
        self.model_name = model
        self.name = "gemma-local" if "gemma" in model.lower() else "transformers-local"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline  # noqa: F401  (может отсутствовать -> fallback)
        import torch  # noqa: F401
        log.info("Загружаю модель %s (первый запуск скачает веса)...", self.model_name)
        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        return self._pipe

    def analyze(self, v: dict) -> list[dict]:
        pipe = self._load()
        text = _vacancy_text(v)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Вакансия:\n{text}\n\nВерни JSON-массив замечаний."},
        ]
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_name)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\nВакансия:\n{text}\n\nВерни JSON-массив замечаний."
        result = pipe(prompt, max_new_tokens=self.max_new_tokens,
                      do_sample=self.temperature > 0,
                      temperature=self.temperature if self.temperature > 0 else None,
                      return_full_text=False)
        generated = result[0]["generated_text"] if result else ""
        remarks = []
        for r in _parse_model_json(generated):
            ok = _valid_remark(r, v)
            if ok:
                remarks.append(ok)
        return remarks


class ApiAnalyzer:
    """OpenAI-совместимый API (OpenRouter / Groq / Google AI Studio — есть бесплатные)."""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def analyze(self, v: dict) -> list[dict]:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Вакансия:\n{_vacancy_text(v)}\n\nВерни JSON-массив замечаний."},
            ],
            "temperature": 0.3,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        remarks = []
        for r in _parse_model_json(content):
            ok = _valid_remark(r, v)
            if ok:
                remarks.append(ok)
        return remarks


def build_ai_analyzer(ai_cfg: dict):
    """Собрать анализатор по конфигу. Возвращает (analyzer, status_str)."""
    if not ai_cfg.get("enabled", True) or ai_cfg.get("provider") == "off":
        return FallbackAnalyzer(), "fallback (ai off)"
    provider = str(ai_cfg.get("provider", "local")).lower()
    if provider == "api":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            log.warning("OPENAI_API_KEY не задан — использую fallback")
            return FallbackAnalyzer(), "fallback (нет ключа API)"
        return ApiAnalyzer(os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
                           key, os.getenv("OPENAI_MODEL", "google/gemma-2-9b-it:free")), "api"
    return GemmaLocalAnalyzer(ai_cfg.get("model", "google/gemma-2-2b-it"),
                              int(ai_cfg.get("max_new_tokens", 512)),
                              float(ai_cfg.get("temperature", 0.3))), "gemma-local"
