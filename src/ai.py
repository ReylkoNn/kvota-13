"""ИИ-анализ вакансий (роль №2: бэкенд / ИИ — Gemma).

Шаг, на котором используется ИИ: после rule-based проверки модель Gemma
перечитывает вакансию и ищет неоднозначные формулировки, которые правила
не ловят, и предлагает конкретные уточнения по полям. Замечания ИИ
объединяются с правилами в analyzer.py и влияют на итоговый отчёт.

Режимы (config.yaml -> ai.provider / env AI_PROVIDER):
  local — модель через Transformers (для доступа к Gemma может потребоваться аккаунт);
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

from .normalize import _ALIASES
from .rules import TOPIC_ORDER

log = logging.getLogger("kvota13.ai")

SYSTEM_PROMPT = (
    "Ты редактор вакансий. Найди только конкретные неоднозначные формулировки "
    "в предоставленных полях, которые мешают понять условия работы. Сопоставь все поля "
    "между собой: если адрес, график, оплата или другие сведения уже указаны в другом поле, "
    "не утверждай, что они отсутствуют. Не придумывай противоречия, факты или требования. "
    "Пропуски и противоречия проверяют отдельные правила: не сообщай о них. "
    "Если формулировка понятна или ты сомневаешься, верни []. "
    "Для каждого замечания скопируй field точно из ключа переданного JSON, "
    "а quote — точную непустую подстроку именно значения этого поля. "
    "Укажи в problem, что непонятно именно в цитате, а в recommendation — конкретный вопрос "
    "или уточнение без добавления фактов. Верни только JSON-массив объектов: "
    '{"topic":"График","field":"description","quote":"...",'
    '"problem":"...","recommendation":"...","severity":"средняя"}. '
    "topic — одно из: Название, Обязанности, Требования, Оплата, График, "
    "Место работы, Тип занятости, Описание. "
    "Если обоснованных замечаний нет, верни []."
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


# Поддерживаем только поля вакансии, а не служебные ID, ссылки и данные работодателя.
_TOPIC_KEYS = {
    "Название": ("title",),
    "Обязанности": ("duties",),
    "Требования": ("requirements",),
    "Оплата": ("salary_text",),
    "График": ("schedule_type", "schedule_text"),
    "Место работы": ("workplace", "address", "city"),
    "Тип занятости": ("employment_type",),
    "Описание": ("description",),
}
_EXTRA_FIELDS = {"salaryMin", "salaryMax", "salary_min", "salary_max", "salary_from",
                 "salary_to", "salaryFrom", "salaryTo", "minSalary", "maxSalary",
                 "зарплатаОт", "зарплатаДо", "addresses"}


def _flatten_source(value, path: str, out: dict[str, str]) -> None:
    """Пути соответствуют исходному JSON, например addresses.address[0].location."""
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_source(item, f"{path}.{key}", out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_source(item, f"{path}[{index}]", out)
    elif isinstance(value, str) and value.strip():
        out[path] = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[path] = str(value)


def _source_fields(v: dict) -> dict[str, str]:
    fields: dict[str, str] = {}
    relevant = {key for group in _TOPIC_KEYS.values() for canon in group
                for key in _ALIASES[canon]} | set(_ALIASES["description"]) | _EXTRA_FIELDS
    for key, value in v.get("raw", {}).items():
        if key in relevant:
            _flatten_source(value, key, fields)
    return fields


def _vacancy_text(v: dict) -> str:
    # Модель видит точные имена полей и исходные значения, не синтетические метки.
    return json.dumps(_source_fields(v), ensure_ascii=False, indent=2)


def _valid_remark(r: dict, v: dict) -> dict | None:
    """Консервативный фильтр: привязка к источнику, а не проверка смысла ИИ."""
    if not isinstance(r, dict):
        return None
    if any(not isinstance(r.get(k), str) for k in ("topic", "field", "quote", "problem", "recommendation")):
        return None
    topic, field, quote, problem, rec = (r[k].strip() for k in
                                        ("topic", "field", "quote", "problem", "recommendation"))
    fields = _source_fields(v)
    if topic not in TOPIC_ORDER or not field or not quote or not problem or not rec:
        return None
    if field not in fields or quote not in fields[field]:
        return None
    # Противоречия/пропуски оставлены правилам: модель не всегда приводит оба значения.
    if re.search(r"противореч|несовместим", problem + " " + rec, re.I):
        return None
    root = re.split(r"[.\[]", field, maxsplit=1)[0]
    allowed = {alias for canon in _TOPIC_KEYS[topic] for alias in _ALIASES[canon]}
    allowed.update(_ALIASES["description"])
    if topic == "Оплата":
        allowed.update(_EXTRA_FIELDS - {"addresses"})
    if topic == "Место работы":
        allowed.add("addresses")
    if root not in allowed:
        return None
    sev = str(r.get("severity", "средняя")).lower()
    if sev not in ("высокая", "средняя", "низкая"):
        sev = "средняя"
    return {"topic": topic, "field": field, "quote": quote,
            "problem": problem, "recommendation": rec, "severity": sev, "source": "ai"}


def _accepted_remarks(text: str, v: dict) -> list[dict]:
    candidates = _parse_model_json(text)
    remarks = [valid for r in candidates if (valid := _valid_remark(r, v)) is not None]
    log.info("Ответ модели по вакансии %s: предложено %d, принято %d", v.get("id", ""),
             len(candidates), len(remarks))
    return remarks


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
    """Локальная модель через Transformers; доступ к весам определяется поставщиком."""

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
        # Используем уже загруженный токенизатор, не проверяя сеть для каждой вакансии.
        tokenizer = getattr(pipe, "tokenizer", None)
        try:
            if tokenizer is None:
                raise ValueError("Токенизатор отсутствует")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Gemma не всегда поддерживает system role; объединяем инструкции с user.
            merged = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + messages[1]["content"]}]
            try:
                if tokenizer is None:
                    raise ValueError("Токенизатор отсутствует")
                prompt = tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = merged[0]["content"]
        result = pipe(prompt, max_new_tokens=self.max_new_tokens,
                      do_sample=self.temperature > 0,
                      temperature=self.temperature if self.temperature > 0 else None,
                      return_full_text=False)
        generated = result[0]["generated_text"] if result else ""
        return _accepted_remarks(generated, v)


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
        return _accepted_remarks(content, v)


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
