"""Загрузка конфигурации: config.yaml (+ переопределения из .env / окружения)."""
from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "source_url": "http://opendata.trudvsem.ru/api/v1/vacancies/region/3900000000000",
    "region": 39,
    "pages": 2,
    "page_size": 20,
    "max_vacancies": 40,
    "timeout_sec": 30,
    "schedule_cron": "0 9 * * *",
    "output_report": "report.json",
    "error_report": "report_error.json",
    "ai": {
        "enabled": True,
        "provider": "local",
        "model": "google/gemma-2-2b-it",
        "max_new_tokens": 512,
        "temperature": 0.3,
        "max_vacancies_for_ai": 40,
    },
}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None) -> dict:
    _load_dotenv(ROOT / ".env")
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    p = Path(path) if path else (ROOT / "config.yaml")
    if p.exists():
        if yaml is None:
            raise RuntimeError("Нужен PyYAML: pip install pyyaml")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    # Переопределения из окружения (удобно жюри без правки yaml)
    if os.getenv("GEMMA_MODEL"):
        cfg["ai"]["model"] = os.getenv("GEMMA_MODEL")
    if os.getenv("AI_PROVIDER"):
        cfg["ai"]["provider"] = os.getenv("AI_PROVIDER").lower()
    if os.getenv("SOURCE_URL"):
        cfg["source_url"] = os.getenv("SOURCE_URL")
    return cfg
