"""Адаптер командного конвейера к анализатору с сохранением исходных полей."""
import logging
from pathlib import Path

from src.analyzer import analyze_batch
from src.normalize import normalize_vacancy

from .ai_cache import CachedAnalyzer

log = logging.getLogger(__name__)


def analyze_raw_vacancies(raw_data, ai_config, mode='check', cache_config=None, protected_paths=()):
    # Не используем здесь app.normalize: он теряет исходные поля для цитат ИИ.
    # Проверочный режим должен анализировать все записи, включая повторные ID.
    vacancies = [normalize_vacancy(raw) for raw in raw_data]
    cache_config = cache_config or {}
    wrapper = None
    if (mode == 'api' and ai_config.get('enabled', True) and ai_config.get('provider') != 'off'
            and cache_config.get('enabled', True) and not ai_config.get('debug_dir')):
        path = Path(cache_config.get('path', 'reports/ai-cache-v1.json'))
        max_entries = cache_config.get('max_entries', 2000)
        if path.resolve() in {Path(p).resolve() for p in protected_paths if p}:
            log.warning('Путь кэша совпадает с отчётом/файлом ошибки: кэш отключён')
        elif not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            log.warning('cache.max_entries должен быть положительным целым числом: кэш отключён')
        else:
            wrapper = lambda analyzer: CachedAnalyzer(analyzer, ai_config, path, max_entries)
    return analyze_batch(vacancies, ai_config, analyzer_wrapper=wrapper)
