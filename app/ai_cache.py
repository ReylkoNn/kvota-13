"""Кэш успешных вызовов модели для рабочего режима.

Ключ содержит исходную вакансию, настройки модели и версию кода анализа.
В файле хранятся только хеши ключей и ответы модели; ответы могут содержать
фрагменты вакансий. Старый reports/cache.json с заглушкой не используется.
"""
import copy
import hashlib
import json
import logging
import os
from pathlib import Path

from src.ai import _valid_remark

log = logging.getLogger(__name__)
CACHE_FORMAT = 'kvota-ai-cache-v1'
ROOT = Path(__file__).resolve().parent.parent


def _code_hash():
    digest = hashlib.sha256()
    for name in ('src/ai.py', 'src/analyzer.py', 'src/rules.py', 'src/normalize.py', 'app/ai_cache.py'):
        digest.update(name.encode('utf-8'))
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


class CachedAnalyzer:
    def __init__(self, analyzer, ai_cfg, path, max_entries=2000):
        self.inner = analyzer
        self.name = analyzer.name
        self.model_name = getattr(analyzer, 'model_name', None)
        self.model = getattr(analyzer, 'model', None)
        self.path = Path(path)
        self.max_entries = max_entries
        self.entries = {}
        self.pending = {}
        self.hits = 0
        self.identity = {
            'format': CACHE_FORMAT,
            'code': _code_hash(),
            'provider': self.name,
            'model': self.model_name or self.model,
            'base_url': getattr(analyzer, 'base_url', None),
            'max_new_tokens': ai_cfg.get('max_new_tokens', 512),
            'temperature': ai_cfg.get('temperature', 0.3),
        }
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if isinstance(data, dict) and data.get('format') == CACHE_FORMAT and isinstance(data.get('entries'), dict):
                self.entries = data['entries']
            else:
                log.warning('Неизвестный формат кэша %s: будет создан новый', self.path)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.warning('Кэш недоступен или повреждён (%s); выполняем анализ заново', exc)

    def _key(self, stage, v, extra=None):
        payload = {'identity': self.identity, 'stage': stage, 'vacancy': v.get('raw', {}), 'extra': extra}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()

    def begin_vacancy(self):
        self.pending = {}
        self._start_hits = self.hits

    def finish_vacancy(self, success):
        used_cache = self.hits > self._start_hits
        if success and self.pending:
            self.entries.update(self.pending)
            if len(self.entries) > self.max_entries:
                self.entries = dict(list(self.entries.items())[-self.max_entries:])
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + '.tmp')
                tmp.write_text(json.dumps({'format': CACHE_FORMAT, 'entries': self.entries},
                                          ensure_ascii=False, indent=2), encoding='utf-8')
                os.replace(tmp, self.path)
            except (OSError, ValueError, TypeError) as exc:
                log.warning('Не удалось сохранить кэш модели: %s', exc)
        self.pending = {}
        return used_cache

    def choose_option(self, v, remark, options):
        choose = getattr(self.inner, 'choose_option', None)
        if not callable(choose):
            return None
        key = self._key('choice', v, {'remark': remark, 'options': options})
        value = self.entries.get(key)
        if isinstance(value, str) and value in options:
            self.hits += 1
            log.info('Использован кэш выбора ИИ для вакансии %s', v.get('id', ''))
            return value
        value = choose(v, remark, options)
        if isinstance(value, str) and value in options:
            self.pending[key] = value
        return value

    def analyze(self, v):
        key = self._key('remarks', v)
        value = self.entries.get(key)
        if isinstance(value, list):
            checked = [_valid_remark(item, v) for item in value]
            if all(item is not None for item in checked):
                self.hits += 1
                log.info('Использован кэш замечаний ИИ для вакансии %s', v.get('id', ''))
                return checked
        result = self.inner.analyze(v)
        if isinstance(result, list):
            self.pending[key] = copy.deepcopy(result)
        return result
