import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import unittest
from pathlib import Path
from unittest.mock import patch

from app.integration import analyze_raw_vacancies
from src.normalize import normalize_vacancy
from src.rules import analyze_rules


SAMPLE = Path(__file__).resolve().parent.parent / 'data/kantkod_sample_vacancies.json'


def raw_example():
    return {'id': 'cache-1', 'description': 'Иногда возможны командировки. ' + 'x' * 100,
            'duty': 'Работа с документами', 'schedule': 'Гибкий график'}


class FakeModel:
    name = 'gemma-local'
    model_name = 'test-gemma'

    def __init__(self, fail=False):
        self.analyze_calls = 0
        self.choice_calls = 0
        self.fail = fail

    def choose_option(self, v, remark, options):
        self.choice_calls += 1
        return 'B'

    def analyze(self, v):
        self.analyze_calls += 1
        if self.fail:
            raise ValueError('bad model output')
        return [{'topic': 'Описание', 'field': 'description', 'quote': 'возможны командировки',
                 'problem': 'Неизвестна частота', 'recommendation': 'Уточните частоту командировок.',
                 'severity': 'средняя', 'source': 'ai'}]


class TeamCacheAndSampleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache_path = Path(self.temp.name) / 'ai-cache-v1.json'
        self.cache = {'enabled': True, 'path': str(self.cache_path), 'max_entries': 2000}
        self.cfg = {'enabled': True, 'provider': 'local', 'model': 'test-gemma'}

    def run_model(self, model, raw=None, cfg=None, mode='api', cache=None):
        with patch('src.analyzer.build_ai_analyzer', return_value=(model, model.name)):
            return analyze_raw_vacancies([raw or raw_example()], cfg or self.cfg, mode,
                                         cache or self.cache)

    def test_rules_match_official_sample_004_and_008(self):
        payload = json.loads(SAMPLE.read_text(encoding='utf8'))
        cases = {v['vacancy']['id']: v['vacancy'] for v in payload['results']['vacancies']}
        rules004 = analyze_rules(normalize_vacancy(cases['DEMO-39-004']))
        conflict = next(r for r in rules004 if r['topic'] == 'Место работы' and ' | ' in r['quote'])
        self.assertEqual(conflict['field'], 'duty | requirement.qualification')
        for field, quote in zip(conflict['field'].split(' | '), conflict['quote'].split(' | ')):
            value = cases['DEMO-39-004']
            for part in field.split('.'):
                value = value[part]
            self.assertIn(quote, value)
        rules008 = analyze_rules(normalize_vacancy(cases['DEMO-39-008']))
        schedule = [r for r in rules008 if r['topic'] == 'График']
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]['field'], 'requirement.qualification')
        self.assertIn('правила гибкого графика', schedule[0]['problem'])
        self.assertNotIn('дни и смены не понятны', schedule[0]['problem'])

    def test_no_false_office_contradiction_when_presence_is_not_required(self):
        v = normalize_vacancy({'id': 'x', 'duty': 'Работа полностью дистанционная.',
                               'requirement': {'qualification': 'Обязательное присутствие в офисе не требуется.'},
                               'addresses': {'address': [{'location': 'Калининград'}]}})
        self.assertFalse(any('дистанционная работа и обязательное' in r['problem'].lower() for r in analyze_rules(v)))

    def test_dedup_keeps_distinct_quotes_and_problems(self):
        from app.report import deduplicate_issues
        base = {'topic': 'График', 'field': 'schedule', 'quote': 'Гибкий график',
                'problem': 'Неясны часы', 'recommendation': 'Уточните часы', 'source': 'rule'}
        other_quote = {**base, 'quote': 'Сменный график'}
        other_problem = {**base, 'problem': 'Неясны выходные', 'recommendation': 'Уточните выходные'}
        self.assertEqual(deduplicate_issues([base, {**base, 'source': 'ai'}, other_quote, other_problem]),
                         [base, other_quote, other_problem])

    def test_explicit_negated_remote_does_not_create_conflict(self):
        v = normalize_vacancy({'duty': 'Полностью дистанционная работа не предусмотрена.',
                               'requirement': {'qualification': 'Обязательное присутствие в офисе.'}})
        self.assertFalse(any('дистанционная работа и обязательное' in r['problem'].lower() for r in analyze_rules(v)))

    def test_conflict_detected_without_address(self):
        v = normalize_vacancy({'duty': 'Работа полностью дистанционная.',
                               'requirement': {'qualification': 'Обязательное ежедневное присутствие в офисе.'}})
        self.assertTrue(any(' | ' in r['quote'] and r['topic'] == 'Место работы' for r in analyze_rules(v)))

    def test_success_reused_from_cache_and_rules_run_on_both(self):
        first = FakeModel()
        out1, status1 = self.run_model(first)
        self.assertTrue(self.cache_path.exists())
        self.assertEqual(first.analyze_calls, 1)
        self.assertEqual(first.choice_calls, 1)
        second = FakeModel()
        out2, status2 = self.run_model(second)
        self.assertEqual(second.analyze_calls, 0)
        self.assertEqual(second.choice_calls, 0)
        self.assertTrue(out2[0]['ai_cache_hit'])
        self.assertEqual(out1[0]['remarks'], out2[0]['remarks'])
        self.assertEqual(status1, status2)
        self.assertTrue(any(r['source'] == 'rule' for r in out2[0]['remarks']))
        self.assertTrue(any(r['source'] == 'ai' for r in out2[0]['remarks']))

    def test_changed_vacancy_or_model_settings_miss(self):
        self.run_model(FakeModel())
        changed = raw_example()
        changed['description'] += ' Дополнительное условие.'
        model = FakeModel()
        self.run_model(model, raw=changed)
        self.assertEqual(model.analyze_calls, 1)
        model = FakeModel()
        self.run_model(model, cfg={**self.cfg, 'temperature': 0.1})
        self.assertEqual(model.analyze_calls, 1)

    def test_failed_model_does_not_cache_choice_or_fallback(self):
        model = FakeModel(fail=True)
        _, status = self.run_model(model)
        self.assertEqual(status, 'mixed')
        self.assertFalse(self.cache_path.exists())
        recovered = FakeModel()
        self.run_model(recovered)
        self.assertEqual(recovered.analyze_calls, 1)
        self.assertEqual(recovered.choice_calls, 1)

    def test_old_placeholder_cache_is_never_read(self):
        (Path(self.temp.name) / 'cache.json').write_text('{"x": []}', encoding='utf8')
        model = FakeModel()
        out, _ = self.run_model(model)
        self.assertEqual(model.analyze_calls, 1)
        self.assertFalse(out[0].get('ai_cache_hit', False))

    def test_check_ignores_cache(self):
        self.run_model(FakeModel())
        model = FakeModel()
        out, _ = self.run_model(model, mode='check')
        self.assertEqual(model.analyze_calls, 1)
        self.assertNotIn('ai_cache_hit', out[0])

    def test_invalid_cache_entry_is_not_accepted(self):
        self.run_model(FakeModel())
        data = json.loads(self.cache_path.read_text(encoding='utf8'))
        for key, value in data['entries'].items():
            if isinstance(value, list):
                data['entries'][key] = [{'topic': 'Описание', 'field': 'description',
                                        'quote': 'выдуманная цитата', 'problem': 'x', 'recommendation': 'y'}]
        self.cache_path.write_text(json.dumps(data), encoding='utf8')
        model = FakeModel()
        self.run_model(model)
        self.assertEqual(model.analyze_calls, 1)

    def test_cli_http_retry_two_pages_and_cache_across_runs(self):
        import main
        class Handler(BaseHTTPRequestHandler):
            calls = 0
            offsets = []
            def log_message(self, *args):
                pass
            def do_GET(self):
                Handler.calls += 1
                if Handler.calls == 1:
                    self.send_response(503)
                    self.end_headers()
                    return
                params = parse_qs(urlparse(self.path).query)
                offset = int(params['offset'][0])
                Handler.offsets.append(offset)
                raw = [dict(raw_example(), id=str(i)) for i in range(offset, offset + 2)]
                blob = json.dumps({'results': {'vacancies': [{'vacancy': v} for v in raw]}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(blob)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        output = Path(self.temp.name) / 'report.json'
        cfg = {'source': {'url': 'http://127.0.0.1:%d/vacancies' % server.server_port,
                          'page_size': 2, 'max_vacancies': 4, 'max_pages': 2,
                          'max_retries': 2, 'retry_delay_seconds': 0},
               'output': {'report_path': str(output)}, 'error_report': str(Path(self.temp.name)/'error.json'),
               'ai': self.cfg, 'cache': self.cache, 'schedule': {'every_hours': 24}}
        first = FakeModel()
        with patch('main.load_config', return_value=cfg), patch('main.time.sleep'), \
             patch('src.analyzer.build_ai_analyzer', return_value=(first, first.name)), \
             patch('sys.argv', ['main.py', 'run']):
            main.main()
        self.assertEqual(first.analyze_calls, 4)
        self.assertEqual(Handler.offsets, [0, 2])
        second = FakeModel()
        with patch('main.load_config', return_value=cfg), \
             patch('src.analyzer.build_ai_analyzer', return_value=(second, second.name)), \
             patch('sys.argv', ['main.py', 'run']):
            main.main()
        report = json.loads(output.read_text(encoding='utf8'))
        self.assertEqual(second.analyze_calls, 0)
        self.assertEqual(second.choice_calls, 0)
        self.assertEqual(report['ai_cache_hits'], 4)
        self.assertEqual(report['total'], 4)
        self.assertTrue(all(v['ai_cache_hit'] for v in report['vacancies']))
        self.assertEqual(Handler.offsets, [0, 2, 0, 2])

    def test_report_path_collision_disables_cache(self):
        model = FakeModel()
        with patch('src.analyzer.build_ai_analyzer', return_value=(model, model.name)):
            analyze_raw_vacancies([raw_example()], self.cfg, 'api', self.cache, [str(self.cache_path)])
        self.assertFalse(self.cache_path.exists())
        self.assertEqual(model.analyze_calls, 1)


if __name__ == '__main__':
    unittest.main()
