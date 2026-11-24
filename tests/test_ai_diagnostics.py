"""Диагностика и тестовый набор. Пайплайн здесь имитирует модель, не оценивает её качество."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai import ApiAnalyzer, _accepted_remarks, _parse_model_json
from src.main import main
from src.normalize import extract_list, normalize_vacancy
from src.rules import analyze_rules


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        self.v = normalize_vacancy({"id": "probe-1", "description": "Командировки по области возникают регулярно."})
        self.good = {"topic": "Описание", "field": "description",
                     "quote": "Командировки по области возникают регулярно",
                     "problem": "Неясна частота командировок.",
                     "recommendation": "Укажите, как часто происходят поездки.", "severity": "средняя"}

    def test_debug_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            _accepted_remarks(json.dumps([self.good], ensure_ascii=False), self.v)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_debug_explains_rejections_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            invalid = {**self.good, "field": "выдуманное поле"}
            text = json.dumps([self.good, invalid], ensure_ascii=False)
            self.assertEqual(len(_accepted_remarks(text, self.v, folder)), 1)
            self.assertEqual(len(_accepted_remarks('[]', self.v, folder)), 0)
            files = list(Path(folder).glob('*.json'))
            self.assertEqual(len(files), 2)
            docs = [json.loads(x.read_text(encoding='utf-8')) for x in files]
            doc = next(x for x in docs if x['raw_response'] == text)
            self.assertEqual(doc['vacancy_id'], 'probe-1')
            self.assertEqual(doc['input_fields']['description'], self.v['description'])
            self.assertIsNone(doc['parse_error'])
            self.assertTrue(doc['candidates'][0]['accepted'])
            self.assertEqual(doc['candidates'][1]['rejection_reason'], 'unknown_source_field')
            self.assertFalse(doc['candidates'][1]['accepted'])

    def test_malformed_json_is_diagnosed_and_still_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                _accepted_remarks('недопустимый JSON', self.v, folder)
            record = json.loads(next(Path(folder).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['raw_response'], 'недопустимый JSON')
            self.assertIsNotNone(record['parse_error'])
        # A nested array inside a truncated response must not be mistaken for a valid response.
        with self.assertRaises(ValueError):
            _parse_model_json('[{"quote": "значение [1]"}')
        self.assertEqual(_parse_model_json('```json\n[]\n```\nготово'), [])
        self.assertEqual(_parse_model_json('результат: []\nдругая заметка [x]'), [])

    def test_gemma_unescaped_inner_quotes_trigger_fallback_not_partial_json(self):
        # Сокращённый синтетический пример полученного ответа: первый объект
        # корректен, второй содержит неэкранированные кавычки и не закрыт верно.
        text = ('```json\n[{"topic":"График","field":"scheduleType",'
                '"quote":"Гибкий график","problem":"Неясны детали",'
                '"recommendation":"Уточните график"},'
                '{"topic":"Описание","field":"description",'
                '"quote":"Полный соцпакет","problem":"Неясны детали",'
                '"recommendation":"Уточните условия "полного соцпакета.",'
                '"severity":"средняя"}]\n```')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                _accepted_remarks(text, self.v, folder)
            record = json.loads(next(Path(folder).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['raw_response'], text)
            self.assertEqual(record['candidates'], [])
            self.assertIsNotNone(record['parse_error'])

    def test_debug_write_failure_does_not_discard_valid_model_response(self):
        with tempfile.TemporaryDirectory() as folder:
            not_a_directory = Path(folder) / "file"
            not_a_directory.write_text("x", encoding="utf-8")
            accepted = _accepted_remarks(json.dumps([self.good], ensure_ascii=False), self.v,
                                         not_a_directory)
            self.assertEqual(len(accepted), 1)

    def test_api_debug_includes_model_not_authorization(self):
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self):
                content = json.dumps([good], ensure_ascii=False)
                return json.dumps({"choices": [{"message": {"content": content}}]}, ensure_ascii=False).encode()
        good = self.good
        with tempfile.TemporaryDirectory() as folder:
            model = ApiAnalyzer("https://example.test/v1", "secret-token", "model-name", folder, 0.0)
            with patch("src.ai.urllib.request.urlopen", return_value=Reply()) as urlopen:
                self.assertEqual(len(model.analyze(self.v)), 1)
            request = urlopen.call_args.args[0]
            self.assertEqual(json.loads(request.data)["temperature"], 0.0)
            debug = next(Path(folder).glob('*.json')).read_text(encoding='utf-8')
            self.assertEqual(json.loads(debug)['model'], 'model-name')
            self.assertNotIn('secret-token', debug)

    def test_fixture_has_three_valid_distinct_synthetic_cases(self):
        cases = extract_list(json.loads(Path('data/ai_probe_input.json').read_text(encoding='utf-8')))
        self.assertEqual([x['id'] for x in cases], ['probe-trip', 'probe-bonus', 'probe-clear'])
        # В этом наборе существующие правила не обнаруживают неоднозначность: именно её проверяем через модель.
        for raw in cases:
            self.assertEqual(analyze_rules(normalize_vacancy(raw)), [])

    def test_cli_with_mock_pipeline_and_zero_temperature(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return '\n'.join(x['content'] for x in messages)
        class Pipeline:
            tokenizer = Tokenizer()
            def __call__(self, prompt, **kwargs):
                self.kwargs = kwargs
                candidates = [good] if 'Командировки по области возникают регулярно' in prompt else []
                return [{'generated_text': json.dumps(candidates, ensure_ascii=False)}]
        good = self.good
        pipeline = Pipeline()
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'report.json'
            debug = Path(folder) / 'debug'
            with patch('src.ai.GemmaLocalAnalyzer._load', return_value=pipeline):
                result = main(['check', '--input', 'data/ai_probe_input.json', '--provider', 'local',
                               '--model', 'mock-model', '--temperature', '0',
                               '--ai-debug-dir', str(debug), '--output', str(report)])
            self.assertEqual(result, 0)
            self.assertEqual(pipeline.kwargs['do_sample'], False)
            self.assertNotIn('temperature', pipeline.kwargs)
            data = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(data['meta']['ai_status'], 'transformers-local')
            self.assertEqual([sum(r['source'] == 'ai' for r in v['remarks']) for v in data['vacancies']], [1, 0, 0])
            self.assertEqual(len(list(debug.glob('*.json'))), 3)

    def test_bad_temperature_is_rejected(self):
        with self.assertRaises(SystemExit):
            main(['check', '--input', 'data/ai_probe_input.json', '--temperature', '-1'])

if __name__ == '__main__':
    unittest.main()
