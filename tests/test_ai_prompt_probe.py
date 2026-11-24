"""Проверка диагностического эксперимента с имитацией пайплайна, не тест модели."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai_prompt_probe import _decode_yes_no, main, run_probe
from src.normalize import extract_list, normalize_vacancy


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(x['content'] for x in messages)


class FakePipeline:
    tokenizer = FakeTokenizer()
    def __init__(self):
        self.calls = []
    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if 'Ответь одним словом' in prompt:
            answer = 'ДА' if 'регулярно' in prompt else 'НЕТ'
        elif 'регулярно' in prompt:
            answer = json.dumps([{'topic':'Описание', 'field':'description', 'quote':'регулярно',
                                  'problem':'Частота не ясна', 'recommendation':'Уточните частоту',
                                  'severity':'средняя'}], ensure_ascii=False)
        else:
            answer = '[]'
        return [{'generated_text': answer}]


class TestPromptProbe(unittest.TestCase):
    def test_real_fixture_two_strategies_mock_only(self):
        raw = json.loads(Path('data/ai_probe_input.json').read_text(encoding='utf-8'))
        vacancies = [normalize_vacancy(x) for x in extract_list(raw)]
        fake = FakePipeline()
        results = run_probe(vacancies, fake)
        self.assertEqual(len(results), 6)
        self.assertEqual(len(fake.calls), 6)
        self.assertTrue(all(x[1]['do_sample'] is False for x in fake.calls))
        self.assertTrue(all('probe-trip' not in x[0] for x in fake.calls))
        trip = [x for x in results if x['vacancy_id'] == 'probe-trip']
        self.assertTrue(trip[0]['accepted_remarks'])
        self.assertEqual(trip[1]['answer'], 'ДА')

    def test_yes_no_strict(self):
        self.assertEqual(_decode_yes_no('  Да.'), 'ДА')
        self.assertEqual(_decode_yes_no('нет'), 'НЕТ')
        self.assertIsNone(_decode_yes_no('Возможно'))
        self.assertIsNone(_decode_yes_no('даже'))

    def test_cli_enforces_offline_before_loading(self):
        fake = FakePipeline()
        observed = {}
        def load(_):
            observed['hub'] = os.getenv('HF_HUB_OFFLINE')
            observed['transformers'] = os.getenv('TRANSFORMERS_OFFLINE')
            return fake
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=False):
            output = Path(folder) / 'result.json'
            with patch('src.ai_prompt_probe.GemmaLocalAnalyzer._load', load):
                code = main(['--input','data/ai_probe_input.json','--model','mock-model','--output',str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(observed, {'hub':'1', 'transformers':'1'})
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(len(result['results']), 6)
            self.assertTrue(result['offline'])
            self.assertEqual(result['model'], 'mock-model')

    def test_missing_cached_model_returns_error_no_output(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=False):
            output = Path(folder) / 'result.json'
            with patch('src.ai_prompt_probe.GemmaLocalAnalyzer._load', side_effect=OSError('нет модели')):
                code = main(['--input','data/ai_probe_input.json','--model','mock-model','--output',str(output)])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())

if __name__ == '__main__':
    unittest.main()
