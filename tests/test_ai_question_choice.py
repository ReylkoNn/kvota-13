"""Выбор модели из двух безопасных шаблонов; реальные веса не используются."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai import ApiAnalyzer, GemmaLocalAnalyzer, _parse_choice
from src.analyzer import _question_options, analyze_batch, analyze_vacancy
from src.normalize import normalize_vacancy
from src.report import build_report
from src.rules import analyze_rules


class ChoiceModel:
    name = 'transformers-local'
    model_name = 'mock-model'
    def __init__(self, choice='B'):
        self.choice = choice
        self.calls = []
    def analyze(self, v):
        return []
    def choose_option(self, v, remark, options):
        self.calls.append((v, dict(remark), dict(options)))
        if isinstance(self.choice, Exception):
            raise self.choice
        return self.choice


def vague_schedule():
    return normalize_vacancy({
        'id': 'schedule', 'vacancyName': 'Повар', 'salary': '35000–45000',
        'duties': 'Готовить блюда, контролировать качество и вести документацию.',
        'requirements': 'Опыт поваром от 2 лет, 4 разряд.',
        'scheduleType': 'Гибкий график', 'employmentType': 'Полная занятость',
        'workPlace': 'Калининград', 'description': 'Работа поваром в столовой учреждения.',
    })


class TestAIQuestionChoice(unittest.TestCase):
    def test_choices_change_recommendation_and_keep_rule_provenance(self):
        vacancy = vague_schedule()
        baseline, _ = analyze_vacancy(vacancy, {'enabled': False})
        old = next(r for r in baseline if r['topic'] == 'График')
        a, status_a = analyze_vacancy(vacancy, {'enabled': True}, ChoiceModel('A'))
        b, status_b = analyze_vacancy(vacancy, {'enabled': True}, ChoiceModel('B'))
        ra = next(r for r in a if r['topic'] == 'График')
        rb = next(r for r in b if r['topic'] == 'График')
        self.assertEqual(status_a, 'transformers-local')
        self.assertEqual(status_b, 'transformers-local')
        self.assertNotEqual(ra['recommendation'], rb['recommendation'])
        for remark in (ra, rb):
            self.assertEqual(remark['source'], 'rule')
            self.assertEqual(remark['quote'], old['quote'])
            self.assertEqual(remark['problem'], old['problem'])
            self.assertEqual(remark['recommendation_source'], 'ai-template-choice')
            self.assertEqual(remark['recommendation_model'], 'mock-model')
            self.assertEqual(remark['recommendation_original'], old['recommendation'])
            self.assertNotEqual(remark['recommendation'], old['recommendation'])
            self.assertIn('дни', remark['recommendation'].lower())
            self.assertIn('час', remark['recommendation'].lower())

    def test_no_call_when_model_disabled(self):
        model = ChoiceModel()
        remarks, status = analyze_vacancy(vague_schedule(), {'enabled': False}, model)
        self.assertEqual(status, 'fallback')
        self.assertFalse(model.calls)
        self.assertFalse(any(r.get('recommendation_source') for r in remarks))

    def test_no_changes_on_invalid_choices_or_exception(self):
        baseline = analyze_rules(vague_schedule())
        for response in ('A or B', '', None, [], RuntimeError('failed')):
            with self.subTest(response=response):
                remarks, status = analyze_vacancy(vague_schedule(), {'enabled': True}, ChoiceModel(response))
                self.assertEqual(status, 'transformers-local')
                self.assertEqual(remarks, baseline)
                self.assertFalse(any(r.get('recommendation_source') for r in remarks))

    def test_unsupported_or_contradiction_not_changed(self):
        r = {'source':'rule', 'topic':'Оплата', 'problem':'Противоречие: минимальная оплата больше максимальной.'}
        self.assertIsNone(_question_options(r))
        self.assertIsNone(_question_options({'source':'ai', 'topic':'График', 'problem':'Не указан график работы.'}))

    def test_choice_is_retained_when_free_generation_falls_back(self):
        class Broken(ChoiceModel):
            def analyze(self, v):
                raise ValueError('invalid JSON')
        model = Broken()
        remarks, status = analyze_vacancy(vague_schedule(), {'enabled': True}, model)
        self.assertEqual(status, 'mixed')
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(any(r.get('recommendation_source') == 'ai-template-choice' for r in remarks))

    def test_fallback_when_neither_model_stage_succeeds(self):
        class Broken(ChoiceModel):
            def analyze(self, v):
                raise ValueError('invalid JSON')
        remarks, status = analyze_vacancy(vague_schedule(), {'enabled': True}, Broken('invalid'))
        self.assertEqual(status, 'fallback')
        self.assertFalse(any(r.get('recommendation_source') for r in remarks))

    def test_report_counts_ai_remarks_and_selections_separately(self):
        model = ChoiceModel('B')
        with patch('src.analyzer.build_ai_analyzer', return_value=(model, 'transformers-local')):
            items, status = analyze_batch([vague_schedule()], {'enabled': True})
        report = build_report(items, 'check', 'test', status)
        self.assertEqual(report['meta']['ai_generated_remarks'], 0)
        self.assertEqual(report['meta']['ai_selected_recommendations'], 1)
        self.assertEqual(report['vacancies'][0]['remarks'][0]['source'], 'rule')

    def test_parse_choice_is_strict(self):
        for text in ('A', ' a ', '"B"', 'B.'):
            self.assertIn(_parse_choice(text), ('A', 'B'))
        for text in ('A и B', 'выбираю A', '', 'ДА', 'C'):
            self.assertIsNone(_parse_choice(text))

    def test_local_template_choice_runs_real_model_interface_with_fake_pipe(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return '\n'.join(m['content'] for m in messages)
        class Pipeline:
            tokenizer = Tokenizer()
            def __call__(self, prompt, **kwargs):
                self.prompt, self.kwargs = prompt, kwargs
                return [{'generated_text': 'B'}]
        pipeline = Pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            model = GemmaLocalAnalyzer('mock-model', debug_dir=tmp)
            model._pipe = pipeline
            issue = analyze_rules(vague_schedule())[0]
            options = _question_options(issue)
            self.assertEqual(model.choose_option(vague_schedule(), issue, options), 'B')
            self.assertEqual(pipeline.kwargs['max_new_tokens'], 8)
            self.assertFalse(pipeline.kwargs['do_sample'])
            self.assertIn('Гибкий график', pipeline.prompt)
            record = json.loads(next(Path(tmp).glob('*.json')).read_text(encoding='utf8'))
            self.assertEqual(record['raw_response'], 'B')
            self.assertEqual(record['candidates'][0]['stage'], 'recommendation_choice')

    def test_local_full_report_with_mock_pipeline_and_empty_model_remarks(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return '\n'.join(m['content'] for m in messages)
        class Pipeline:
            tokenizer = Tokenizer()
            def __call__(self, prompt, **kwargs):
                return [{'generated_text': 'B' if kwargs['max_new_tokens'] == 8 else '[]'}]
        model = GemmaLocalAnalyzer('mock-model', temperature=0)
        model._pipe = Pipeline()
        with patch('src.analyzer.build_ai_analyzer', return_value=(model, 'transformers-local')):
            items, status = analyze_batch([vague_schedule()], {'enabled': True})
        report = build_report(items, 'check', 'test', status)
        self.assertEqual(report['meta']['ai_generated_remarks'], 0)
        self.assertEqual(report['meta']['ai_selected_recommendations'], 1)
        self.assertEqual(report['vacancies'][0]['remarks'][0]['recommendation_choice'], 'B')

    def test_api_choice_does_not_log_key(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self): return b'{"choices":[{"message":{"content":"A"}}]}'
        with tempfile.TemporaryDirectory() as tmp:
            model = ApiAnalyzer('https://example.test/v1', 'private-secret', 'my-model', tmp)
            issue = analyze_rules(vague_schedule())[0]
            with patch('src.ai.urllib.request.urlopen', return_value=Response()):
                self.assertEqual(model.choose_option(vague_schedule(), issue, _question_options(issue)), 'A')
            debug = next(Path(tmp).glob('*.json')).read_text(encoding='utf8')
            self.assertNotIn('private-secret', debug)

if __name__ == '__main__':
    unittest.main()
