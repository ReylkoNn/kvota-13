"""Регрессии на синтетических примерах, выявленные при ручном просмотре отчёта Gemma."""
import unittest
from unittest.mock import patch

from src.ai import _valid_remark
from src.analyzer import analyze_batch
from src.normalize import normalize_vacancy
from src.rules import analyze_rules


def model_remark(**overrides):
    out = {'topic':'График', 'field':'scheduleType', 'quote':'5/2, 09:00–18:00',
           'problem':'Неясны дни и продолжительность рабочего дня в рамках 5/2',
           'recommendation':'Уточните дни и продолжительность рабочего дня в рамках 5/2',
           'severity':'средняя'}
    out.update(overrides)
    return out


class TestRealReportRegressions(unittest.TestCase):
    def test_existing_hours_guard_rejects_claim_that_duration_is_unclear(self):
        v = normalize_vacancy({'scheduleType':'5/2, 09:00–18:00'})
        self.assertIsNone(_valid_remark(model_remark(), v))

    def test_existing_hours_do_not_reject_question_about_exact_weekdays(self):
        v = normalize_vacancy({'scheduleType':'5/2, 09:00–18:00'})
        candidate = model_remark(problem='Неясно, в какие дни недели работает сотрудник.',
                                 recommendation='Уточните дни недели.')
        self.assertIsNotNone(_valid_remark(candidate, v))

    def test_existing_hours_do_not_reject_question_about_breaks(self):
        v = normalize_vacancy({'scheduleType':'5/2, 09:00–18:00'})
        candidate = model_remark(problem='Неясна продолжительность перерыва.',
                                 recommendation='Уточните продолжительность перерыва.')
        self.assertIsNotNone(_valid_remark(candidate, v))

    def test_model_may_ask_duration_if_hours_are_unspecified(self):
        v = normalize_vacancy({'scheduleType':'5/2'})
        candidate = model_remark(quote='5/2')
        self.assertIsNotNone(_valid_remark(candidate, v))

    def test_exact_duplicate_of_rule_remark_is_not_added(self):
        v = normalize_vacancy({'id':'1', 'scheduleType':'Гибкий график'})
        duplicate = {'topic':'График', 'field':'scheduleType', 'quote':'Гибкий график',
                     'problem':'Неясны детали гибкого графика',
                     'recommendation':'Уточните время работы', 'severity':'средняя', 'source':'ai'}
        class Mock:
            name = 'mock-model'
            def analyze(self, v): return [duplicate]
        with patch('src.analyzer.build_ai_analyzer', return_value=(Mock(), 'mock')):
            output, _ = analyze_batch([v], {'enabled':True})
        schedule = [r for r in output[0]['remarks'] if r['topic']=='График' and r['quote']=='Гибкий график']
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]['source'], 'rule')

    def test_different_quotation_is_not_deduplicated(self):
        v = normalize_vacancy({'id':'1', 'scheduleType':'Гибкий график',
                               'description':'Смены устанавливаются индивидуально.'})
        candidate = {'topic':'График', 'field':'description', 'quote':'Смены устанавливаются индивидуально',
                     'problem':'Неясно, как формируются смены', 'recommendation':'Уточните порядок составления смен',
                     'severity':'средняя', 'source':'ai'}
        class Mock:
            name = 'mock-model'
            def analyze(self, v): return [candidate]
        with patch('src.analyzer.build_ai_analyzer', return_value=(Mock(), 'mock')):
            output, _ = analyze_batch([v], {'enabled':True})
        self.assertTrue(any(r['source']=='ai' for r in output[0]['remarks']))

    def test_distinct_long_quotes_are_not_truncated_for_deduplication(self):
        prefix = 'x' * 70
        v = normalize_vacancy({'id':'1', 'description':prefix + ' A ' + prefix + ' B'})
        def candidate(quote):
            return {'topic':'Описание', 'field':'description', 'quote':quote,
                    'problem':'Неясно', 'recommendation':'Уточните', 'severity':'средняя', 'source':'ai'}
        class Mock:
            name = 'mock-model'
            def analyze(self, v): return [candidate(prefix + ' A'), candidate(prefix + ' B')]
        with patch('src.analyzer.build_ai_analyzer', return_value=(Mock(), 'mock')):
            output, _ = analyze_batch([v], {'enabled':True})
        self.assertEqual(sum(r['source']=='ai' for r in output[0]['remarks']), 2)

    def test_salary_contradiction_references_both_actual_source_fields(self):
        v = normalize_vacancy({'id':'1', 'salary_from':'60 000', 'salary_to':'40 000'})
        r = next(x for x in analyze_rules(v) if x['topic']=='Оплата' and 'Противоречие' in x['problem'])
        self.assertEqual(r['field'], 'salary_from | salary_to')
        self.assertEqual(r['quote'], '60 000 | 40 000')

    def test_work_format_quotes_name_the_correct_fields(self):
        v = normalize_vacancy({'id':'1', 'workPlace':'В офисе', 'description':'Можно работать удалённо.'})
        r = next(x for x in analyze_rules(v) if x['topic']=='Место работы' and ' | ' in x['quote'])
        self.assertEqual(r['field'], 'description | workPlace')
        self.assertEqual(r['quote'], 'удалённо | В офисе')
        self.assertIn('порядок', r['problem'])

    def test_employment_discrepancy_references_source_fields(self):
        v = normalize_vacancy({'id':'1', 'employment':'Частичная занятость',
                               'vacancyDescription':'Работа на 40 часов в неделю, полный день.'})
        r = next(x for x in analyze_rules(v) if x['topic']=='Тип занятости' and ' | ' in x['quote'])
        self.assertEqual(r['field'], 'employment | vacancyDescription')
        self.assertEqual(r['quote'], 'Частичная занятость | 40 часов')
        self.assertNotIn('незакон', r['problem'].lower())

if __name__ == '__main__':
    unittest.main()
