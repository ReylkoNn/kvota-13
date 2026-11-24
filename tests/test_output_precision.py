"""Границы консервативной фильтрации: все данные вымышлены."""
import json
import unittest

from src.ai import _accepted_remarks, _validate_remark
from src.analyzer import analyze_vacancy
from src.normalize import normalize_vacancy


class MockModel:
    name = 'gemma-local'
    def __init__(self, rows):
        self.rows = rows
    def analyze(self, vacancy):
        return _accepted_remarks(json.dumps(self.rows, ensure_ascii=False), vacancy)


class TestOutputPrecision(unittest.TestCase):
    def test_preferred_experience_not_reasked_generically(self):
        q = 'Опыт работы по поверке и калибровке приветствуется (1–3 года).'
        v = normalize_vacancy({'requirement': {'qualification': q}})
        candidate = {'topic': 'Требования', 'field': 'requirement.qualification', 'quote': q,
                     'problem': 'Неясны требования к опыту работы.',
                     'recommendation': 'Уточните, требуется ли опыт работы по поверке и калибровке.'}
        self.assertEqual(_validate_remark(candidate, v)[1], 'experience_preference_already_stated')
        # Содержательный вопрос и возможный конфликт с другим полем не скрываем.
        specific = dict(candidate, recommendation='Уточните, как оценивается опыт работы по поверке.')
        self.assertIsNotNone(_validate_remark(specific, v)[0])
        conflicting = normalize_vacancy({'requirement': {'qualification': q, 'experience': 3}})
        self.assertIsNotNone(_validate_remark(candidate, conflicting)[0])
        mandatory = normalize_vacancy({'requirement': {'qualification': q},
                                       'requirements': 'Обязателен опыт работы от 3 лет.'})
        self.assertIsNotNone(_validate_remark(candidate, mandatory)[0])

    def test_question_about_missing_experience_is_not_rejected(self):
        q = 'Знание офисных программ приветствуется.'
        v = normalize_vacancy({'requirement': {'qualification': q}})
        r = {'topic': 'Требования', 'field': 'requirement.qualification', 'quote': q,
             'problem': 'Неясны требования к опыту.', 'recommendation': 'Уточните, требуется ли опыт работы.'}
        self.assertIsNotNone(_validate_remark(r, v)[0])

    def test_ungrammatical_interaction_question_is_rejected(self):
        quote = 'Управление проектами и взаимодействие с подрядными организациями'
        v = normalize_vacancy({'duty': quote})
        row = {'topic': 'Обязанности', 'field': 'duty', 'quote': quote,
               'problem': 'Неясны проекты и подрядные организации.',
               'recommendation': 'Уточните, какие конкретные проекты и подрядные организации нужно взаимодействовать.'}
        self.assertEqual(_validate_remark(row, v)[1], 'invalid_interaction_wording')
        row['recommendation'] = 'Уточните, с какими подрядными организациями предстоит взаимодействовать.'
        self.assertIsNotNone(_validate_remark(row, v)[0])

    def test_same_duties_cited_by_rule_and_ai_is_not_repeated(self):
        q = 'Уборка помещений, выполнение должностных обязанностей согласно инструкции.'
        v = normalize_vacancy({'id': 'test', 'duty': q})
        row = {'topic': 'Обязанности', 'field': 'duty', 'quote': q,
               'problem': 'Неясны должностные обязанности.',
               'recommendation': 'Уточните должностные обязанности.'}
        remarks, status = analyze_vacancy(v, {'enabled': True}, MockModel([row]))
        self.assertEqual(status, 'gemma-local')
        self.assertEqual(len([r for r in remarks if r['topic'] == 'Обязанности']), 1)
        self.assertEqual(next(r for r in remarks if r['topic'] == 'Обязанности')['source'], 'rule')

    def test_distinct_question_on_same_duties_kept(self):
        q = 'Уборка помещений, выполнение должностных обязанностей согласно инструкции.'
        v = normalize_vacancy({'id': 'test', 'duty': q})
        row = {'topic': 'Обязанности', 'field': 'duty', 'quote': q,
               'problem': 'Неясна частота уборки.', 'recommendation': 'Уточните частоту уборки.'}
        remarks, _ = analyze_vacancy(v, {'enabled': True}, MockModel([row]))
        self.assertTrue(any(r['source'] == 'ai' and r['problem'] == row['problem'] for r in remarks))

    def test_pay_fallback_does_not_repeat_rule_question(self):
        v = normalize_vacancy({'salary': 'по договорённости',
                               'description': 'Стабильная зарплата, своевременная оплата. Полный соцпакет.'})
        remarks, status = analyze_vacancy(v, {'enabled': False})
        self.assertEqual(status, 'fallback')
        self.assertEqual(len([r for r in remarks if r['topic'] == 'Оплата']), 1)
        self.assertTrue(any(r['source'] == 'fallback' and r['quote'] == 'Полный соцпакет' for r in remarks))

    def test_employment_in_qualification_is_not_marked_missing(self):
        v = normalize_vacancy({'schedule': 'Полный рабочий день',
                               'requirement': {'qualification': 'Полная занятость, опыт от 2 лет.'}})
        self.assertFalse(any(r['topic'] == 'Тип занятости' for r in analyze_vacancy(v, {'enabled': False})[0]))

    def test_full_workday_alone_does_not_prove_employment_type(self):
        v = normalize_vacancy({'schedule': 'Полный рабочий день'})
        r = next(r for r in analyze_vacancy(v, {'enabled': False})[0] if r['topic'] == 'Тип занятости')
        self.assertTrue(r['problem'].startswith('Не удалось найти'))
        self.assertIn('Проверьте', r['recommendation'])


if __name__ == '__main__':
    unittest.main()
