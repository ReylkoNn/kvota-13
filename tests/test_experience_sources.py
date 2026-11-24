"""Проверка сопоставления двух полей опыта на синтетических данных."""
import json
import unittest

from src.ai import _accepted_remarks
from src.analyzer import analyze_vacancy
from src.normalize import normalize_vacancy
from src.rules import analyze_rules


class Model:
    name = 'gemma-local'
    def __init__(self, rows):
        self.rows = rows
    def analyze(self, vacancy):
        return _accepted_remarks(json.dumps(self.rows, ensure_ascii=False), vacancy)


class TestExperienceSources(unittest.TestCase):
    def setUp(self):
        self.qualification = 'Умение читать документы. Опыт работы по калибровке приветствуется (1–3 года).'

    def vacancy(self, experience=1, qualification=None):
        return normalize_vacancy({'id': 'synthetic', 'job-name': 'Инженер',
                                  'requirement': {'experience': experience,
                                                  'qualification': qualification if qualification is not None else self.qualification}})

    def test_two_source_values_are_quoted_without_inventing_units_or_conflict(self):
        v = self.vacancy()
        remark = next(r for r in analyze_rules(v) if r['field'] == 'requirement.qualification | requirement.experience')
        self.assertEqual(remark['source'], 'rule')
        self.assertEqual(remark['quote'], 'Опыт работы по калибровке приветствуется | 1')
        self.assertNotIn('противореч', remark['problem'].lower())
        self.assertNotIn('1 год', remark['problem'].lower())
        self.assertIn('уточните', remark['recommendation'].lower())

    def test_zero_unknown_negative_or_negated_preference_not_treated_as_positive(self):
        for value in (0, '0', None, '', 'нет данных', -1, True, 'nan'):
            with self.subTest(value=value):
                self.assertFalse(any(r['field'] == 'requirement.qualification | requirement.experience'
                                     for r in analyze_rules(self.vacancy(value))))
        v = self.vacancy(1, 'Опыт работы по калибровке не приветствуется.')
        self.assertFalse(any(r['field'] == 'requirement.qualification | requirement.experience'
                             for r in analyze_rules(v)))

    def test_preferred_experience_question_is_deduplicated_against_two_field_rule(self):
        v = self.vacancy()
        r = {'topic': 'Требования', 'field': 'requirement.qualification', 'quote': self.qualification,
             'problem': 'Неясны требования к опыту.',
             'recommendation': 'Уточните, требуется ли опыт работы по калибровке.'}
        remarks, status = analyze_vacancy(v, {'enabled': True}, Model([r]))
        self.assertEqual(status, 'gemma-local')
        self.assertTrue(any(x['field'] == 'requirement.qualification | requirement.experience' for x in remarks))
        self.assertFalse(any(x['source'] == 'ai' and x['topic'] == 'Требования' for x in remarks))

    def test_specific_ai_question_is_retained(self):
        v = self.vacancy()
        r = {'topic': 'Требования', 'field': 'requirement.qualification', 'quote': self.qualification,
             'problem': 'Неясно, как оценивается практический опыт.',
             'recommendation': 'Уточните, как оценивается практический опыт.'}
        remarks, _ = analyze_vacancy(v, {'enabled': True}, Model([r]))
        self.assertTrue(any(x['source'] == 'ai' and x['problem'] == r['problem'] for x in remarks))


if __name__ == '__main__':
    unittest.main()
