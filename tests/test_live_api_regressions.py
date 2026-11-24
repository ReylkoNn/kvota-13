"""Синтетические регрессии по структуре и формулировкам реального отчёта.

Это не сетевые тесты и не проверка неизменности текущего API.
"""
import unittest
from unittest.mock import patch

from src.ai import _validate_remark
from src.config import DEFAULTS, load_config
from src.normalize import normalize_vacancy
from src.rules import analyze_rules


class TestLiveApiRegressions(unittest.TestCase):
    def test_default_endpoint_matches_successful_user_check(self):
        expected = 'http://opendata.trudvsem.ru/api/v1/vacancies/region/3900000000000'
        self.assertEqual(DEFAULTS['source_url'], expected)
        with patch.dict('os.environ', {'SOURCE_URL': ''}):
            self.assertEqual(load_config()['source_url'], expected)

    def test_nested_qualification_is_seen_alongside_other_requirements(self):
        v = normalize_vacancy({
            'requirements': 'Ответственность',
            'requirement': {'qualification': 'Опыт работы инженером от 2 лет.'},
            'description': 'Офисная работа.',
        })
        self.assertEqual(v['requirements'], 'Ответственность')
        self.assertEqual(v['requirement_qualification'], 'Опыт работы инженером от 2 лет.')
        self.assertIn(v['requirement_qualification'], v['all_text'])
        self.assertFalse(any(r['topic'] == 'Требования' for r in analyze_rules(v)))

    def test_nested_qualification_not_mistaken_for_missing_requirements(self):
        v = normalize_vacancy({'requirement': {'qualification': 'Опыт работы от 2 лет.'}})
        self.assertFalse(any(r['topic'] == 'Требования' for r in analyze_rules(v)))

    def test_nested_soft_skill_quote_names_actual_source(self):
        v = normalize_vacancy({'requirement': {'qualification': 'Коммуникабельность и ответственность'}})
        r = next(r for r in analyze_rules(v) if r['topic'] == 'Требования' and r['quote'])
        self.assertEqual(r['field'], 'requirement.qualification')
        self.assertIn(r['quote'], v['raw']['requirement']['qualification'])

    def test_ai_guard_can_see_nested_qualification(self):
        v = normalize_vacancy({'description': 'Работа в офисе.',
                               'requirement': {'qualification': 'Опыт работы от 2 лет.'}})
        r = {'topic': 'Описание', 'field': 'description', 'quote': 'Работа в офисе',
             'problem': 'Неясны требования к работе.', 'recommendation': 'Уточните требования к работе.'}
        self.assertEqual(_validate_remark(r, v)[1], 'generic_requirements_already_present')

    def test_broad_duties_question_is_rejected_specific_one_is_kept(self):
        v = normalize_vacancy({'duty': 'Управление проектами, сопровождение систем и подготовка отчётов.'})
        r = {'topic': 'Обязанности', 'field': 'duty', 'quote': v['duties'],
             'problem': 'Неясны конкретные обязанности, которые нужно выполнять в рамках работы.',
             'recommendation': 'Перечислите конкретные обязанности, которые нужно выполнять в рамках работы.'}
        self.assertEqual(_validate_remark(r, v)[1], 'generic_duties_question')
        r.update(problem='Неясно, какие показатели отслеживать.',
                 recommendation='Уточните показатели для отчётности.')
        self.assertIsNotNone(_validate_remark(r, v)[0])


if __name__ == '__main__':
    unittest.main()
