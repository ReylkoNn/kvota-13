"""Регрессии из просмотра синтетического отчёта Gemma, без запуска модели."""
import json
import tempfile
import unittest
from pathlib import Path

from src.ai import _accepted_remarks, _validate_remark
from src.analyzer import analyze_vacancy
from src.normalize import normalize_vacancy


def candidate(**changes):
    result = {
        'topic': 'Описание', 'field': 'description',
        'quote': 'Требуется секретарь',
        'problem': 'Необходимо уточнить, какие именно обязанности и требования от секретаря.',
        'recommendation': 'Уточните обязанности и требования от секретаря.',
        'severity': 'средняя',
    }
    result.update(changes)
    return result


class JsonModel:
    name = 'gemma-local'
    def __init__(self, responses):
        self.responses = responses
    def analyze(self, v):
        return _accepted_remarks(self.responses, v)


class TestFullReportRegressions(unittest.TestCase):
    def test_cook_generic_claim_rejected_but_specific_question_preserved(self):
        v = normalize_vacancy({
            'id': '1002', 'description': 'Приготовление блюд по меню, контроль качества, соблюдение санитарных норм.',
            'requirements': 'Опыт работы поваром от 2 лет, 4 разряд, медицинская книжка.',
        })
        r = candidate(quote=v['description'], problem='Неясны требования к работе.',
                      recommendation='Уточните требования к работе.')
        valid, reason = _validate_remark(r, v)
        self.assertIsNone(valid)
        self.assertEqual(reason, 'generic_requirements_already_present')
        specific = dict(r, problem='Неясны требования к действующей медицинской книжке.',
                        recommendation='Уточните требования к действующей медицинской книжке.')
        self.assertIsNotNone(_validate_remark(specific, v)[0])
        # Эвристика не должна утверждать наличие требований, которых нет.
        empty = normalize_vacancy({'description': v['description']})
        self.assertIsNotNone(_validate_remark(r, empty)[0])

    def test_generic_secretary_ai_repeats_missing_rules(self):
        v = normalize_vacancy({'id': '1004', 'description': 'Требуется секретарь.', 'duties': '', 'requirements': ''})
        data = [candidate(topic='Обязанности'), candidate(topic='Описание')]
        remarks, status = analyze_vacancy(v, {'enabled': True}, JsonModel(json.dumps(data, ensure_ascii=False)))
        self.assertEqual(status, 'gemma-local')
        self.assertFalse(any(r['source'] == 'ai' for r in remarks))
        self.assertTrue(any(r['topic'] == 'Обязанности' and r['source'] == 'rule' for r in remarks))
        self.assertTrue(any(r['topic'] == 'Требования' and r['source'] == 'rule' for r in remarks))

    def test_specific_secretary_question_survives_missing_rules(self):
        v = normalize_vacancy({'id': '1004', 'description': 'Требуется секретарь.', 'duties': '', 'requirements': ''})
        r = candidate(problem='Непонятно, требуется ли владение иностранным языком.',
                      recommendation='Уточните, требуется ли владение иностранным языком.')
        remarks, _ = analyze_vacancy(v, {'enabled': True}, JsonModel(json.dumps([r], ensure_ascii=False)))
        self.assertTrue(any(x['source'] == 'ai' for x in remarks))

    def test_generic_question_not_suppressed_if_one_rule_missing(self):
        v = normalize_vacancy({'description': 'Требуется секретарь.', 'duties': '',
                               'requirements': 'Опыт работы от года.'})
        remarks, _ = analyze_vacancy(v, {'enabled': True}, JsonModel(json.dumps([candidate()], ensure_ascii=False)))
        self.assertTrue(any(x['source'] == 'ai' for x in remarks))

    def test_ai_duplicates_across_topics_suppressed_but_distinct_questions_remain(self):
        v = normalize_vacancy({'id': 'case', 'description': 'Требуется секретарь.',
                               'duties': 'Ведение документов и приём посетителей.',
                               'requirements': 'Опыт работы от года.'})
        a = candidate(topic='Обязанности', problem='Неясен объём работы.',
                      recommendation='Уточните объём работы.')
        b = dict(a, topic='Описание')
        c = dict(a, topic='Требования', problem='Неясен требуемый опыт.',
                 recommendation='Уточните требуемый опыт.')
        remarks, _ = analyze_vacancy(v, {'enabled': True}, JsonModel(json.dumps([a, b, c], ensure_ascii=False)))
        self.assertEqual([r['problem'] for r in remarks if r['source'] == 'ai'],
                         ['Неясен объём работы.', 'Неясен требуемый опыт.'])

    def test_unescaped_quotes_in_1003_are_not_repaired(self):
        v = normalize_vacancy({'id': '1003', 'scheduleType': '5/2'})
        text = ('[{"topic":"График","field":"scheduleType","quote":"5/2",'
                '"problem":"Неясны детали графика.",'
                '"recommendation":"Уточните, что подразумевается под "5/2".",'
                '"severity":"средняя"}]')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _accepted_remarks(text, v, directory)
            diagnostic = json.loads(next(Path(directory).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(diagnostic['raw_response'], text)
            self.assertIsNotNone(diagnostic['parse_error'])
            self.assertEqual(diagnostic['candidates'], [])
        remarks, status = analyze_vacancy(v, {'enabled': True}, JsonModel(text))
        self.assertEqual(status, 'fallback')
        self.assertFalse(any(x['source'] == 'ai' for x in remarks))


if __name__ == '__main__':
    unittest.main()
