import unittest

from src.ai import FallbackAnalyzer
from src.analyzer import analyze_batch, analyze_vacancy
from src.normalize import normalize_vacancy

AI_OFF = {"enabled": False, "provider": "off"}


class TestAI(unittest.TestCase):
    def test_fallback_finds_ambiguous(self):
        v = normalize_vacancy({"id": "1", "vacancyName": "X",
                               "description": "Стабильная зарплата, дружный коллектив."})
        r = FallbackAnalyzer().analyze(v)
        self.assertTrue(any(x["source"] == "fallback" for x in r))

    def test_analyzer_merges_and_dedups(self):
        v = normalize_vacancy({"id": "1", "vacancyName": "X",
                               "description": "Стабильная зарплата. " + "x" * 120})
        remarks, status = analyze_vacancy(v, AI_OFF)
        self.assertEqual(status, "fallback")
        keys = [(r["topic"], r["field"], r["quote"]) for r in remarks]
        self.assertEqual(len(keys), len(set(keys)))

    def test_batch_report_shape(self):
        vs = [normalize_vacancy({"id": "1", "vacancyName": "Повар",
                                 "description": "x" * 120,
                                 "duties": "Готовить блюда, контролировать качество.",
                                 "requirements": "Опыт от 2 лет.",
                                 "salary": "35000-45000", "scheduleType": "5/2",
                                 "employmentType": "Полная занятость",
                                 "workPlace": "Калининград"})]
        results, _ = analyze_batch(vs, AI_OFF)
        self.assertEqual(results[0]["id"], "1")
        self.assertIn("url", results[0])
        self.assertIsInstance(results[0]["remarks"], list)


if __name__ == "__main__":
    unittest.main()
