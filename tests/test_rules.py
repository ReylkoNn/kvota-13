import unittest

from src.normalize import normalize_vacancy
from src.rules import analyze_rules


def _v(**kw):
    base = {"id": "1", "vacancyName": "Повар", "description": "x" * 120,
            "duties": "Готовить блюда, контролировать качество, вести документацию.",
            "requirements": "Опыт от 2 лет, 4 разряд.",
            "salary": "35000-45000", "scheduleType": "5/2, 09:00-18:00",
            "employmentType": "Полная занятость", "workPlace": "Калининград"}
    base.update(kw)
    return normalize_vacancy(base)


class TestRules(unittest.TestCase):
    def test_good_vacancy_no_remarks(self):
        self.assertEqual(analyze_rules(_v()), [])

    def test_flexible_schedule_without_days(self):
        r = analyze_rules(_v(scheduleType="Гибкий график", description="x" * 120))
        self.assertTrue(any(x["topic"] == "График" for x in r))

    def test_missing_salary(self):
        r = analyze_rules(_v(salary="", salaryMin=None, salaryMax=None,
                             description="Готовить блюда. " + "x" * 120))
        self.assertTrue(any(x["topic"] == "Оплата" for x in r))

    def test_contradiction_salary(self):
        r = analyze_rules(_v(salaryMin=60000, salaryMax=40000))
        self.assertTrue(any("Противоречие" in x["problem"] for x in r))

    def test_structured_field_covers_description_gap(self):
        # Оплата есть в полях — отсутствие в описании не проблема
        r = analyze_rules(_v(description="Коротко."))
        self.assertFalse(any(x["topic"] == "Оплата" for x in r))


if __name__ == "__main__":
    unittest.main()
