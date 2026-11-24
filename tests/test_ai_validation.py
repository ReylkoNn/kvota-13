"""Регрессии по реальному отчёту и проверка валидного пути до финального отчёта."""
import json
import unittest
from unittest.mock import patch

from src.ai import GemmaLocalAnalyzer, _accepted_remarks, _valid_remark, _vacancy_text
from src.analyzer import analyze_batch
from src.normalize import normalize_vacancy


def remark(**overrides):
    result = {"topic": "График", "field": "description", "quote": "нестандартные часы",
              "problem": "Неясно, какие часы подразумеваются.",
              "recommendation": "Укажите часы работы.", "severity": "средняя"}
    result.update(overrides)
    return result


class TestModelValidation(unittest.TestCase):
    def setUp(self):
        self.v = normalize_vacancy({
            "id": "42", "job-name": "Повар", "description": "Работа: нестандартные часы.",
            "requirements": "Опыт работы поваром от 2 лет, 4 разряд, медицинская книжка.",
            "duties": "Приготовление блюд.", "address": "Калининград, ул. Мира, 1",
            "salary_min": 35000, "addresses": {"address": [{"location": "Калининград, Мира 1"}]},
        })

    def test_prompt_has_original_field_names_and_structured_values(self):
        fields = json.loads(_vacancy_text(self.v))
        self.assertEqual(fields["job-name"], "Повар")
        self.assertEqual(fields["salary_min"], "35000")
        self.assertEqual(fields["addresses.address[0].location"], "Калининград, Мира 1")
        self.assertNotIn("id", fields)

    def test_good_observation_is_kept(self):
        accepted = _valid_remark(remark(), self.v)
        self.assertEqual(accepted["source"], "ai")
        self.assertEqual(accepted["quote"], "нестандартные часы")

    def test_model_cannot_copy_prompt_example_into_unrelated_vacancy(self):
        copied = remark(topic="Описание", quote="Возможны командировки",
                        problem="Неясны частота и длительность командировок.",
                        recommendation="Уточните частоту и длительность командировок.")
        self.assertIsNone(_valid_remark(copied, self.v))
        vacancy = normalize_vacancy({"description": "Возможны командировки."})
        self.assertIsNotNone(_valid_remark(copied, vacancy))

    def test_user_cook_false_positive_invalid_field_and_topic(self):
        candidate = remark(topic="Работа", field="Работа",
                           quote="Опыт работы поваром от 2 лет, 4 разряд, медицинская книжка.",
                           problem="Проблема с определением места работы")
        self.assertIsNone(_valid_remark(candidate, self.v))

    def test_quote_must_be_in_declared_field_not_another_field(self):
        self.assertIsNone(_valid_remark(remark(field="requirements"), self.v))
        self.assertIsNone(_valid_remark(remark(field="requirements", topic="Место работы",
                                              quote="Опыт работы поваром"), self.v))

    def test_user_1c_false_contradiction_rejected(self):
        v = normalize_vacancy({"duties": "Ведение бухгалтерского учёта, отчётность, сверки.",
                               "requirements": "Опыт от 3 лет, знание 1С."})
        candidate = remark(topic="Требования", field="requirements", quote="Опыт от 3 лет, знание 1С.",
                           problem="Требование о знании 1С может вызывать противоречие с условиями работы")
        self.assertIsNone(_valid_remark(candidate, v))

    def test_unknown_field_empty_quote_and_nonstring_values_rejected(self):
        for candidate in (remark(field="неизвестно"), remark(quote=""), remark(topic="Оператор ЭВМ"),
                          remark(quote=None), remark(problem=None), remark(quote="придумано")):
            with self.subTest(candidate=candidate):
                self.assertIsNone(_valid_remark(candidate, self.v))

    def test_nested_location_uses_source_path(self):
        candidate = remark(topic="Место работы", field="addresses.address[0].location",
                           quote="Мира 1", problem="Неясен корпус.", recommendation="Уточните корпус.")
        self.assertEqual(_valid_remark(candidate, self.v)["field"], "addresses.address[0].location")

    def test_gemma_role_fallback_and_result_added_to_report(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if messages[0]["role"] == "system":
                    raise ValueError("system unsupported")
                self.seen = messages
                return messages[0]["content"]
        class Pipeline:
            tokenizer = Tokenizer()
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return [{"generated_text": json.dumps([remark()], ensure_ascii=False)}]
        pipeline = Pipeline()
        model = GemmaLocalAnalyzer("google/gemma-2-2b-it")
        model._pipe = pipeline
        with patch("src.analyzer.build_ai_analyzer", return_value=(model, "gemma-local")):
            result, status = analyze_batch([self.v], {"enabled": True})
        self.assertEqual(status, "gemma-local")
        self.assertTrue(any(x["source"] == "ai" for x in result[0]["remarks"]))
        self.assertEqual(pipeline.tokenizer.seen[0]["role"], "user")
        self.assertIn('"description"', pipeline.prompt)

    def test_filtered_candidates_are_not_silently_marked_as_ai_remarks(self):
        candidates = [remark(), remark(topic="Работа", field="Работа")]
        accepted = _accepted_remarks(json.dumps(candidates, ensure_ascii=False), self.v)
        self.assertEqual(len(accepted), 1)


if __name__ == "__main__":
    unittest.main()
