import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai import _parse_model_json, _valid_remark
from src.analyzer import analyze_batch
from src.fetcher import FetchError, fetch_vacancies
from src.main import run_check, run_fetch
from src.normalize import extract_list, normalize_vacancy


def response(*items):
    return {"results": {"vacancies": [{"vacancy": x} for x in items]}}


class IntegrationTest(unittest.TestCase):
    def test_api_structure_and_two_pages_dedup(self):
        first = response({"id": "a", "job-name": "Повар", "salary_min": 30000,
                          "salary_max": 40000, "duty": "Приготовление пищи",
                          "addresses": {"address": [{"location": "Калининград, ул. Мира 1"}]},
                          "vac_url": "https://example.org/a"}, {"id": "b"})
        second = response({"id": "b"}, {"id": "c"})
        with patch("src.fetcher._get_json", side_effect=[first, second]) as get:
            vacancies = fetch_vacancies("https://example.org/api", pages=2, page_size=2, max_vacancies=4)
        self.assertEqual([v["id"] for v in vacancies], ["a", "b", "c"])
        self.assertIn("offset=2", get.call_args_list[1].args[0])
        self.assertEqual(vacancies[0]["title"], "Повар")
        self.assertEqual(vacancies[0]["salary_min"], 30000)
        self.assertEqual(vacancies[0]["address"], "Калининград, ул. Мира 1")
        self.assertEqual(vacancies[0]["url"], "https://example.org/a")

    def test_failed_second_page_preserves_report(self):
        with tempfile.TemporaryDirectory() as d:
            report, error = Path(d)/"report.json", Path(d)/"error.json"
            report.write_text('old report')
            cfg = {"source_url": "https://example.org", "pages": 2, "page_size": 1,
                   "max_vacancies": 2, "timeout_sec": 1, "error_report": str(error), "ai": {"enabled": False}}
            with patch("src.fetcher._get_json", side_effect=[response({"id": "1"}), FetchError("down")]):
                self.assertEqual(run_fetch(cfg, str(report)), 2)
            self.assertEqual(report.read_text(), 'old report')
            self.assertEqual(json.loads(error.read_text())["error"], "down")

    def test_unknown_api_shape_is_error_not_empty_page(self):
        with patch("src.fetcher._get_json", return_value={"error": "bad gateway"}):
            with self.assertRaises(FetchError):
                fetch_vacancies("https://example.org")
        self.assertEqual(extract_list(response()), [])

    def test_local_check_empty_and_nested(self):
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d)/"input.json", Path(d)/"output.json"
            inp.write_text(json.dumps(response({"id": "1", "job-name": "Повар"})))
            with patch("src.fetcher._get_json", side_effect=AssertionError("network")):
                self.assertEqual(run_check({"ai": {"enabled": False}}, str(inp), str(out)), 0)
            self.assertEqual(json.loads(out.read_text())["vacancies"][0]["id"], "1")
            inp.write_text("[]")
            self.assertEqual(run_check({"ai": {"enabled": False}}, str(inp), str(out)), 0)

    def test_model_output_influences_report(self):
        class MockModel:
            name = "mock-model"
            def analyze(self, v):
                return [{"topic":"Описание", "field":"description", "quote":"нестандартные часы", "problem":"Неясно", "recommendation":"Укажите часы", "source":"ai", "severity":"средняя"}]
        v = normalize_vacancy({"id": "1", "description": "нестандартные часы " + "x" * 100})
        with patch("src.analyzer.build_ai_analyzer", return_value=(MockModel(), "mock-model")):
            result, status = analyze_batch([v], {"enabled": True})
        self.assertTrue(any(r["source"] == "ai" for r in result[0]["remarks"]))
        self.assertEqual(status, "mock-model")

    def test_invalid_model_json_triggers_fallback_and_bad_quotes_rejected(self):
        with self.assertRaises(ValueError):
            _parse_model_json("not json")
        v = normalize_vacancy({"id": "1", "description": "Стабильная зарплата. Полный соцпакет."})
        bad = {"topic": "Оплата", "field": "description", "quote": "нет | выдумано", "problem": "x", "recommendation": "y"}
        self.assertIsNone(_valid_remark(bad, v))
        class BadModel:
            def analyze(self, v):
                return _parse_model_json("bad")
        with patch("src.analyzer.build_ai_analyzer", return_value=(BadModel(), "ai")):
            result, status = analyze_batch([v], {"enabled": True})
        self.assertEqual(status, "fallback")
        self.assertTrue(any(r["source"] == "fallback" for r in result[0]["remarks"]))

if __name__ == "__main__":
    unittest.main()
