import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from app.fetch import ApiError, fetch_vacancies
from app.integration import analyze_raw_vacancies


def response(*items):
    return {'results': {'vacancies': [{'vacancy': item} for item in items]}}


class TeamIntegrationTest(unittest.TestCase):
    def test_adapter_supplies_original_fields_and_model_changes_report(self):
        class Model:
            name = 'gemma-local'

            def analyze(self, vacancy):
                self.raw = vacancy['raw']
                return [{'topic': 'Описание', 'field': 'description',
                         'quote': 'возможны командировки', 'problem': 'Неизвестна частота',
                         'recommendation': 'Уточните частоту командировок.',
                         'source': 'ai', 'severity': 'средняя'}]

        model = Model()
        raw = {'id': '1', 'description': 'Иногда возможны командировки.'}
        with patch('src.analyzer.build_ai_analyzer', return_value=(model, model.name)):
            results, status = analyze_raw_vacancies([raw], {'enabled': True, 'provider': 'local'})
        self.assertEqual(model.raw, raw)
        self.assertEqual(status, 'gemma-local')
        self.assertTrue(any(r['source'] == 'ai' for r in results[0]['remarks']))

    def test_check_retains_duplicate_and_missing_ids_and_reports_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / 'input.json', Path(d) / 'out.json'
            inp.write_text(json.dumps([{'id': 'same'}, {'id': 'same'}, {'duty': 'Работа'}]), encoding='utf8')
            with patch('sys.argv', ['main.py', 'check', '--input', str(inp), '--output', str(out), '--no-ai']):
                main.main()
            report = json.loads(out.read_text(encoding='utf8'))
            self.assertEqual(report['total'], 3)
            self.assertEqual([x['id'] for x in report['vacancies']], ['same', 'same', ''])
            self.assertIsNone(report['model'])
            self.assertEqual(report['ai_status'], 'fallback')
            self.assertTrue(all('issues' in v for v in report['vacancies']))

    def test_check_cli_end_to_end_with_simulated_model(self):
        class Model:
            name = 'gemma-local'

            def analyze(self, vacancy):
                return [{'topic': 'Описание', 'field': 'description', 'quote': 'возможны командировки',
                         'problem': 'Неизвестна частота', 'recommendation': 'Уточните частоту',
                         'source': 'ai', 'severity': 'средняя'}]

        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / 'in.json', Path(d) / 'out.json'
            inp.write_text(json.dumps([{'id': '1', 'description': 'возможны командировки ' + 'x' * 100}]), encoding='utf8')
            with patch('src.analyzer.build_ai_analyzer', return_value=(Model(), 'gemma-local')), \
                 patch('sys.argv', ['main.py', 'check', '--input', str(inp), '--output', str(out)]):
                main.main()
            report = json.loads(out.read_text(encoding='utf8'))
            self.assertEqual(report['ai_status'], 'gemma-local')
            self.assertEqual(report['ai_generated_issues'], 1)
            self.assertTrue(any(r['source'] == 'ai' for r in report['vacancies'][0]['issues']))

    def test_two_pages_deduplicated_before_limit(self):
        pages = [[{'id': '1'}, {'id': '2'}], [{'id': '2'}, {'id': '3'}],
                 [{'id': '4'}]]
        with patch('app.fetch.fetch_page', side_effect=pages) as fetch:
            vacancies = fetch_vacancies('https://example.test', 2, 4, 3)
        self.assertEqual([v['id'] for v in vacancies], ['1', '2', '3', '4'])
        self.assertEqual([call.args[1] for call in fetch.call_args_list], [0, 2, 4])

    def test_bad_page_does_not_overwrite_report_after_retries(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'report.json'
            output.write_text('old', encoding='utf8')
            config = {'source': {'url': 'https://example.test', 'page_size': 1,
                                 'max_vacancies': 2, 'max_retries': 2,
                                 'retry_delay_seconds': 0, 'max_pages': 2},
                      'output': {'report_path': str(output)}, 'error_report': str(Path(d) / 'error.json'),
                      'ai': {'enabled': False}, 'schedule': {'every_hours': 24}}
            class Reply:
                def __init__(self, value): self.value = value
                def raise_for_status(self): pass
                def json(self): return self.value
            replies = [Reply(response({'id': 'one'})), Reply({'error': 'failure'})] * 2
            with patch('main.load_config', return_value=config), \
                 patch('app.fetch.requests.get', side_effect=replies) as get, \
                 patch('main.time.sleep') as sleep, \
                 patch('sys.argv', ['main.py', 'run']):
                with self.assertRaises(SystemExit) as ex:
                    main.main()
            self.assertEqual(ex.exception.code, 1)
            self.assertEqual(output.read_text(encoding='utf8'), 'old')
            self.assertEqual(get.call_count, 4)
            self.assertIn('Некорректный формат', json.loads((Path(d) / 'error.json').read_text(encoding='utf8'))['error'])
            sleep.assert_called_once_with(0)

    def test_report_keeps_model_provenance(self):
        from app.report import save_integrated_report
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'report.json')
            result = [{'id': '1', 'title': 'A', 'url': '', 'ai_status': 'mixed',
                       'remarks': [{'source': 'rule', 'recommendation_source': 'ai-template-choice'}]}]
            save_integrated_report(result, 'check', 'test-model', path, 'mixed')
            report = json.loads(Path(path).read_text(encoding='utf8'))
            self.assertEqual(report['model'], 'test-model')
            self.assertEqual(report['ai_selected_recommendations'], 1)
            result[0]['ai_status'] = 'fallback'
            save_integrated_report(result, 'check', 'test-model', path, 'fallback')
            self.assertIsNone(json.loads(Path(path).read_text(encoding='utf8'))['model'])

    def test_schedule_runs_once_then_waits_and_can_stop(self):
        with tempfile.TemporaryDirectory() as d:
            config = {'source': {}, 'output': {'report_path': str(Path(d) / 'report.json')},
                      'ai': {'enabled': False}, 'schedule': {'every_hours': 2}}
            with patch('main.load_config', return_value=config), \
                 patch('main.fetch_with_retries', return_value=[{'id': '1'}]) as fetch, \
                 patch('main.time.sleep', side_effect=KeyboardInterrupt) as sleep, \
                 patch('sys.argv', ['main.py', 'schedule']):
                with self.assertRaises(SystemExit) as ex:
                    main.main()
            self.assertEqual(ex.exception.code, 0)
            fetch.assert_called_once()
            sleep.assert_called_once_with(7200)
            self.assertTrue(Path(config['output']['report_path']).exists())


if __name__ == '__main__':
    unittest.main()
