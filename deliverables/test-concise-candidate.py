import json
import unittest
from pathlib import Path
from unittest.mock import patch
from app.fetch import load_local
from app.llm import LLMError, analyze_vacancy, cache_key
from app.normalize import normalize_all

class ConciseCandidateTests(unittest.TestCase):
    def setUp(self):
        self.v = normalize_all([v for v in load_local('data/kantkod_sample_vacancies.json') if v['id'] == 'DEMO-39-002'])[0]
        self.cfg = {'provider': 'local', 'model': 'test', 'prompt_strategy': 'concise'}
        self.issue = {'topic':'обязанности', 'type':'ambiguous', 'field':self.v['duty'],
                      'quote':'различных поручений', 'values':[], 'problem':'Неясен перечень поручений',
                      'recommendation':'Какие поручения?'}

    def analyze(self, issues, v=None):
        with patch('app.llm.generate_text', return_value=json.dumps(issues,ensure_ascii=False)):
            return analyze_vacancy(v or self.v, self.cfg)

    def test_unique_exact_field_value_resolved(self):
        self.assertEqual(self.analyze([self.issue])[0]['field'], 'duty')

    def test_ambiguous_value_and_false_quote_rejected(self):
        duplicate = dict(self.v, requirements=self.v['duty'])
        with self.assertRaises(LLMError): self.analyze([self.issue], duplicate)
        with self.assertRaises(LLMError): self.analyze([{**self.issue, 'quote':'отсутствует'}])
        with self.assertRaises(LLMError): self.analyze([{**self.issue, 'recommendation':None}])

    def test_cache_keys_separate_strategies(self):
        self.assertNotEqual(cache_key(self.v,self.cfg), cache_key(self.v,{**self.cfg, 'prompt_strategy':'original'}))
