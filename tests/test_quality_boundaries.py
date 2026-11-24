"""Проверки граничных случаев правил и достоверности ссылок на источник.

Синтетические данные. Проверяют известные случаи, не доказывают качество на наборе жюри.
"""
import unittest

from src.ai import FallbackAnalyzer
from src.normalize import normalize_vacancy
from src.rules import analyze_rules


def vacancy(**changes):
    raw = {
        'id': 'quality-1', 'job-name': 'Повар',
        'duty': 'Приготовление блюд по меню, контроль качества и ведение документации.',
        'requirements': 'Опыт работы поваром от 2 лет, профессиональное образование.',
        'salary_min': 40000, 'salary_max': 50000,
        'schedule': '5/2, 09:00–18:00', 'employment': 'Полная занятость',
        'city': 'Советск', 'description': 'Работа в учреждении.',
    }
    raw.update(changes)
    return normalize_vacancy(raw)


def topics(v):
    return [r['topic'] for r in analyze_rules(v)]


class QualityBoundaries(unittest.TestCase):
    def test_pay_in_text_not_marked_vague_due_to_description(self):
        v = vacancy(salary_min=None, salary_max=None, salary='50 000 рублей',
                    description='Стабильная зарплата и своевременная оплата.')
        self.assertNotIn('Оплата', topics(v))
        self.assertFalse(any(x['topic'] == 'Оплата' for x in FallbackAnalyzer().analyze(v)))

    def test_pay_in_description_not_marked_vague_in_fallback(self):
        v = vacancy(salary_min=None, salary_max=None,
                    description='Стабильная зарплата 50 000 рублей в месяц.')
        self.assertNotIn('Оплата', topics(v))
        self.assertFalse(any(x['topic'] == 'Оплата' for x in FallbackAnalyzer().analyze(v)))

    def test_zero_salary_fields_are_not_a_real_salary(self):
        v = vacancy(salary_min=0, salary_max=0)
        self.assertIn('Оплата', topics(v))

    def test_one_real_salary_bound_counts(self):
        v = vacancy(salary_min=40000, salary_max=0)
        self.assertNotIn('Оплата', topics(v))

    def test_salary_not_a_number_does_not_satisfy_pay(self):
        v = vacancy(salary_min=0, salary_max=0, salary='не указана')
        self.assertIn('Оплата', topics(v))

    def test_long_boilerplate_is_not_a_duty_description(self):
        v = vacancy(duty='', description='Дружный коллектив, стабильная организация, удобное расположение. '
                                        'Оформление при приёме в установленном порядке.')
        self.assertIn('Обязанности', topics(v))

    def test_described_duties_count_without_structured_duty(self):
        v = vacancy(duty='', description='Обязанности: готовить блюда по меню, контролировать качество, '
                                        'соблюдать санитарные нормы.')
        self.assertNotIn('Обязанности', topics(v))

    def test_cyrillic_city_in_structured_field_covers_vague_office(self):
        v = vacancy(workPlace='В офисе', city='Советск')
        self.assertNotIn('Место работы', topics(v))

    def test_remote_location_in_description_counts_without_structured_field(self):
        v = vacancy(city='', description='Работа полностью удалённая.')
        self.assertNotIn('Место работы', topics(v))

    def test_remote_and_office_format_is_flagged_for_clarification(self):
        v = vacancy(workPlace='В офисе', description='Возможна удалённая работа.')
        self.assertTrue(any(r['topic'] == 'Место работы' and 'сочетания' in r['problem']
                            for r in analyze_rules(v)))

    def test_negated_remote_does_not_create_format_conflict(self):
        for description in ('Удалённая работа не предусмотрена.', 'Не возможна удалённая работа.'):
            v = vacancy(workPlace='В офисе', description=description)
            self.assertFalse(any(r['topic'] == 'Место работы' and 'сочетания' in r['problem']
                                 for r in analyze_rules(v)))

    def test_nested_api_requirements_are_not_repr_string(self):
        v = vacancy(requirements='', requirement={'experience': 2, 'education': 'Среднее профессиональное'})
        self.assertNotIn("{'", v['requirements'])
        self.assertNotIn('Требования', topics(v))

    def test_explicit_absence_of_required_experience_is_information(self):
        v = vacancy(requirements='Опыт работы не требуется.', requirement={'experience': 0})
        self.assertNotIn('Требования', topics(v))

    def test_soft_skills_quote_exactly_from_input(self):
        raw = {'requirements': 'Коммуникабельность; ОТВЕТСТВЕННОСТЬ'}
        v = normalize_vacancy(raw)
        remark = next(r for r in analyze_rules(v) if r['topic'] == 'Требования' and r['quote'])
        self.assertIn(remark['quote'], raw[remark['field']])

    def test_skills_fallback_does_not_assert_missing_job_type_when_present(self):
        v = vacancy(description='Официальное трудоустройство.')
        self.assertFalse(any(x['topic'] == 'Тип занятости' for x in FallbackAnalyzer().analyze(v)))

    def test_salary_amount_not_inferred_from_unrelated_number(self):
        v = vacancy(salary_min=0, salary_max=0, description='Опыт 3 года, офис на улице Мира, 12.')
        self.assertIn('Оплата', topics(v))

    def test_description_length_does_not_hide_missing_requirements(self):
        v = vacancy(requirements='', description='Дружный коллектив и удобное расположение. '
                                               'Стабильная организация и комфортные условия.')
        self.assertIn('Требования', topics(v))

    def test_zero_nested_experience_is_not_assumed_to_be_no_experience(self):
        v = vacancy(requirements='', requirement={'experience': 0}, description='Работа в учреждении.')
        self.assertIn('Требования', topics(v))

    def test_list_skill_quote_references_actual_item(self):
        v = vacancy(requirements=['Коммуникабельность', 'ОТВЕТСТВЕННОСТЬ'])
        remark = next(r for r in analyze_rules(v) if r['topic'] == 'Требования' and r['quote'])
        self.assertIn(remark['field'], ('requirements[0]', 'requirements[1]'))
        index = int(remark['field'][-2])
        self.assertIn(remark['quote'], v['raw']['requirements'][index])

    def test_fallback_does_not_scan_unrelated_employer_field(self):
        v = vacancy(companyName='Дружный коллектив')
        self.assertFalse(any(r['field'] == 'companyName' for r in FallbackAnalyzer().analyze(v)))

    def test_no_claim_of_misleading_growth_is_asserted(self):
        v = vacancy(description='Возможность роста.')
        remarks = FallbackAnalyzer().analyze(v)
        self.assertTrue(remarks)
        self.assertFalse(any('вводит в заблуждение' in r['problem'].lower() for r in remarks))

if __name__ == '__main__':
    unittest.main()
