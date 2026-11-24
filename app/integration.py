"""Адаптер командного конвейера к анализатору с сохранением исходных полей."""
from src.analyzer import analyze_batch
from src.normalize import normalize_vacancy


def analyze_raw_vacancies(raw_data, ai_config):
    # Не используем здесь app.normalize: он теряет исходные поля для цитат ИИ.
    # Проверочный режим должен анализировать все записи, включая повторные ID.
    vacancies = [normalize_vacancy(raw) for raw in raw_data]
    return analyze_batch(vacancies, ai_config)
