import json
import os
from datetime import datetime

def save_report(vacancies, issues_dict, mode, model, report_path):
    report_data = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "model": model,
        "vacancies": []
    }

    for v in vacancies:
        v_id = v.get("id")
        # на всякий пожарный
        if not v_id:
            continue

        report_data["vacancies"].append({
            "id": v_id,
            "url": v.get("vac_url", ""),
            "issues": issues_dict.get(v_id, [])
        })

    # временный файл
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    tmp_path = report_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, report_path)
    print(f"Отчет успешно сохранен в {report_path}")

def deduplicate_issues(issues):
    """Консервативный вариант дедупликации коллеги для объединённой схемы.

    Не склеиваем разные цитаты или проблемы только потому, что совпали тема и поле.
    Правило и результат модели с полностью совпадающими содержательными полями
    считаем дублем; порядок исходных замечаний не меняем.
    """
    seen, result = set(), []
    for issue in issues:
        key = tuple(' '.join(str(issue.get(field) or '').casefold().split())
                    for field in ('topic', 'type', 'field', 'quote', 'problem', 'recommendation'))
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def save_integrated_report(results, mode, requested_model, report_path, ai_status):
    """Командный формат issues плюс прозрачный статус ИИ для каждой вакансии."""
    actual_model_statuses = {'gemma-local', 'transformers-local', 'api', 'mixed'}
    model_used = any(v.get('ai_status') in actual_model_statuses for v in results)
    report_data = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'mode': mode,
        'model': requested_model if model_used else None,
        'requested_model': requested_model,
        'ai_status': ai_status,
        'total': len(results),
        'ai_cache_hits': sum(1 for v in results if v.get('ai_cache_hit')),
        'ai_generated_issues': sum(1 for v in results for r in v['remarks'] if r.get('source') == 'ai'),
        'ai_selected_recommendations': sum(1 for v in results for r in v['remarks']
                                           if r.get('recommendation_source') == 'ai-template-choice'),
        'vacancies': [
            {'id': v.get('id', ''), 'url': v.get('url') or '', 'title': v.get('title', ''),
             'ai_status': v.get('ai_status', 'off'), 'ai_cache_hit': bool(v.get('ai_cache_hit')),
             'issues': v['remarks']}
            for v in results
        ],
    }
    parent = os.path.dirname(report_path) or '.'
    os.makedirs(parent, exist_ok=True)
    tmp_path = report_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, report_path)
