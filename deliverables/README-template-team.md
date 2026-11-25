# Vacancy Check

[Кратко опишите, какую задачу решает проект.]

## Участники и вклад

| Участник | Вклад |
|---|---|
| [Имя 1] | [Фактический вклад] |
| [Имя 2] | [Фактический вклад] |
| [Имя 3] | [Фактический вклад] |
| [Имя 4] | [Фактический вклад] |
| [Имя 5] | [Фактический вклад] |

## Ссылки

- Репозиторий: [ссылка]
- Проверенный коммит: [ссылка или SHA]
- Видео демонстрации: [публичная ссылка]
- Учебные данные: `data/kantkod_sample_vacancies.json`
- Пример отчёта: `data/example_report.json`

## Возможности

[Кратко перечислите возможности проекта.]

## Установка

Нужны Python 3.13 и [Ollama](https://ollama.com/download).
Откройте терминал в папке проекта.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если политика PowerShell запрещает активацию, вместо неё можно установить зависимости напрямую:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

В этом случае в следующих командах вместо `python` используйте `.\.venv\Scripts\python.exe`.

### Linux / macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Модель

Запустите Ollama и загрузите модель:

```bash
ollama pull gemma4:e2b
```

[Вставить скриншот установки модели.]

## Настройки

[Опишите основные настройки файла `config.yaml`.]

[Вставить скриншот настроек.]

## Запуск

Проверка локального файла:

```bash
python main.py check --input data/kantkod_sample_vacancies.json
```

Загрузка через API:

```bash
python main.py run
```

Запуск по расписанию:

```bash
python main.py schedule
```

Проверка без ИИ:

```bash
python main.py check --input data/kantkod_sample_vacancies.json --no-ai
```

[Вставить скриншот запуска.]

## Результат

[Опишите расположение и формат отчёта, а также источники замечаний `rules` и `llm`.]

[Вставить скриншот фрагмента отчёта.]

## Проверка

```bash
python -m unittest discover -s tests -v
```

```bash
python -m tools.verify_model --output reports/model-verification.json
```

[Укажите, что именно проверялось и какие ограничения остались в итоговой версии.]
