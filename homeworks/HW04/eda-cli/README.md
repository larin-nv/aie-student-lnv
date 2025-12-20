# HW04 – HTTP-сервис качества датасетов поверх проекта eda-cli (FastAPI + REST API).

HTTP-сервис качества датасетов поверх проекта eda-cli (FastAPI + REST API).
Используется в рамках Задания 04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (HW04):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).


```bash
uv run eda-cli sample --count 7 data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`);
- `--count` - количество случайных строк (по умолчанию `4`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

Параметры:

- `--out-dir` – каталог для отчёта (по умолчанию `reports`);
- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`);
- `--max-hist-columns` - максимум числовых колонок для гистограмм (по умолчанию `6`);
- `--top-k-categories` - количество top-значений для категориальных признаков (по умолчанию `5`);
- `--title`- заголовок отчёта (по умолчанию `EDA-отчёт`);
- `--min-missing-share` - порог доли пропусков проблемной колонки (по умолчанию `0.05`).

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

### Вызов eda-cli report с новыми опциями

```bash
uv run eda-cli report data/example.csv --out-dir reports --max-hist-columns 9 --top-k-categories 4 --title Отчёт_EDA --min-missing-share 0.07
```

## Тесты

```bash
uv run pytest -q
```

## HTTP сервис

Запуск сервиса
```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

URL после запуска

- `http://127.0.0.1:8000/docs` - простейший health-check сервиса.

Системный эдпоинт:

- `/health` - простейший health-check сервиса.

Эндпоинты эвристической оценки качества:

- `/quality` - эндпоинт-заглушка, который принимает агрегированные признаки датасета и возвращает эвристическую оценку качества;
- `/quality-from-csv` - эндпоинт, который принимает CSV-файл, запускает EDA-ядро (summarize_dataset + missing_table + compute_quality_flags) и возвращает оценку качества данных;
- `/quality-flags-from-csv` - эндпоинт, который принимает CSV-файл, запускает EDA-ядро и возвращает флаги качества данных.

Эндпоинт по выводу данных CSV:

- `/dataset-sample` - эндпоинт, который принимает CSV-файл и количество строк (n) и возвращает случайные строки из CSV-файла.

