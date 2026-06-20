import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

RAW_DATA_PATH = Path(config["data"]["raw_dir"]) / config["data"]["raw_file"]
PROCESSED_DATA_PATH = Path(config["data"]["processed_dir"]) / config["data"]["processed_file"]
SAMPLE_DATA_PATH = Path(config["data"]["processed_dir"]) / config["data"]["sample_file"]
REFERENCE_DATE = pd.Timestamp(config["preprocessing"]["reference_date"])
BURNOUT_THRESHOLD = config["preprocessing"]["burnout_threshold"]
SAMPLE_SIZE = config["preprocessing"]["sample_size"]

CATEGORICAL_ENCODINGS = {
    "wfh_setup_available": {"Yes": 1, "No": 0},
    "company_type": {"Service": 0, "Product": 1},
    "gender": {"Male": 0, "Female": 1},
}

FEATURE_COLUMNS = [
    "gender",
    "company_type",
    "wfh_setup_available",
    "designation",
    "resource_allocation",
    "mental_fatigue_score",
    "tenure_months",
    "fatigue_resource_ratio",
    "is_senior",
    "high_workload",
]

TARGET_COLUMN = "burnout_target"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Загружает исходный CSV-файл с учётом его формата."""
    if not path.exists():
        raise FileNotFoundError(
            f"Файл сырых данных не найден: {path}. "
            "Положите burnout_raw.csv в data/raw/."
        )

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8",
    )

    # Нормализация имён колонок: lower_case с подчёркиваниями
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет полный пайплайн предобработки к сырому датафрейму."""
    df = df.copy()

    # 1. Создание целевой переменной: бинаризация Burn Rate
    df[TARGET_COLUMN] = (df["burn_rate"] > BURNOUT_THRESHOLD).astype(int)

    # 2. Обработка пропусков
    # Числовые признаки — заполняем медианой (устойчиво к выбросам)
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(TARGET_COLUMN)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Категориальные признаки — заполняем модой (самое частое значение)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Feature Engineering — создание новых признаков
    # Дата найма в формат datetime
    df["date_of_joining"] = pd.to_datetime(
        df["date_of_joining"], format="%d.%m.%Y", errors="coerce"
    )

    # Стаж в месяцах (относительно фиксированной даты для воспроизводимости)
    df["tenure_months"] = (
        (REFERENCE_DATE - df["date_of_joining"]).dt.days / 30.44
    ).clip(lower=0)

    # Отношение ментальной усталости к выделенным ресурсам
    df["fatigue_resource_ratio"] = df["mental_fatigue_score"] / (
        df["resource_allocation"] + 1
    )

    # Бинарные флаги
    df["is_senior"] = (df["designation"] >= 3).astype(int)
    df["high_workload"] = (df["resource_allocation"] >= 6).astype(int)

    # 4. Кодирование категориальных признаков в числовой формат
    for col, mapping in CATEGORICAL_ENCODINGS.items():
        df[col] = df[col].map(mapping)

    # 5. Выбор финальных признаков + целевая переменная
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Сохраняет обработанный датасет и небольшой сэмпл для тестов."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Полный датасет в стандартном CSV формате (запятая, точка)
    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        sep=",",
        decimal=".",
    )

    # Сэмпл для быстрого запуска тестов
    df.head(SAMPLE_SIZE).to_csv(
        SAMPLE_DATA_PATH,
        index=False,
        sep=",",
        decimal=".",
    )

    print(
        f"Сохранено: {len(df)} строк в {PROCESSED_DATA_PATH}, "
        f"сэмпл {SAMPLE_SIZE} строк в {SAMPLE_DATA_PATH}"
    )


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Загружает обработанный датасет (для ноутбуков и инференса)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Обработанный файл не найден: {path}. "
            "Запустите python -m src.data.prepare_data."
        )
    return pd.read_csv(path)


def get_feature_columns() -> list[str]:
    """Возвращает список признаков (для валидации входных данных в API)."""
    return FEATURE_COLUMNS.copy()


def split_xy(df: pd.DataFrame):
    """Разделяет датафрейм на матрицу признаков X и вектор таргета y."""
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    """Запускает полный пайплайн: загрузка -> предобработка -> сохранение."""
    print(f"Загрузка сырых данных из {RAW_DATA_PATH}...")
    raw_df = load_raw_data()
    print(f"Загружено строк: {raw_df.shape[0]}, колонок: {raw_df.shape[1]}")

    print("Применение пайплайна предобработки...")
    processed_df = preprocess_data(raw_df)

    save_processed_data(processed_df)
    print(
        f"Итоговая матрица: {processed_df.shape[0]} строк, "
        f"{len(FEATURE_COLUMNS)} признаков + таргет"
    )


if __name__ == "__main__":
    main()