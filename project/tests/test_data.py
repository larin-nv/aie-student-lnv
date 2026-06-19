"""
Тесты для модуля предобработки данных.

Запуск: pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.prepare_data import (
    load_raw_data,
    preprocess_data,
    get_feature_columns,
    split_xy,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def test_get_feature_columns():
    """Проверка списка признаков."""
    features = get_feature_columns()
    assert len(features) == 10
    assert "mental_fatigue_score" in features
    assert "resource_allocation" in features
    assert "burnout_target" not in features


def test_preprocess_data_pipeline():
    """Тест полного пайплайна предобработки."""
    # Создаем тестовый DataFrame с уже нормализованными именами колонок
    test_data = pd.DataFrame({
        "employee_id": ["test1", "test2"],
        "date_of_joining": ["01.01.2020", "01.01.2021"],
        "gender": ["Male", "Female"],
        "company_type": ["Service", "Product"],
        "wfh_setup_available": ["Yes", "No"],
        "designation": [2.0, 3.0],
        "resource_allocation": [5.0, 7.0],
        "mental_fatigue_score": [4.0, 8.0],
        "burn_rate": [0.3, 0.7]
    })
    
    result = preprocess_data(test_data)
    
    # Проверяем структуру
    assert len(result) == 2
    assert TARGET_COLUMN in result.columns
    assert "tenure_months" in result.columns
    assert "fatigue_resource_ratio" in result.columns
    
    # Проверяем бинаризацию таргета
    assert result[TARGET_COLUMN].iloc[0] == 0  # 0.3 <= 0.5
    assert result[TARGET_COLUMN].iloc[1] == 1  # 0.7 > 0.5
    
    # Проверяем кодирование категорий
    assert result["gender"].iloc[0] == 0  # Male
    assert result["gender"].iloc[1] == 1  # Female
    
    # Проверяем наличие всех признаков
    for col in FEATURE_COLUMNS:
        assert col in result.columns


def test_split_xy():
    """Тест разделения на X и y."""
    # Создаем DataFrame со всеми признаками
    test_data = {col: [0, 1] for col in FEATURE_COLUMNS}
    test_data[TARGET_COLUMN] = [0, 1]
    test_df = pd.DataFrame(test_data)
    
    X, y = split_xy(test_df)
    
    assert len(X.columns) == 10
    assert TARGET_COLUMN not in X.columns
    assert len(y) == 2
    assert list(y) == [0, 1]