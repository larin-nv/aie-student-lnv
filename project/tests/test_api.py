"""
Тесты для API сервиса.

Запуск: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.service.main import app

client = TestClient(app)


def test_health_check():
    """Проверка эндпоинта /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_type"] == "CatBoostClassifier"


def test_predict_high_risk():
    """Тест предсказания для сотрудника с высоким риском выгорания."""
    request_data = {
        "gender": 0,
        "company_type": 0,
        "wfh_setup_available": 0,
        "designation": 2,
        "resource_allocation": 7,
        "mental_fatigue_score": 7,
        "tenure_months": 12,
        "fatigue_resource_ratio": 1,
        "is_senior": 0,
        "high_workload": 0
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "burnout_probability" in data
    assert "risk_level" in data
    assert "top_factors" in data
    
    assert 0.0 <= data["burnout_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(data["top_factors"]) == 3
    
    # Для этого запроса ожидаем высокий риск
    assert data["risk_level"] == "HIGH"
    assert data["burnout_probability"] > 0.7


def test_predict_low_risk():
    """Тест предсказания для сотрудника с низким риском выгорания."""
    request_data = {
        "gender": 1,
        "company_type": 1,
        "wfh_setup_available": 1,
        "designation": 1,
        "resource_allocation": 3,
        "mental_fatigue_score": 2,
        "tenure_months": 6,
        "fatigue_resource_ratio": 0.5,
        "is_senior": 0,
        "high_workload": 0
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["risk_level"] == "LOW"
    assert data["burnout_probability"] < 0.3


def test_predict_invalid_data():
    """Тест обработки невалидных данных."""
    request_data = {
        "gender": "invalid",  # Должно быть int
        "company_type": 0,
        "wfh_setup_available": 0,
        "designation": 2,
        "resource_allocation": 7,
        "mental_fatigue_score": 7,
        "tenure_months": 12,
        "fatigue_resource_ratio": 1,
        "is_senior": 0,
        "high_workload": 0
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


def test_predict_missing_fields():
    """Тест обработки запроса с отсутствующими полями."""
    request_data = {
        "gender": 0,
        "company_type": 0
        # Отсутствуют обязательные поля
    }
    
    response = client.post("/predict", json=request_data)
    # Pydantic использует значения по умолчанию, поэтому запрос пройдет
    assert response.status_code == 200