"""
REST API сервис для прогнозирования риска выгорания сотрудников.

Эндпоинты:
- GET /health — проверка работоспособности сервиса
- POST /predict — получение предсказания риска выгорания
"""

import os
import time
import uuid
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Загрузка конфигурации
# ---------------------------------------------------------------------------

# Загружаем переменные окружения из .env (если есть)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"

def load_config():
    """Загружает конфигурацию из config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "api": {"host": "0.0.0.0", "port": 8000},
        "model": {"artifacts_dir": "artifacts", "model_file": "model.pkl", "features_file": "feature_columns.pkl"},
        "logging": {"level": "INFO"}
    }

config = load_config()

# ---------------------------------------------------------------------------
# Настройка логирования
# ---------------------------------------------------------------------------

log_level = os.getenv("LOG_LEVEL", config["logging"]["level"])
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация путей
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = BASE_DIR / os.getenv("ARTIFACTS_DIR", config["model"]["artifacts_dir"])
MODEL_PATH = ARTIFACTS_DIR / config["model"]["model_file"]
FEATURES_PATH = ARTIFACTS_DIR / config["model"]["features_file"]

# ---------------------------------------------------------------------------
# Загрузка артефактов при старте приложения
# ---------------------------------------------------------------------------

def load_artifacts():
    """Загружает модель и список признаков из artifacts/."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. "
            "Запустите ноутбук exp02_modeling_and_evaluation.ipynb."
        )
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Файл признаков не найден: {FEATURES_PATH}."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


model: CatBoostClassifier = None
feature_columns: list = None

try:
    model, feature_columns = load_artifacts()
    logger.info(f"Модель загружена: {type(model).__name__}")
    logger.info(f"Признаки: {feature_columns}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")

# ---------------------------------------------------------------------------
# Инициализация FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HR Wellbeing Assistant API",
    description="Сервис прогнозирования риска эмоционального выгорания сотрудников",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Pydantic-модели для валидации входных данных
# ---------------------------------------------------------------------------

class BurnoutRequest(BaseModel):
    """Входные данные для предсказания риска выгорания."""

    gender: int = Field(0, description="Пол: 0 = Male, 1 = Female")
    company_type: int = Field(0, description="Тип компании: 0 = Service, 1 = Product")
    wfh_setup_available: int = Field(0, description="Удаленная работа: 0 = No, 1 = Yes")
    designation: float = Field(2.0, description="Уровень должности (1-5)")
    resource_allocation: float = Field(3.0, description="Выделенные ресурсы (1-10)")
    mental_fatigue_score: float = Field(3.0, description="Ментальная усталость (0-10)")
    tenure_months: float = Field(12.0, description="Стаж в месяцах")
    fatigue_resource_ratio: float = Field(1.0, description="Отношение усталости к ресурсам")
    is_senior: int = Field(0, description="Руководящая должность: 0 = No, 1 = Yes")
    high_workload: int = Field(0, description="Высокая нагрузка: 0 = No, 1 = Yes")


class BurnoutResponse(BaseModel):
    """Ответ сервиса с предсказанием."""

    burnout_probability: float = Field(..., description="Вероятность выгорания (0.0 - 1.0)")
    risk_level: str = Field(..., description="Уровень риска: LOW, MEDIUM, HIGH")
    top_factors: list = Field(..., description="Топ-3 фактора риска")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_risk_level(probability: float) -> str:
    """Определяет уровень риска по вероятности."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def get_top_factors(request: BurnoutRequest, model: CatBoostClassifier) -> list:
    """Возвращает топ-3 фактора риска на основе важности признаков."""
    importances = model.get_feature_importance()
    feature_importance = dict(zip(feature_columns, importances))
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    top_factors = []
    request_dict = request.model_dump()
    for feature, importance in sorted_features[:3]:
        top_factors.append({
            "feature": feature,
            "importance": round(importance, 2),
            "value": request_dict.get(feature)
        })
    
    return top_factors


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    status = "ok" if model is not None else "error"
    
    latency = round((time.time() - start_time) * 1000, 2)
    logger.info(f"/health id={request_id} status=200 latency={latency}ms")
    
    return {
        "status": status,
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=BurnoutResponse)
def predict_burnout(request: BurnoutRequest):
    """
    Предсказание риска выгорания сотрудника.
    
    Возвращает вероятность выгорания, уровень риска и топ-3 фактора.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    if model is None:
        logger.error(f"/predict id={request_id} status=500 error=Model not loaded")
        raise HTTPException(
            status_code=500,
            detail="Модель не загружена. Проверьте наличие файлов в artifacts/."
        )
    
    request_dict = request.model_dump()
    input_data = pd.DataFrame([request_dict])
    
    if set(feature_columns) != set(input_data.columns):
        missing = set(feature_columns) - set(input_data.columns)
        extra = set(input_data.columns) - set(feature_columns)
        logger.error(f"/predict id={request_id} status=400 error=Invalid features")
        raise HTTPException(
            status_code=400,
            detail=f"Несоответствие признаков. Отсутствуют: {missing}, Лишние: {extra}"
        )
    
    input_data = input_data[feature_columns]
    
    probability = float(model.predict_proba(input_data)[0][1])
    risk_level = get_risk_level(probability)
    top_factors = get_top_factors(request, model)
    
    latency = round((time.time() - start_time) * 1000, 2)
    logger.info(f"/predict id={request_id} status=200 latency={latency}ms risk={risk_level}")
    
    return BurnoutResponse(
        burnout_probability=round(probability, 4),
        risk_level=risk_level,
        top_factors=top_factors
    )


# ---------------------------------------------------------------------------
# Точка входа для запуска сервиса
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", config["api"]["host"]),
        port=int(os.getenv("API_PORT", config["api"]["port"]))
    )