# Используем лёгкий образ Python 3.10
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Копируем код проекта
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/

# Открываем порт для API
EXPOSE 8000

# Запускаем сервис
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]