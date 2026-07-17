# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы приложения
COPY api_app.py .
COPY model_architecture.py .
COPY model_nn_v5_turbo.pth .
COPY scalers_v5.pkl .

# Открываем порт для FastAPI
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]