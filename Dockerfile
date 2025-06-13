# Используем официальный образ Python
FROM python:3.13-slim

# Устанавливаем зависимости OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Копируем модель YOLO
COPY best.pt ./best.pt

# Команда для запуска сервиса
CMD ["python", "fire_detection_service.py"]