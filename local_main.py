import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
import time
from datetime import datetime
import numpy as np
import requests
import base64
import json

# Конфигурация
class Config:
    # Модель YOLO
    YOLO_MODEL = "best.pt"  # Ваша кастомная модель
    
    # Настройки SMTP
    SMTP_SERVER = "smtp.timeweb.ru"
    SMTP_PORT = 587
    EMAIL = "admin@ar-ucheba.ru"
    PASSWORD = "gxgmsdt4s9"  # Пароль для SMTP
    ALERT_EMAIL = "nicksrnk@gmail.com"  # Получатель уведомлений
    
    # Настройки Django API
    DJANGO_API_URL = "http://127.0.0.1:8000/api/fire-detection/"
    API_TIMEOUT = 5  # Таймаут запроса в секундах
    
    # Настройки оповещений
    ALERT_COOLDOWN = 300  # Интервал между уведомлениями (5 минут)
    CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности для детекции


# Инициализация
model = YOLO(Config.YOLO_MODEL)
last_alert_time = 0

def prepare_alert_data(frame, confidence, coordinates):
    """Подготавливает все данные для оповещения"""
    timestamp = datetime.now().isoformat()
    frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
    
    return {
        "frame": frame_data,
        "confidence": float(confidence),
        "detection_type": "fire",
        "coordinates": coordinates,
        "timestamp": timestamp  # Явное добавление времени
    }

# В вашем скрипте обнаружения
def send_to_django(alert_data):
    try:
        response = requests.post(
            Config.DJANGO_API_URL,
            json={
                "frame": f"data:image/jpeg;base64,{alert_data['frame']}",
                "confidence": alert_data["confidence"],
                "detection_type": alert_data["detection_type"],
                "coordinates": alert_data["coordinates"]
            },
            headers={'Content-Type': 'application/json'},
            timeout=Config.API_TIMEOUT
        )
        return response.json()
    except Exception as e:
        print(f"Django API Error: {str(e)}")
        return None

def send_email_alert(alert_data, django_response):
    """Отправка email с информацией о пожаре"""
    try:
        msg = MIMEMultipart()
        msg["From"] = Config.EMAIL
        msg["To"] = Config.ALERT_EMAIL
        msg["Subject"] = f"🚨 ПОЖАР ОБНАРУЖЕН ({alert_data['detection_type']})"
        
        # Текст письма
        body = f"""
        <h2>Система обнаружения пожаров</h2>
        <p><strong>Тип:</strong> {alert_data['detection_type']}</p>
        <p><strong>Уверенность:</strong> {alert_data['confidence']:.2%}</p>
        <p><strong>Время:</strong> {datetime.fromisoformat(alert_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>ID события:</strong> {django_response.get('event_id', 'N/A')}</p>
        <img src="cid:fire_image" style="max-width:600px;border:1px solid #ddd">
        """
        
        # Добавляем изображение
        img_data = base64.b64decode(alert_data['frame'])
        image = MIMEImage(img_data)
        image.add_header('Content-ID', '<fire_image>')
        msg.attach(image)
        msg.attach(MIMEText(body, 'html'))
        
        # Отправка
        with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.EMAIL, Config.PASSWORD)
            server.send_message(msg)
        
        print("Email alert sent successfully!")
        return True
    except Exception as e:
        print(f"Email Error: {str(e)}")
        return False

def process_detection(frame, box):
    """Обработка обнаруженного пожара"""
    global last_alert_time
    
    coordinates = {
        'x1': int(box.xyxy[0][0]),
        'y1': int(box.xyxy[0][1]),
        'x2': int(box.xyxy[0][2]),
        'y2': int(box.xyxy[0][3])
    }
    
    # Проверка временного интервала
    current_time = time.time()
    if current_time - last_alert_time < Config.ALERT_COOLDOWN:
        print(f"Слишком рано для нового уведомления. Осталось: {Config.ALERT_COOLDOWN - (current_time - last_alert_time):.0f} сек.")
        return False
    
    # Подготовка данных
    alert_data = prepare_alert_data(
        frame=frame,
        confidence=float(box.conf[0]),
        coordinates=coordinates
    )
    
    # Отправка в Django
    django_response = send_to_django(alert_data)
    if not django_response:
        return False
    
    # Отправка email
    if send_email_alert(alert_data, django_response):
        last_alert_time = current_time
        return True
    return False

# Основной цикл обработки
def main():
    cap = cv2.VideoCapture("fire_video.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция
        results = model(frame, classes=[0], conf=0.5)
        
        # Обработка результатов
        for result in results:
            for box in result.boxes:
                if box.conf[0] > 0.5:
                    # Визуализация
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Fire {box.conf[0]:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Обработка обнаружения
                    process_detection(frame.copy(), box)
        
        cv2.imshow("Fire Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()