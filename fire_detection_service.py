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
import logging
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fire_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FireDetectionService:
    def __init__(self):
        # Конфигурация
        self.config = {
            'YOLO_MODEL': 'best.pt',
            'SMTP_SERVER': 'smtp.timeweb.ru',
            'SMTP_PORT': 587,
            'EMAIL': 'admin@ar-ucheba.ru',
            'PASSWORD': 'gxgmsdt4s9',
            'ALERT_EMAIL': 'nicksrnk@gmail.com',
            'DJANGO_API_URL': 'http://83.222.9.213:9999/api/fire-detection/',
            'API_TIMEOUT': 5,
            'ALERT_COOLDOWN': 300,
            'VIDEO_SOURCE': 'fire_video.mp4'  # или номер камеры (0) или RTSP-поток
        }
        
        # Инициализация модели
        self.model = YOLO(self.config['YOLO_MODEL'])
        self.last_alert_time = 0

    def prepare_frame_data(self, frame):
        """Подготовка изображения для отправки"""
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return base64.b64encode(buffer).decode('utf-8')

    def send_to_django(self, alert_data):
        """Отправка данных в Django API с улучшенной обработкой base64"""
        try:
            # Подготовка изображения
            _, buffer = cv2.imencode('.jpg', alert_data['frame'])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Формируем payload
            payload = {
                "frame": frame_b64,  # Отправляем чистый base64 без префикса
                "confidence": alert_data["confidence"],
                "detection_type": alert_data["detection_type"],
                "coordinates": alert_data["coordinates"],
                "detection_time": alert_data["timestamp"]
            }
            
            response = requests.post(
                self.config['DJANGO_API_URL'],
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.config['API_TIMEOUT']
            )
            
            if response.status_code == 201:
                return response.json()
            raise Exception(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Django API Error: {str(e)}")
            return None

    def send_email_alert(self, alert_data, django_response):
        """Упрощенная отправка email со ссылкой на изображение в Django"""
        try:
            # Базовые настройки письма
            msg = MIMEMultipart()
            msg['From'] = f"Fire Detection <{self.config['EMAIL']}>"
            msg['To'] = self.config['ALERT_EMAIL']
            msg['Subject'] = "🚨 Обнаружено возгорание"
            
            # Получаем URL изображения из Django API
            image_url = f"{self.config['DJANGO_API_URL'].rsplit('/', 2)[0]}/events/{django_response['event_id']}/image/"
            
            # HTML содержимое письма
            html_content = f"""
            <html>
                <body>
                    <h2>Система обнаружения пожаров</h2>
                    <p><strong>Тип:</strong> {alert_data['detection_type']}</p>
                    <p><strong>Уверенность:</strong> {alert_data['confidence']:.2%}</p>
                    <p><strong>Время:</strong> {datetime.fromisoformat(alert_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>ID события:</strong> {django_response['event_id']}</p>
                    <p><a href="{image_url}">Ссылка на изображение с обнаружением</a></p>
                    <p>Для просмотра перейдите в <a href="{self.config['DJANGO_API_URL'].rsplit('/', 2)[0]}">систему мониторинга</a></p>
                </body>
            </html>
            """
            
            # Прикрепляем HTML версию
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # Текстовая версия для клиентов без поддержки HTML
            text_content = f"""
            Система обнаружения пожаров
            Тип: {alert_data['detection_type']}
            Уверенность: {alert_data['confidence']:.2%}
            Время: {datetime.fromisoformat(alert_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
            ID события: {django_response['event_id']}
            Ссылка на изображение: {image_url}
            """
            msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
            
            # Отправка письма
            with smtplib.SMTP(self.config['SMTP_SERVER'], self.config['SMTP_PORT']) as server:
                server.starttls()
                server.login(self.config['EMAIL'], self.config['PASSWORD'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent with image link: {image_url}")
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}", exc_info=True)
            return False

    def process_frame(self, frame):
        """Обработка кадра"""
        results = self.model(frame, classes=[0], conf=0.5)
        detection = None
        
        for result in results:
            for box in result.boxes:
                if box.conf[0] > 0.5:
                    detection = {
                        'frame': frame.copy(),
                        'confidence': float(box.conf[0]),
                        'detection_type': 'fire',
                        'coordinates': {
                            'x1': int(box.xyxy[0][0]),
                            'y1': int(box.xyxy[0][1]),
                            'x2': int(box.xyxy[0][2]),
                            'y2': int(box.xyxy[0][3])
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    logger.info(f"Fire detected with confidence: {box.conf[0]:.2f}")
        
        return detection

    def run(self):
        """Основной цикл работы сервиса"""
        logger.info("Starting Fire Detection Service")
        
        cap = cv2.VideoCapture(self.config['VIDEO_SOURCE'])
        if not cap.isOpened():
            logger.error("Error opening video source")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Video frame read error, restarting capture...")
                    cap.release()
                    time.sleep(5)
                    cap = cv2.VideoCapture(self.config['VIDEO_SOURCE'])
                    continue
                
                detection = self.process_frame(frame)
                
                if detection and (time.time() - self.last_alert_time) > self.config['ALERT_COOLDOWN']:
                    django_response = self.send_to_django(detection)
                    if django_response:
                        if self.send_email_alert(detection, django_response):
                            self.last_alert_time = time.time()
                
                time.sleep(0.1)  # Небольшая задержка для снижения нагрузки
                
        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Service error: {str(e)}")
        finally:
            cap.release()
            logger.info("Service stopped")

if __name__ == "__main__":
    service = FireDetectionService()
    service.run()