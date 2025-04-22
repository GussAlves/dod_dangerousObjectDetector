from ultralytics import YOLO
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import time
import numpy as np

# Configurações de e-mail
EMAIL_CONFIG = {
    'sender': 'astolfotheduck@gmail.com',
    'password': 'xekhhaakldgqbudh',  
    'receiver': 'gadaguitarra@gmail.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

# Controle de tempo entre e-mails
last_email_time = 0
EMAIL_COOLDOWN = 300  # 5 minutos entre e-mails

def send_alert_email(object_type, detection_frame):
    """Função para enviar e-mail de alerta com imagem anexada"""
    try:
        # Configurar mensagem
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']
        msg['Subject'] = f"ALERTA: {object_type} detectado!"
        
        # Corpo do e-mail em HTML
        body = f"""
        <h2>Alerta de Segurança</h2>
        <p>Objeto <strong>{object_type}</strong> detectado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        <p>Este é um alerta automático do sistema de vigilância.</p>
        <p><img src="cid:detection_image" width="640"></p>
        """
        msg.attach(MIMEText(body, 'html'))

        # Converter frame para JPEG em memória
        _, buffer = cv2.imencode('.jpg', detection_frame)
        detection_bytes = buffer.tobytes()

        # Anexar imagem
        img_part = MIMEImage(detection_bytes)
        img_part.add_header('Content-ID', '<detection_image>')
        img_part.add_header('Content-Disposition', 'attachment; filename="detection.jpg"')
        msg.attach(img_part)

        # Enviar e-mail
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.send_message(msg)
        
        print(f"E-mail com imagem enviado para {EMAIL_CONFIG['receiver']}")
        return True
    
    except Exception as e:
        print(f"Falha ao enviar e-mail: {str(e)}")
        return False

def main():
    global last_email_time
    
    # Iniciar câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera")
        return

    # Carregar modelo YOLO
    model = YOLO("treinamento_yolo/runs/detect/treino_customizado7/weights/best.pt")

    print("Sistema iniciado - pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Fazer uma cópia do frame original para possível envio
        original_frame = frame.copy()

        # Detecção de objetos
        results = model.track(frame, persist=True)
        
        # Verificar detecções
        knife_detected = False
        detection_frame = None
        
        for result in results:
            if result.boxes is not None:
                for box, cls in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu()):
                    class_name = model.names[int(cls)]
                    
                    # Se detectar faca
                    if class_name == "knife":
                        knife_detected = True
                        # Criar frame com a detecção para enviar
                        detection_frame = result.plot()
                        
            # Mostrar resultados na tela
            frame = result.plot()
        
        # Processar detecção de faca
        if knife_detected and detection_frame is not None:
            current_time = time.time()
            if current_time - last_email_time > EMAIL_COOLDOWN:
                if send_alert_email("knife", detection_frame):
                    last_email_time = current_time
                    print("Alerta de faca enviado com imagem")

        # Exibir frame
        cv2.imshow("Detector de Objetos", frame)
        
        # Sair ao pressionar 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Sistema encerrado")

if __name__ == "__main__":
    main()