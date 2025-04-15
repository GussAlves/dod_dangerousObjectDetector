from ultralytics import YOLO
import os
from datetime import datetime
import shutil
import cv2

class VideoProcessor:
    def __init__(self, input_dir="data/input", processed_dir="data/processed", reports_dir="data/reports"):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.model = None
        
        # Cria diretórios se não existirem
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.load_model()
    
    def load_model(self):
        """Carrega o modelo YOLO"""
        if not os.path.exists("yolov8n.pt"):
            raise FileNotFoundError("Modelo YOLOv8 não encontrado!")

        try:
            self.model = YOLO("yolov8n.pt", task='detect')  # Ou seu modelo customizado
            return True
        except Exception as e:
            print(f"Erro crítico ao carregar modelo: {e}")
            QMessageBox.critical(None, "Erro", "Falha ao carregar o modelo YOLO")
            return False
    
    def process_video(self, video_path, user_email):
        try:
            if not self.model:
                return False, "Modelo não foi carregado corretamente"
            
            # Verifica se o vídeo pode ser aberto
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Não foi possível abrir o vídeo"
            cap.release()

            # Copia o vídeo para a pasta de input
            input_copy = os.path.join(self.input_dir, os.path.basename(video_path))
            shutil.copy2(video_path, input_copy)
            
            # Processa o vídeo
            results = self.model.track(
                source=input_copy,
                save=True,
                project=self.processed_dir,
                name=os.path.splitext(os.path.basename(video_path))[0]
            )
            
            # Gera relatório
            report_content = self._generate_report(video_path, user_email, results)
            report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return True, f"Processamento concluído! Relatório salvo em {report_path}"
        except Exception as e:
            return False, f"Erro no processamento: {str(e)}"
    
    def _generate_report(self, video_path, user_email, results):
        report = f"""
        Relatório de Processamento - YOLOv8
        Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Usuário: {user_email}
        Vídeo processado: {os.path.basename(video_path)}
        
        Estatísticas:
        - Tempo por frame: {results[0].speed['inference']} ms
        - Objetos detectados: {len(results[0].boxes)}
        
        Detalhes das detecções:
        """
        
        for i, result in enumerate(results):
            report += f"\nFrame {i+1}:\n"
            for box in result.boxes:
                report += f"- {box.cls[0]}: confiança {box.conf[0]:.2f}\n"
        
        return report