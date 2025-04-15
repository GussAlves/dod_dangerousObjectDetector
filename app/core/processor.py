from ultralytics import YOLO
import os
from datetime import datetime

class VideoProcessor:
    def __init__(self, input_dir="data/input", processed_dir="data/processed", reports_dir="data/reports"):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.model = YOLO("yolov8n.pt")  # Carrega o modelo YOLOv8 nano (substitua pelo seu modelo)
        
        # Cria diretórios se não existirem
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def process_video(self, video_path, user_email):
        try:
            # Processa o vídeo com YOLOv8
            results = self.model(video_path, save=True, project=self.processed_dir)
            
            # Gera relatório
            report_content = self._generate_report(video_path, user_email, results)
            report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return True, "Processamento concluído com sucesso!"
        except Exception as e:
            return False, f"Erro no processamento: {str(e)}"
    
    def _generate_report(self, video_path, user_email, results):
        report = f"""
        Relatório de Processamento - YOLOv8
        Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Usuário: {user_email}
        Vídeo processado: {os.path.basename(video_path)}
        
        Estatísticas:
        - Duração: {results[0].speed['inference']} ms por frame
        - Objetos detectados: {len(results[0].boxes)}
        
        Detalhes das detecções:
        """
        
        for i, result in enumerate(results):
            report += f"\nFrame {i+1}:\n"
            for box in result.boxes:
                report += f"- {box.cls[0]}: confiança {box.conf[0]:.2f}\n"
        
        return report