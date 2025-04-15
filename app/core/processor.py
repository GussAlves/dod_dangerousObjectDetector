from ultralytics import YOLO
import os
from datetime import datetime
import shutil
import cv2
import json
from PyQt5.QtWidgets import QMessageBox

class VideoProcessor:
    def __init__(self, input_dir="data/input", processed_dir="data/processed", reports_dir="data/reports"):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.model = None
        self.cache_file = os.path.join(processed_dir, "processing_cache.json")
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self._load_model()
        self.cache = self._load_cache()

    def _load_model(self):
        """Carrega o modelo YOLO"""
        try:
            self.model = YOLO('yolov8n.pt', task='detect')
            return True
        except Exception as e:
            error_msg = f"Erro ao carregar modelo: {str(e)}"
            print(error_msg)
            QMessageBox.critical(None, "Erro", "Falha ao carregar o modelo YOLO\n" + error_msg)
            return False

    def _load_cache(self):
        """Carrega o cache de processamentos anteriores"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_cache(self):
        """Salva o cache em arquivo"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_video_hash(self, video_path):
        """Gera um hash único para o vídeo baseado em tamanho e data de modificação"""
        try:
            stat = os.stat(video_path)
            return f"{stat.st_size}_{stat.st_mtime}"
        except Exception:
            return None

    def is_video_processed(self, video_path, base_name):
        """Verifica se o vídeo já foi processado e está atualizado"""
        video_hash = self.get_video_hash(video_path)
        if not video_hash:
            return False
        
        if base_name in self.cache:
            cached_data = self.cache[base_name]
            if isinstance(cached_data, dict):
                cached_hash = cached_data.get('hash')
                output_dir = os.path.join(self.processed_dir, base_name)
                if cached_hash == video_hash and os.path.exists(output_dir):
                    return True
        return False

    def process_video(self, video_path, user_email, force_reprocess=False, progress_callback=None):
        """
        Processa um vídeo com possibilidade de acompanhamento de progresso
        
        Args:
            video_path: Caminho do vídeo
            user_email: Email do usuário
            force_reprocess: Se True, ignora cache
            progress_callback: Função para reportar progresso (recebe % e mensagem)
        """
        try:
            if not self.model:
                return False, "Modelo não foi carregado corretamente"
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(self.processed_dir, base_name)
            
            # Verifica se deve reprocessar
            if not force_reprocess and self.is_video_processed(video_path, base_name):
                if progress_callback:
                    progress_callback(100, "Vídeo já processado - usando cache")
                return True, f"Vídeo já processado anteriormente. Resultados em: {output_dir}"
            
            # Verificação do vídeo
            if progress_callback:
                progress_callback(10, "Verificando vídeo...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Não foi possível abrir o vídeo"
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Remove resultados anteriores se existirem
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            # Processamento com acompanhamento manual do progresso
            if progress_callback:
                progress_callback(20, f"Iniciando processamento de {total_frames} frames...")
            
            results = []
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                result = self.model.track(
                    frame,
                    persist=True,
                    verbose=False
                )
                results.extend(result)
                
                # Atualiza progresso
                if progress_callback and total_frames > 0:
                    percent = 20 + int((frame_count / total_frames) * 70)
                    progress_callback(percent, f"Processando frame {frame_count}/{total_frames}")
            
            cap.release()
            
            if progress_callback:
                progress_callback(95, "Gerando relatório...")
            
            # Gera relatório
            report_content = self._generate_report(video_path, user_email, results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"report_{base_name}_{timestamp}.txt"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Atualiza cache
            self.cache[base_name] = {
                'hash': self.get_video_hash(video_path),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': user_email,
                'timestamp': timestamp
            }
            self.save_cache()
            
            if progress_callback:
                progress_callback(100, "Processamento concluído!")
            
            return True, f"Processamento {'re' if force_reprocess else ''}concluído! Resultados em: {output_dir}"
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Erro: {str(e)}")
            return False, f"Erro no processamento: {str(e)}"
    
    def _generate_report(self, video_path, user_email, results):
        """Gera o relatório de processamento"""
        if not results:
            return "Nenhum resultado para gerar relatório"
            
        report = f"""Relatório de Processamento - YOLOv8
Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Usuário: {user_email}
Vídeo processado: {os.path.basename(video_path)}

Estatísticas:
- Tempo por frame: {results[0].speed['inference']} ms
- Objetos detectados: {sum(len(r.boxes) for r in results)}

Detalhes das detecções:"""

        for i, result in enumerate(results):
            report += f"\n\nFrame {i+1}:"
            for box in result.boxes:
                report += f"\n- {box.cls[0]}: confiança {box.conf[0]:.2f}"
        
        return report

    def clear_processing_cache(self, video_name=None):
        """Limpa o cache de processamento"""
        if video_name:
            if video_name in self.cache:
                del self.cache[video_name]
        else:
            self.cache = {}
        self.save_cache()