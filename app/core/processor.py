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
        self.frames_dir = os.path.join(processed_dir, "frames")  # Nova pasta para frames
        self.model = None
        self.cache_file = os.path.join(processed_dir, "processing_cache.json")
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)  # Cria a pasta de frames
        
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
                output_dir = os.path.join(self.frames_dir, base_name)  # Agora verifica na pasta de frames
                if cached_hash == video_hash and os.path.exists(output_dir):
                    return True
        return False

    def extract_frames(self, video_path, output_dir, progress_callback=None):
        """Extrai frames do vídeo e salva em uma pasta"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Não foi possível abrir o vídeo"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                
                if progress_callback and total_frames > 0:
                    percent = int((frame_count / total_frames) * 50)  # Extração representa 50% do progresso
                    progress_callback(percent, f"Extraindo frame {frame_count}/{total_frames}")
            
            cap.release()
            return True, f"{frame_count} frames extraídos com sucesso"
        except Exception as e:
            return False, f"Erro ao extrair frames: {str(e)}"

    def process_frames(self, frames_dir, progress_callback=None):
        """Processa cada frame individualmente usando YOLOv8"""
        try:
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            total_frames = len(frame_files)
            results = []
            
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    continue
                
                result = self.model.track(
                    frame,
                    persist=True,
                    verbose=False
                )
                results.extend(result)
                
                if progress_callback and total_frames > 0:
                    percent = 50 + int((i / total_frames) * 45)  # Processamento representa 45% do progresso
                    progress_callback(percent, f"Processando frame {i+1}/{total_frames}")
            
            return True, results
        except Exception as e:
            return False, f"Erro ao processar frames: {str(e)}"

    def process_video(self, video_path, user_email, force_reprocess=False, progress_callback=None):
        """
        Processa um vídeo convertendo para frames primeiro e depois processando cada frame
        
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
            frames_output_dir = os.path.join(self.frames_dir, base_name)  # Pasta específica para os frames deste vídeo
            
            # Verifica se deve reprocessar
            if not force_reprocess and self.is_video_processed(video_path, base_name):
                if progress_callback:
                    progress_callback(100, "Vídeo já processado - usando cache")
                return True, f"Vídeo já processado anteriormente. Frames em: {frames_output_dir}"
            
            # Verificação do vídeo
            if progress_callback:
                progress_callback(5, "Verificando vídeo...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Não foi possível abrir o vídeo"
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Remove resultados anteriores se existirem
            if os.path.exists(frames_output_dir):
                shutil.rmtree(frames_output_dir)
            
            # Extrai os frames do vídeo
            if progress_callback:
                progress_callback(10, "Extraindo frames do vídeo...")
            
            success, message = self.extract_frames(video_path, frames_output_dir, progress_callback)
            if not success:
                return False, message
            
            # Processa os frames extraídos
            if progress_callback:
                progress_callback(50, "Processando frames extraídos...")
            
            success, results = self.process_frames(frames_output_dir, progress_callback)
            if not success:
                return False, results  # Neste caso, 'results' contém a mensagem de erro
            
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
                'timestamp': timestamp,
                'frames_dir': frames_output_dir  # Armazena o caminho dos frames
            }
            self.save_cache()
            
            if progress_callback:
                progress_callback(100, "Processamento concluído!")
            
            return True, f"Processamento {'re' if force_reprocess else ''}concluído! Frames processados em: {frames_output_dir}"
            
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
                # Remove a pasta de frames correspondente
                frames_dir = self.cache[video_name].get('frames_dir')
                if frames_dir and os.path.exists(frames_dir):
                    shutil.rmtree(frames_dir)
                del self.cache[video_name]
        else:
            # Remove todas as pastas de frames
            for video_data in self.cache.values():
                if isinstance(video_data, dict) and 'frames_dir' in video_data:
                    frames_dir = video_data['frames_dir']
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
            self.cache = {}
        self.save_cache()