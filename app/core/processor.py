from ultralytics import YOLO
import os
from datetime import datetime
import shutil
import cv2
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from collections import defaultdict
import pandas as pd
from PyQt5.QtWidgets import QMessageBox

class VideoProcessor:
    def __init__(self, input_dir="data/input", processed_dir="data/processed", reports_dir="data/reports"):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.frames_dir = os.path.join(processed_dir, "frames")
        self.model = None
        self.cache_file = os.path.join(processed_dir, "processing_cache.json")
        
        # Configurações de e-mail (remetente fixo)
        self.email_config = {
            'sender': 'astolfotheduck@gmail.com',
            'password': 'xekhhaakldgqbudh',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587
        }
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self._load_model()
        self.cache = self._load_cache()

    def _load_model(self):
        """Carrega o modelo YOLO personalizado (knife como classe 0)"""
        try:
            self.model = YOLO('treinamento_yolo/runs/detect/treino_customizado7/weights/best.pt')
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
        """Gera um hash único para o vídeo"""
        try:
            stat = os.stat(video_path)
            return f"{stat.st_size}_{stat.st_mtime}"
        except Exception:
            return None

    def is_video_processed(self, video_path, base_name):
        """Verifica se o vídeo já foi processado"""
        video_hash = self.get_video_hash(video_path)
        if not video_hash:
            return False
        
        if base_name in self.cache:
            cached_data = self.cache[base_name]
            if isinstance(cached_data, dict):
                cached_hash = cached_data.get('hash')
                output_dir = os.path.join(self.frames_dir, base_name)
                if cached_hash == video_hash and os.path.exists(output_dir):
                    return True
        return False

    def extract_frames(self, video_path, output_dir, progress_callback=None):
        """Extrai frames do vídeo"""
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
                    percent = int((frame_count / total_frames) * 50)
                    progress_callback(percent, f"Extraindo frame {frame_count}/{total_frames}")
            
            cap.release()
            return True, f"{frame_count} frames extraídos com sucesso"
        except Exception as e:
            return False, f"Erro ao extrair frames: {str(e)}"

    def process_frames(self, frames_dir, progress_callback=None):
        """Processa frames focando apenas em facas (classe 0)"""
        try:
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            total_frames = len(frame_files)
            
            # Estruturas para análise de facas
            knife_stats = {
                'total_detections': 0,
                'frames_with_knives': 0,
                'detections_per_frame': defaultdict(int),
                'confidence_sum': 0.0,
                'tracked_knives': defaultdict(list)
            }
            
            results = []
            
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    continue
                
                # Processa apenas a classe 0 (knife)
                result = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    classes=[0]  # Foca apenas em facas
                )
                
                # Analisa detecções de facas
                for r in result:
                    if r.boxes is not None:
                        for box, conf in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu()):
                            knife_stats['total_detections'] += 1
                            knife_stats['detections_per_frame'][i] += 1
                            knife_stats['confidence_sum'] += float(conf)
                            
                            # Rastreamento de facas individuais
                            if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                                track_id = int(r.boxes.id[0])
                                knife_stats['tracked_knives'][track_id].append({
                                    'frame': i,
                                    'bbox': box.tolist(),
                                    'confidence': float(conf)
                                })
                
                if knife_stats['detections_per_frame'][i] > 0:
                    knife_stats['frames_with_knives'] += 1
                
                results.extend(result)
                
                if progress_callback and total_frames > 0:
                    percent = 50 + int((i / total_frames) * 45)
                    progress_callback(percent, f"Processando frame {i+1}/{total_frames}")
            
            return True, (results, knife_stats)
        except Exception as e:
            return False, f"Erro ao processar frames: {str(e)}"

    def _generate_report(self, video_path, user_email, processing_data):
        """Gera relatório detalhado de facas detectadas"""
        results, knife_stats = processing_data
        
        if not results:
            return "Nenhuma faca detectada no vídeo", None
            
        # Cálculo de estatísticas
        total_frames = len(knife_stats['detections_per_frame'])
        detection_rate = (knife_stats['frames_with_knives'] / total_frames) * 100 if total_frames > 0 else 0
        avg_confidence = knife_stats['confidence_sum'] / knife_stats['total_detections'] if knife_stats['total_detections'] > 0 else 0
        
        # Gera relatório em markdown
        report = f"""# Relatório de Detecção de Facas

## Metadados
- **Vídeo analisado:** {os.path.basename(video_path)}
- **Data/Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Destinatário:** {user_email}
- **Total de frames:** {total_frames}

## Estatísticas de Detecção
- **Total de detecções:** {knife_stats['total_detections']}
- **Frames com facas:** {knife_stats['frames_with_knives']} ({detection_rate:.2f}%)
- **Confiança média:** {avg_confidence:.2f}
- **Facas distintas detectadas:** {len(knife_stats['tracked_knives'])}

## Detalhes por Faca (Rastreamento)"""
        
        # Adiciona informações de cada faca rastreada
        for track_id, detections in knife_stats['tracked_knives'].items():
            first_frame = detections[0]['frame']
            last_frame = detections[-1]['frame']
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            
            report += f"\n\n### Faca ID {track_id}"
            report += f"\n- **Frames:** {first_frame} a {last_frame}"
            report += f"\n- **Ocorrências:** {len(detections)}"
            report += f"\n- **Confiança média:** {avg_conf:.2f}"
        
        # Gera CSV com dados detalhados
        csv_data = []
        for track_id, detections in knife_stats['tracked_knives'].items():
            for d in detections:
                csv_data.append({
                    'track_id': track_id,
                    'frame': d['frame'],
                    'confidence': d['confidence'],
                    'x1': d['bbox'][0],
                    'y1': d['bbox'][1],
                    'x2': d['bbox'][2],
                    'y2': d['bbox'][3]
                })
        
        csv_path = None
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_filename = f"knife_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = os.path.join(self.reports_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            report += f"\n\n## Dados Completos\nDetalhes de todas as detecções salvos em: `{csv_filename}`"
        
        return report, csv_path

    def _send_email_report(self, recipient_email, report_content, attachment_path=None):
        """Envia relatório por e-mail"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = recipient_email
            msg['Subject'] = "Relatório de Detecção de Facas"
            
            # Corpo do e-mail
            msg.attach(MIMEText(report_content, 'plain'))
            
            # Anexa arquivo CSV se existir
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)
            
            # Envia e-mail
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender'], self.email_config['password'])
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Erro ao enviar e-mail: {str(e)}")
            return False

    def process_video(self, video_path, user_email, force_reprocess=False, progress_callback=None):
        """Processa vídeo e gera relatório de facas"""
        try:
            if not self.model:
                return False, "Modelo não carregado"
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            frames_output_dir = os.path.join(self.frames_dir, base_name)
            
            # Verifica cache
            if not force_reprocess and self.is_video_processed(video_path, base_name):
                if progress_callback:
                    progress_callback(100, "Usando cache existente")
                return True, "Vídeo já processado (use 'Reprocessar' para nova análise)"
            
            # Remove processamento anterior se existir
            if os.path.exists(frames_output_dir):
                shutil.rmtree(frames_output_dir)
            
            # Extrai frames
            if progress_callback:
                progress_callback(10, "Extraindo frames...")
            
            success, message = self.extract_frames(video_path, frames_output_dir, progress_callback)
            if not success:
                return False, message
            
            # Processa frames
            if progress_callback:
                progress_callback(50, "Analisando facas...")
            
            success, processing_data = self.process_frames(frames_output_dir, progress_callback)
            if not success:
                return False, processing_data
            
            # Gera relatório
            if progress_callback:
                progress_callback(90, "Gerando relatório...")
            
            report_content, csv_path = self._generate_report(video_path, user_email, processing_data)
            
            # Salva relatório
            report_filename = f"knife_report_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Envia e-mail
            if progress_callback:
                progress_callback(95, "Enviando relatório...")
            
            email_sent = self._send_email_report(user_email, report_content, csv_path)
            email_status = "Relatório enviado por e-mail" if email_sent else "Falha ao enviar e-mail"
            
            # Atualiza cache
            self.cache[base_name] = {
                'hash': self.get_video_hash(video_path),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': user_email,
                'report_path': report_path,
                'csv_path': csv_path
            }
            self.save_cache()
            
            if progress_callback:
                progress_callback(100, "Processamento completo!")
            
            return True, f"Análise concluída! {email_status}"
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Erro: {str(e)}")
            return False, f"Erro no processamento: {str(e)}"

    def clear_processing_cache(self, video_name=None):
        """Limpa o cache de processamento"""
        if video_name:
            if video_name in self.cache:
                frames_dir = os.path.join(self.frames_dir, video_name)
                if os.path.exists(frames_dir):
                    shutil.rmtree(frames_dir)
                del self.cache[video_name]
        else:
            for video_data in self.cache.values():
                if isinstance(video_data, dict) and 'frames_dir' in video_data:
                    frames_dir = video_data['frames_dir']
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
            self.cache = {}
        self.save_cache()