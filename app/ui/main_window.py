import os
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, 
    QDesktopWidget, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from app.core.processor import VideoProcessor

class ProcessingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, processor, video_path, email, force_reprocess=False):
        super().__init__()
        self.processor = processor
        self.video_path = video_path
        self.email = email
        self.force_reprocess = force_reprocess
    
    def run(self):
        try:
            # Estágio 1: Preparação (10%)
            self.progress_updated.emit(10, "Preparando para processamento...")
            
            # Estágio 2: Carregamento do vídeo (20%)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Não foi possível abrir o vídeo")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.progress_updated.emit(20, f"Vídeo carregado ({total_frames} frames)")
            
            # Estágio 3: Processamento (20-90%)
            success, message = self.processor.process_video(
                self.video_path, 
                self.email,
                self.force_reprocess,
                self.progress_updated.emit  # Passa o emitter diretamente
            )
            
            if not success:
                raise Exception(message)
            
            # Estágio 4: Finalização (90-100%)
            self.progress_updated.emit(95, "Gerando relatório...")
            self.progress_updated.emit(100, "Processamento concluído!")
            self.finished.emit(True, message)
            
        except Exception as e:
            self.progress_updated.emit(0, f"Erro: {str(e)}")
            self.finished.emit(False, str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Vídeos YOLOv8")
        self.setFixedSize(900, 550)
        
        # Centraliza a janela
        self.center_window()
        
        # Widgets
        self.email_label = QLabel("E-mail do usuário:")
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("seu@email.com")
        self.select_file_btn = QPushButton("Selecionar Vídeo")
        self.process_btn = QPushButton("Processar Vídeo")
        self.reprocess_btn = QPushButton("Reprocessar Vídeo")
        self.reprocess_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("background-color: #ffcccc;")
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        
        # Label de status
        self.status_label = QLabel("Pronto para processar vídeo")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        
        # Configuração da pré-visualização
        self.video_preview = QLabel()
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setFixedSize(640, 360)
        self.video_preview.setText("Pré-visualização aparecerá aqui")
        self.video_preview.setStyleSheet("""
            border: 2px solid #ccc;
            border-radius: 5px;
            background-color: #f0f0f0;
        """)
        
        # Layout
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        left_panel.addWidget(self.email_label)
        left_panel.addWidget(self.email_input)
        left_panel.addWidget(self.select_file_btn)
        left_panel.addWidget(self.process_btn)
        left_panel.addWidget(self.reprocess_btn)
        left_panel.addWidget(self.cancel_btn)
        left_panel.addWidget(self.progress_bar)
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addLayout(left_panel)
        main_layout.addWidget(self.video_preview)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Conexões
        self.select_file_btn.clicked.connect(self.select_file)
        self.process_btn.clicked.connect(self.process_video)
        self.reprocess_btn.clicked.connect(self.reprocess_video)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        
        # Variáveis
        self.selected_file = None
        self.processor = VideoProcessor()
        self.processing_thread = None
    
    def center_window(self):
        """Centraliza a janela na tela"""
        frame = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecionar Vídeo", 
            "", 
            "Vídeos (*.mp4 *.avi *.mov);;Todos os arquivos (*)"
        )
        if file_path:
            self.selected_file = file_path
            self.show_video_preview(file_path)
            self.reprocess_btn.setEnabled(True)
            self.update_status(f"Vídeo selecionado: {os.path.basename(file_path)}")
            QMessageBox.information(
                self, 
                "Sucesso", 
                f"Vídeo selecionado:\n{os.path.basename(file_path)}"
            )
    
    def show_video_preview(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_preview.setPixmap(QPixmap.fromImage(q_img).scaled(
                    640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
        except Exception as e:
            print(f"Erro na pré-visualização: {e}")
    
    def process_video(self, force_reprocess=False):
        email = self.email_input.text()
        if not email or "@" not in email:
            QMessageBox.warning(self, "Erro", "Por favor, insira um e-mail válido.")
            return
        
        if not self.selected_file:
            QMessageBox.warning(self, "Erro", "Por favor, selecione um vídeo primeiro.")
            return
        
        if force_reprocess:
            reply = QMessageBox.question(
                self, 'Confirmação',
                'Deseja reprocessar este vídeo? Isso substituirá os resultados anteriores.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        self.reset_progress()
        self.set_processing_ui(True)
        
        self.processing_thread = ProcessingThread(
            self.processor,
            self.selected_file,
            email,
            force_reprocess
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def reprocess_video(self):
        self.process_video(force_reprocess=True)
    
    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Confirmação',
                'Deseja realmente cancelar o processamento?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_thread.requestInterruption()
                self.update_status("Processamento cancelado", error=True)
                self.progress_bar.setValue(0)
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.update_status(message)
        
        # Efeito visual de carregamento para valores baixos
        if value < 30:
            self.progress_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #FFA500;
                }
            """)
        elif value < 70:
            self.progress_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #1E90FF;
                }
            """)
        else:
            self.progress_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
    
    def update_status(self, message, error=False):
        self.status_label.setText(message)
        if error:
            self.status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #666;")
    
    def reset_progress(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    def on_processing_finished(self, success, message):
        self.set_processing_ui(False)
        if success:
            self.update_status("Processamento concluído com sucesso!")
            QMessageBox.information(self, "Sucesso", message)
        else:
            self.update_status(f"Erro: {message}", error=True)
            QMessageBox.critical(self, "Erro", message)
    
    def set_processing_ui(self, processing):
        self.select_file_btn.setEnabled(not processing)
        self.process_btn.setEnabled(not processing)
        self.reprocess_btn.setEnabled(not processing and bool(self.selected_file))
        self.cancel_btn.setEnabled(processing)
        self.email_input.setEnabled(not processing)
    
    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Processamento em andamento',
                'Um vídeo está sendo processado. Deseja realmente sair?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_thread.requestInterruption()
                self.processing_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()