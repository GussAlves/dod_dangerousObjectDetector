import os
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QDesktopWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from app.core.processor import VideoProcessor

class ProcessingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int)
    
    def __init__(self, processor, video_path, email, force_reprocess=False):
        super().__init__()
        self.processor = processor
        self.video_path = video_path
        self.email = email
        self.force_reprocess = force_reprocess
    
    def run(self):
        self.progress.emit(20)
        try:
            success, message = self.processor.process_video(
                self.video_path, 
                self.email,
                self.force_reprocess
            )
            self.progress.emit(100)
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Vídeos YOLOv8")
        self.setFixedSize(900, 500)  # Tamanho fixo para evitar redimensionamento
        
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
        
        self.set_buttons_enabled(False)
        self.processing_thread = ProcessingThread(
            self.processor,
            self.selected_file,
            email,
            force_reprocess
        )
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def reprocess_video(self):
        self.process_video(force_reprocess=True)
    
    def on_processing_finished(self, success, message):
        self.set_buttons_enabled(True)
        if success:
            QMessageBox.information(self, "Sucesso", message)
        else:
            QMessageBox.critical(self, "Erro", message)
    
    def set_buttons_enabled(self, enabled):
        self.select_file_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.reprocess_btn.setEnabled(enabled and bool(self.selected_file))