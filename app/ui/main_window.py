import os
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QLineEdit, QPushButton, QMessageBox, QDesktopWidget,
    QProgressBar, QComboBox, QGroupBox, QFileDialog  
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtRemoveInputHook  
from PyQt5.QtGui import QImage, QPixmap
from app.core.processor import VideoProcessor

pyqtRemoveInputHook()

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
            self.progress_updated.emit(10, "Preparando para processamento...")
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Não foi possível abrir o vídeo")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.progress_updated.emit(20, f"Vídeo carregado ({total_frames} frames)")
            
            success, message = self.processor.process_video(
                self.video_path, 
                self.email,
                self.force_reprocess,
                self.progress_updated.emit
            )
            
            if not success:
                raise Exception(message)
            
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
        self.setFixedSize(1000, 600)
        self.center_window()
        
        # Widgets
        self.email_label = QLabel("E-mail do usuário:")
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("seu@email.com")

        # Grupo para seleção de vídeo
        video_group = QGroupBox("Seleção de Vídeo")
        video_layout = QVBoxLayout()
        
        # Opção 1: Lista de vídeos locais
        self.local_video_combobox = QComboBox()
        self.refresh_videos_btn = QPushButton("Atualizar lista local")
        
        # Opção 2: Selecionar arquivo externo
        self.select_file_btn = QPushButton("Selecionar Vídeo Externo")
        
        video_layout.addWidget(QLabel("Vídeos locais (data/input):"))
        video_layout.addWidget(self.local_video_combobox)
        video_layout.addWidget(self.refresh_videos_btn)
        video_layout.addWidget(QLabel("Ou:"))
        video_layout.addWidget(self.select_file_btn)
        video_group.setLayout(video_layout)

        # Botões de ação
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
        
        # Status
        self.status_label = QLabel("Pronto para processar vídeo")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Pré-visualização
        self.video_preview = QLabel()
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setFixedSize(640, 360)
        self.video_preview.setText("Pré-visualização aparecerá aqui")
        self.video_preview.setStyleSheet("border: 2px solid #ccc; border-radius: 5px;")
        
        # Layout
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        left_panel.addWidget(self.email_label)
        left_panel.addWidget(self.email_input)
        left_panel.addWidget(video_group)
        left_panel.addWidget(self.process_btn)
        left_panel.addWidget(self.reprocess_btn)
        left_panel.addWidget(self.cancel_btn)
        left_panel.addWidget(self.progress_bar)
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addLayout(left_panel, 40)
        main_layout.addWidget(self.video_preview, 60)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Conexões
        self.refresh_videos_btn.clicked.connect(self.refresh_video_list)
        self.local_video_combobox.currentIndexChanged.connect(self.on_local_video_selected)
        self.select_file_btn.clicked.connect(self.select_external_file)
        self.process_btn.clicked.connect(self.process_video)
        self.reprocess_btn.clicked.connect(self.reprocess_video)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        
        # Variáveis
        self.selected_file = None
        self.processor = VideoProcessor()
        self.processing_thread = None
        
        # Inicialização
        self.refresh_video_list()
        self.setup_styles()

    def setup_styles(self):
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)

    def center_window(self):
        frame = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

    def refresh_video_list(self):
        """Atualiza a lista de vídeos locais"""
        self.local_video_combobox.clear()
        
        input_dir = "data/input"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        self.local_video_combobox.addItem("-- Selecione um vídeo local --")
        if video_files:
            self.local_video_combobox.addItems(sorted(video_files))
            self.update_status(f"{len(video_files)} vídeo(s) local(is) encontrado(s)")
        else:
            self.local_video_combobox.addItem("Nenhum vídeo local encontrado")
            self.update_status("Adicione vídeos na pasta data/input")

    def on_local_video_selected(self, index):
        """Quando um vídeo local é selecionado"""
        if index > 0:
            video_file = self.local_video_combobox.currentText()
            self.selected_file = os.path.join("data/input", video_file)
            self.show_video_preview(self.selected_file)
            self.update_status(f"Vídeo local selecionado: {video_file}")
            self.enable_processing_buttons(True)

    def select_external_file(self):
        """Seleciona um vídeo externo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo", "", 
            "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos os arquivos (*)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.show_video_preview(file_path)
            self.update_status(f"Vídeo externo selecionado: {os.path.basename(file_path)}")
            self.enable_processing_buttons(True)
            # Reseta a seleção local
            self.local_video_combobox.setCurrentIndex(0)

    def enable_processing_buttons(self, enable):
        """Ativa/desativa botões de processamento"""
        self.process_btn.setEnabled(enable)
        self.reprocess_btn.setEnabled(enable)

    def show_video_preview(self, video_path):
        """Exibe o primeiro frame do vídeo como pré-visualização"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
                self.video_preview.setPixmap(
                    QPixmap.fromImage(q_img).scaled(
                        640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
        except Exception as e:
            print(f"Erro na pré-visualização: {e}")
            self.video_preview.setText("Não foi possível carregar a pré-visualização")

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