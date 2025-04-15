from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QMessageBox
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Vídeos YOLOv8")
        self.setGeometry(100, 100, 400, 200)
        
        # Widgets
        self.email_label = QLabel("E-mail do usuário:")
        self.email_input = QLineEdit()
        self.select_file_btn = QPushButton("Selecionar Vídeo")
        self.process_btn = QPushButton("Processar Vídeo")
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.email_label)
        layout.addWidget(self.email_input)
        layout.addWidget(self.select_file_btn)
        layout.addWidget(self.process_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Conexões
        self.select_file_btn.clicked.connect(self.select_file)
        self.process_btn.clicked.connect(self.process_video)
        
        # Variáveis
        self.selected_file = None
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.selected_file = file_path
            QMessageBox.information(self, "Sucesso", f"Vídeo selecionado: {file_path}")
    
    def process_video(self):
        email = self.email_input.text()
        if not email or "@" not in email:
            QMessageBox.warning(self, "Erro", "Por favor, insira um e-mail válido.")
            return
        
        if not self.selected_file:
            QMessageBox.warning(self, "Erro", "Por favor, selecione um vídeo primeiro.")
            return
        
        # Aqui você chamaria o processador YOLOv8
        QMessageBox.information(self, "Processando", "O vídeo está sendo processado...")
        # TODO: Implementar o processamento real