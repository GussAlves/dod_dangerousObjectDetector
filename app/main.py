import sys
from PyQt5.QtWidgets import QApplication
from app.ui.main_window import MainWindow

def main():
    """Função principal que inicia a aplicação"""
    app = QApplication(sys.argv)
    
    # Configuração do estilo visual
    app.setStyle('Fusion')
    
    # Cria e exibe a janela principal
    window = MainWindow()
    window.show()
    
    # Executa o loop de eventos
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()