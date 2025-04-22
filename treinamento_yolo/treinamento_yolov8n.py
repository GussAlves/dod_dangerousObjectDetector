from ultralytics import YOLO
import torch

def main():
    # 1. Carregar modelo
    model = YOLO("../yolov8n.pt")  

    # 2. Treinamento
    model.train(
        data="dataset_yolov8/dataset.yaml",
        epochs=300, 
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',  # Usa GPU se disponível
        workers=1,  # Reduza para 1 se o erro persistir (ou 0 para desativar multiprocessamento)
        name="treino_customizado"
    )

    # 3. Validação
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Necessário para ambientes congelados (ex: PyInstaller)
    main()