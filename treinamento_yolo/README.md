# Treinamento utilizando YoloV8n 

Para este treinamento foi realizado a extração de imagens e labels utilizando o `fiftyone` junto com a biblioteca `coco-2017`

## Visão Geral
Este projeto consiste em dois scripts principais:

1. download_dataset.py - Baixa e prepara um dataset no formato YOLOv8
2.train_model.py - Realiza o treinamento de um modelo YOLOv8 customizado

## Pré-requisitos
- Python 3.8 ou superior
- GPU NVIDIA (recomendado) com drivers CUDA instalados, pode ser usado somente a CPU 
- Bibliotecas Python listadas em requirements.txt

## Uso
1. Download e Preparação do Dataset
Execute o script para baixar e preparar o dataset:

```
python download_dataset.py
```

Parâmetros Configuráveis (no script):
- CLASSES: Lista de classes para detecção (padrão: ["knife", "scissors"])
- EXPORT_DIR: Diretório de saída (padrão: "./dataset_yolov8")
- TRAIN_SAMPLES: Número de amostras de treino (padrão: 1200)
- VAL_SAMPLES: Número de amostras de validação (padrão: 500)
- TEST_SAMPLES: Número de amostras de teste (padrão: 200)

2. Treinamento do Modelo
Execute o script de treinamento:

```
python train_model.py
```

## Estrutura 

```
├── dataset_yolov8/          # Dataset gerado
├── runs/                    # Resultados do treinamento
├── download_dataset.py      # Script de download do dataset
├── train_model.py           # Script de treinamento
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```
