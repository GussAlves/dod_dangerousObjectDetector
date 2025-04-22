# !pip install fiftyone

import fiftyone as fo
import fiftyone.zoo as foz
import os
from multiprocessing import freeze_support

def main():
    # Configurações
    CLASSES = ["knife", "scissors"]
    EXPORT_DIR = "./dataset_yolov8"  # Diretório principal
    TRAIN_SAMPLES = 1200  # Número de imagens de treino
    VAL_SAMPLES = 500    # Número de imagens de validação
    TEST_SAMPLES = 200   # Número de imagens de teste

    # Criar diretório se não existir
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 1. Baixar e exportar dados de TREINO
    print("Baixando dados de TREINO...")
    train_dataset = foz.load_zoo_dataset(
        'coco-2017',
        split='train',
        classes=CLASSES,
        max_samples=TRAIN_SAMPLES,
        shuffle=True,
        dataset_name="coco-train"
    )

    train_view = train_dataset.filter_labels(
        "ground_truth", 
        fo.ViewField("label").is_in(CLASSES)
    )

    train_view.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
        classes=CLASSES
    )

    # 2. Baixar e exportar dados de VALIDAÇÃO
    print("\nBaixando dados de VALIDAÇÃO...")
    val_dataset = foz.load_zoo_dataset(
        'voc-2012',
        split='validation',
        classes=CLASSES,
        max_samples=VAL_SAMPLES,
        shuffle=True,
        dataset_name="coco-val"
    )

    val_view = val_dataset.filter_labels(
        "ground_truth", 
        fo.ViewField("label").is_in(CLASSES)
    )

    val_view.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val",
        classes=CLASSES
    )

    # 3. Baixar e exportar dados de TESTE (opcional)
    print("\nBaixando dados de TESTE...")
    test_dataset = foz.load_zoo_dataset(
        'coco-2017',
        split='validation',
        classes=CLASSES,
        max_samples=TEST_SAMPLES,
        shuffle=True,
        dataset_name="coco-test"
    )

    test_view = test_dataset.filter_labels(
        "ground_truth", 
        fo.ViewField("label").is_in(CLASSES)
    )

    test_view.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="test",
        classes=CLASSES
    )

    # 4. Gerar arquivo data.yaml automaticamente
    print("\nGerando arquivo data.yaml...")
    yaml_content = f"""train: {os.path.join(EXPORT_DIR, 'train/images')}
val: {os.path.join(EXPORT_DIR, 'val/images')}
test: {os.path.join(EXPORT_DIR, 'test/images')}  # Opcional

nc: {len(CLASSES)}
names: {CLASSES}
"""

    with open(os.path.join(EXPORT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\nProcesso concluído! Dataset YOLOv8 criado em: {EXPORT_DIR}")
    print(f"Estrutura de pastas:")
    print(f"├── train/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"├── val/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"├── test/  # Opcional")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"└── data.yaml")

if _name_ == '_main_':
    freeze_support()
    main()