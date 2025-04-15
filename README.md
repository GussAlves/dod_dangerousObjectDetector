# DOD - Dangerous Object Detector

## Como executar

```
# 1. Ativar venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o programa (a partir da raiz!)
python -m app.main
```

### Estrutura do projeto
```
/meu_projeto
├── /app
│   ├── __init__.py
│   ├── main.py
│   ├── /ui
│   │   ├── __init__.py
│   │   └── main_window.py
│   └── /core
│       ├── __init__.py
│       ├── processor.py
│       └── utils.py
├── /data
│   ├── /input          # (Pasta para vídeos não processados)
│   ├── /processed      # (Pasta para vídeos processados)
│   └── /reports        # (Pasta para relatórios)
├── requirements.txt
└── README.md
```