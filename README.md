# DOD - Dangerous Object Detector

## Como executar

Crie um ambiente virtual Python:  ``` python -m venv venv ```

Ative o ambiente virtual:

```
Windows: venv\Scripts\activate

Linux/Mac: source venv/bin/activate
```

Instale as dependências: ``` pip install -r requirements.txt ```

Execute o aplicativo na pasta raiz: ``` python -m app.main  ```


### Estrutura do projeto
```
/meu_projeto
│
├── /app
│   ├── main.py              # Ponto de entrada principal
│   ├── /ui                  # Arquivos de interface
│   │   └── main_window.py
│   └── /core
│       ├── processor.py     # Lógica de processamento
│       └── utils.py         # Funções auxiliares
│
├── /data
│   ├── /input               # Vídeos a serem processados
│   ├── /processed           # Vídeos processados (framizados)
│   └── /reports             # Relatórios gerados
│
├── requirements.txt         # Dependências do projeto
└── README.md
```