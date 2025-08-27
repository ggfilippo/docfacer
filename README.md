# Docfacer

Docfacer é uma aplicação web em Flask para cadastro e autenticação de usuários
por meio de reconhecimento de documentos e validação facial. O sistema utiliza
OCR e redes YOLO para extrair dados de documentos, verificar a correspondência
com as informações cadastradas e realizar prova de vida através da câmera.

## Pré-requisitos

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- CUDA e drivers para GPU (opcional, mas recomendado para desempenho)
- Conta no [Firebase](https://firebase.google.com/) com Realtime Database e
  Storage configurados
- Modelos YOLO para OCR e prova de vida disponíveis em `../models/ocr.pt` e
  `../models/large_v2_300.pt`

## Bibliotecas principais

As dependências utilizadas no projeto incluem:

- `flask`
- `python-dotenv`
- `pyrebase4`
- `ultralytics`
- `opencv-python`
- `face_recognition`
- `Pillow`
- `requests`
- `easyocr`
- `numpy`
- `rapidfuzz`

## Instalação

```bash
# 1. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate    # Windows

# 2. Instale as dependências
pip install flask python-dotenv pyrebase4 ultralytics opencv-python face_recognition Pillow requests easyocr numpy rapidfuzz

# 3. Defina as variáveis de ambiente do Firebase em um arquivo `.env`
cat <<EOT >> .env
API_KEY="sua_chave"
MSG_SENDER_ID="seu_sender_id"
APP_ID="seu_app_id"
EOT

# 4. Execute a aplicação
python app.py
```

## Como funciona

1. **Cadastro de CPF e informações** – O usuário informa CPF, e-mail, nome e
   data de nascimento, que são salvos no Firebase.
2. **Envio de documento** – Uma foto do documento é enviada. Um modelo YOLO
   detecta regiões relevantes (CPF, nome, etc.) e o `easyocr` extrai o texto
   para validação com os dados cadastrados.
3. **Reconhecimento facial e prova de vida** – O sistema usa `face_recognition`
   para comparar o rosto no documento com o rosto do usuário em tempo real.
   Outro modelo YOLO verifica se há prova de vida.

## Estrutura

```
app.py          # aplicação Flask
static/         # arquivos estáticos
templates/      # páginas HTML
```

## Status das verificações
- Integração com Firebase para autenticação e armazenamento de dados
- OCR com YOLO + easyocr
- Reconhecimento facial e prova de vida

