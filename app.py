from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from dotenv import load_dotenv
import pyrebase
import time
import os
from ultralytics import YOLO
import cv2
import face_recognition
from face_recognition import face_encodings, compare_faces, face_locations
import io
from io import BytesIO
from PIL import Image
import requests
import re
import easyocr
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime

cpf_global = None

app = Flask(__name__)
app.config.update(SECRET_KEY=os.urandom(24))

load_dotenv()

# -------------------------- FIREBASE --------------------------
firebase_config = {
    "apiKey": os.getenv("API_KEY"),
    "authDomain": "livenessSystem.firebaseapp.com",
    "databaseURL": "https://livenesssystem-ddf35-default-rtdb.firebaseio.com/",
    "projectId": "livenessSystem",
    "storageBucket": "livenesssystem-ddf35.appspot.com",
    "messagingSenderId": os.getenv("MSG_SENDER_ID"),
    "appId": os.getenv("APP_ID")
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()
# -----------------------------------------------------------------

# ----------------------------- ROTAS -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cadastro_cpf', methods=['GET', 'POST'])
def cadastro_cpf():
    if request.method == 'POST':
        cpf = request.form['cpf']
        session['cpf'] = cpf  
        return redirect('/cadastro_info')
    return render_template('cadastro_cpf.html')

@app.route('/cadastro_info', methods=['GET', 'POST'])
def cadastro_info():
    if request.method == 'POST':
        email = request.form['email']
        nome = request.form['nome']
        data_nascimento = request.form['data_nascimento']
        senha = request.form['senha'] 

        cpf = session.get('cpf')
        if not cpf:
            return "Erro: CPF não encontrado na sessão", 400

        cpf = re.sub(r'[^\w]', '', cpf)

        try:
            auth.create_user_with_email_and_password(email, senha)
            user_data = {
                "nome": nome,
                "data_nascimento": data_nascimento,
                "email": email,
                "cpf": cpf
            }
            db.child("usuarios").child(cpf).set(user_data)
            return redirect('/cadastro_documento')
        except Exception as e:
            print(f"Erro ao criar usuário: {e}")
            return "Erro ao criar usuário", 400

    return render_template('cadastro_info.html')

############################################################################################################
# Mapeamento de classes YOLO para categorias genéricas
CLASS_MAPPING = {
    0: 'cpf',         # cnh-cpf
    1: 'data_nascimento',  # cnh-dtaNasc
    2: 'nome',        # cnh-nome
    3: 'rg',          # cnh-rg
    4: 'cpf',         # rg-cpf
    5: 'data_nascimento',  # rg-dtaNasc
    6: 'nome',        # rg-nome
    7: 'rg'           # rg-numero
}

def sanitize_cpf(cpf):
    return re.sub(r'[^\w]', '', cpf)

def normalize_date(date_str):
    """
    Normaliza uma data no formato 'DD/MM/YYYY' ou 'YYYY-MM-DD' para 'YYYY-MM-DD'.
    """
    try:
        # Detectar formato e converter para ISO (YYYY-MM-DD)
        if '/' in date_str:  # Formato 'DD/MM/YYYY'
            return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
        elif '-' in date_str:  # Formato 'YYYY-MM-DD'
            return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        else:
            return date_str  # Retorna sem alteração se não for reconhecido
    except ValueError:
        print(f"Formato de data inválido: {date_str}")
        return date_str  # Retorna a string original em caso de erro

def save_to_firebase(storage, path, data):
    try:
        storage.child(path).put(data)
        return True
    except Exception as e:
        print(f"Erro ao salvar no Firebase: {e}")
        return False

def detect_face(image):
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return None
    top, right, bottom, left = face_locations[0]
    return image[top:bottom, left:right]

def perform_ocr(image, model, reader, output_path="diagnostico_yolo.jpg"):
    extracted_data = {'cpf': '', 'data_nascimento': '', 'nome': '', 'rg': ''}
    results = model.predict(source=image, imgsz=640, conf=0.5)

    # Diagnóstico do YOLO - Adicionar anotações à imagem
    annotated_image = image.copy()
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        cls = int(result.cls[0].cpu().numpy())
        x1, y1, x2, y2 = map(int, box)

        # Desenhar a caixa delimitadora
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_image, 
            CLASS_MAPPING.get(cls, 'Desconhecida'), 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )

    # Processar as regiões de interesse (ROI) detectadas pelo YOLO
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        cls = int(result.cls[0].cpu().numpy())
        category = CLASS_MAPPING.get(cls, None)

        if not category:
            continue 

        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]

        # Executar OCR na região de interesse
        ocr_results = reader.readtext(roi)
        for (_, text, _) in ocr_results:
            clean_text = text.strip()
            if category in extracted_data and clean_text not in extracted_data[category]:
                if category == 'data_nascimento':
                    clean_text = normalize_date(clean_text)
                extracted_data[category] = clean_text

    # Validar e formatar os dados extraídos
    extracted_data['cpf'] = re.sub(r'[^\d]', '', extracted_data.get('cpf', '')) 
    extracted_data['data_nascimento'] = re.sub(r'[^\d-]', '', extracted_data.get('data_nascimento', ''))
    extracted_data['nome'] = extracted_data.get('nome', '').title() 
    extracted_data['rg'] = re.sub(r'[^\w]', '', extracted_data.get('rg', ''))

    return extracted_data

def cross_validate_data(extracted_data, user_data):
    """
    Valida combinações de dados entre o extraído e o cadastrado.
    """
    # Comparação estrita do CPF
    if extracted_data.get('cpf') != user_data.get('cpf'):
        return "Erro: O CPF do documento não corresponde ao cadastrado."

    # Comparação da data de nascimento (já normalizada)
    if normalize_date(extracted_data.get('data_nascimento', '')) != normalize_date(user_data.get('data_nascimento', '')):
        return "Erro: A data de nascimento do documento não corresponde ao cadastrado."

    # Se tudo estiver correto
    return None


def calculate_similarity(extracted_data, user_data):
    """
    Calcula a similaridade entre os dados extraídos e os dados do usuário no banco.
    Exibe os valores comparados para depuração.
    """
    similarities = {}
    for key in ['cpf', 'data_nascimento', 'nome']:
        extracted_value = extracted_data.get(key, '').strip().lower()
        user_value = user_data.get(key, '').strip().lower()

        # Normalizar o formato de datas antes da comparação
        if key == 'data_nascimento':
            extracted_value = normalize_date(extracted_value)
            user_value = normalize_date(user_value)

        print(f"Comparando {key}: Extraído = '{extracted_value}' | Armazenado = '{user_value}'")
        similarities[key] = fuzz.ratio(extracted_value, user_value)

    # Calcula a média das similaridades
    overall_similarity = sum(similarities.values()) / len(similarities)
    print(f"Similaridade geral: {overall_similarity}")
    print(f"Similaridades individuais: {similarities}")
    return overall_similarity, similarities


@app.route('/cadastro_documento', methods=['GET', 'POST'])
def cadastro_documento():
    if request.method == 'POST':
        cpf = session.get('cpf')
        if not cpf:
            return render_template('cadastro_documento.html', validation_result="Erro: CPF não encontrado na sessão",
                                   validation_status="error", extracted_data=None)

        cpf = sanitize_cpf(cpf)

        # Verificar se um arquivo foi enviado
        if 'documento' not in request.files or request.files['documento'].filename == '':
            return render_template('cadastro_documento.html', validation_result="Nenhum arquivo selecionado",
                                   validation_status="error", extracted_data=None)

        documento = request.files['documento']
        documento_bytes = documento.read()
        image = cv2.imdecode(np.frombuffer(documento_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Salvar documento no Firebase
        original_doc_path = f"documentos/{cpf}.jpg"
        if not save_to_firebase(storage, original_doc_path, io.BytesIO(documento_bytes)):
            return render_template('cadastro_documento.html', validation_result="Erro ao salvar documento",
                                   validation_status="error", extracted_data=None)

        # Detectar rosto
        face_image = detect_face(image)
        if face_image is None:
            return render_template('cadastro_documento.html', validation_result="Nenhum rosto encontrado no documento",
                                   validation_status="error", extracted_data=None)

        # Salvar rosto no Firebase
        buffer = io.BytesIO()
        Image.fromarray(face_image).save(buffer, format="JPEG")
        buffer.seek(0)
        face_path = f"rostos/{cpf}.jpg"
        if not save_to_firebase(storage, face_path, buffer):
            return render_template('cadastro_documento.html', validation_result="Erro ao salvar rosto",
                                   validation_status="error", extracted_data=None)

        # Executar OCR
        model = YOLO("../models/ocr.pt")
        reader = easyocr.Reader(['pt'], gpu=True)
        extracted_data = perform_ocr(image, model, reader)

        print(f"Dados extraídos do documento: {extracted_data}")

        # Recuperar dados do usuário a partir do CPF
        user_data = db.child("usuarios").child(cpf).get().val()
        if not user_data:
            return render_template('cadastro_documento.html', validation_result="Usuário não encontrado",
                                   validation_status="error", extracted_data=None)

        print(f"Dados do usuário armazenados no banco: {user_data}")

        # Validação cruzada de dados
        validation_error = cross_validate_data(extracted_data, user_data)

        # Registrar o status da validação no Firebase
        ocr_status = {
            "success": validation_error is None,
            "details": validation_error or "Validação OCR bem-sucedida",
            "similarities": calculate_similarity(extracted_data, user_data)[1]
        }

        try:
            db.child("usuarios").child(cpf).child("ocr_status").set(ocr_status)
            print(f"OCR status salvo com sucesso no Firebase para CPF: {cpf}")
        except Exception as e:
            print(f"Erro ao salvar OCR status no Firebase: {e}")


        # Preparar mensagem para o usuário
        if validation_error:
            message = f"{validation_error}\n Você ainda pode continuar para a próxima etapa."
            status = "warning"
        else:
            message = "Validação OCR bem-sucedida! Continue para a próxima etapa."
            status = "success"

        return render_template(
            'cadastro_documento.html',
            validation_result=message,
            validation_status=status,
            extracted_data=extracted_data
        )

    return render_template('cadastro_documento.html', validation_result=None, validation_status=None, extracted_data=None)






############################################################################################################
    
# ----------------------------- ROTA: RECONHECIMENTO FACIAL -----------------------------
@app.route('/reconhecimento_facial')
def reconhecimento_facial():
    # Carregar o CPF da sessão (se necessário)
    global cpf_global
    cpf_global = session.get('cpf')
    if not cpf_global:
        return "Erro: CPF não encontrado na sessão", 400

    return render_template('cadastro_facial.html')

# ----------------------------- ROTA: FEED DE VÍDEO -----------------------------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/progress')
def get_progress():
    global progress, face_check_done, liveness_check_done

    status = "Iniciando reconhecimento facial..."
    if progress == 25:
        status = "Posicione seu rosto corretamente."
    elif progress == 50:
        status = "Face reconhecida. Realizando prova de vida..."
    elif progress == 100:
        status = "Autenticação Bem-Sucedida!"
    elif not face_check_done:
        status = "Erro: rosto não detectado."
    elif not liveness_check_done:
        status = "Erro: prova de vida falhou."

    return jsonify({'progress': progress, 'status': status})


def download_image_as_bytes(face_path):
    try:
        # Obter o URL de download temporário do arquivo no Firebase Storage
        download_url = storage.child(face_path).get_url(None)

        # Fazer download do conteúdo do arquivo como bytes usando requests
        response = requests.get(download_url)
        if response.status_code == 200:
            return response.content  # Retorna os bytes da imagem
        else:
            raise FileNotFoundError(f"Arquivo não encontrado no caminho: {face_path}")
    except Exception as e:
        print(f"Erro ao fazer download do arquivo: {e}")
        return None

# Variáveis globais
progress = 0
face_check_done = False
liveness_check_done = False
face_matched = False
liveness_verified = False

def face_match(frame, documento_face_encoding):
    global face_matched, face_check_done
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    face_matched = False

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    if face_encodings:
        matches = face_recognition.compare_faces([documento_face_encoding], face_encodings[0])
        face_matched = True in matches
        face_check_done = True
    else:
        face_check_done = False

    return face_matched

def yolo_liveness(frame, model):
    global liveness_verified, liveness_check_done
    results = model(frame, stream=True, verbose=False)
    liveness_verified = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            # classe 1 é "pessoa real"
            if conf > 0.6 and cls == 1:
                liveness_verified = True
                liveness_check_done = True


                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                break

    return liveness_verified



def gen_frames():
    global progress, face_check_done, liveness_check_done

    cpf = re.sub(r'[^\w]', '', cpf_global)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Largura
    cap.set(4, 480)  # Altura

    model = YOLO("../models/large_v2_300.pt")
    model.to('cuda')

    face_path = f"rostos/{cpf}.jpg"
    documento_face_bytes = download_image_as_bytes(face_path)
    if documento_face_bytes is None:
        raise FileNotFoundError("Erro: Arquivo de rosto não encontrado no Firebase Storage.")
    documento_face_image = face_recognition.load_image_file(BytesIO(documento_face_bytes))
    documento_face_encoding = face_recognition.face_encodings(documento_face_image)[0]

    while True:
        success, frame = cap.read()
        if not success:
            break

        face_check_done = face_match(frame, documento_face_encoding)
        liveness_check_done = yolo_liveness(frame, model)

        # Atualizar o progresso conforme a verificação ocorre
        if face_check_done and liveness_check_done:
            progress = 100  # Autenticação completa
        elif face_check_done:
            progress = 50  # Progresso parcial
        else:
            progress = 25  # Início do processo

        # Enviar o frame processado
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_to_send = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

        time.sleep(1 / 30)

#-----------------------------------------------------------------------------------------------

# ----------------------------- ROTA: ACESSAR CONTA -----------------------------
@app.route('/acesso', methods=['GET', 'POST'])
def acessar_conta():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Autenticação com Firebase
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = user['idToken']

            # Recuperar CPF do banco com base no email
            users = db.child("usuarios").get().val()
            for cpf, user_data in users.items():
                if user_data['email'] == email:
                    session['cpf'] = cpf
                    break

            # Redirecionar para a página de cadastro de documento
            return redirect('/cadastro_documento')
        except Exception as e:
            print(f"Erro de autenticação: {e}")
            return "Credenciais inválidas", 401

    return render_template('acessar_conta.html')


if __name__ == '__main__':
    app.run(debug=True)


    
