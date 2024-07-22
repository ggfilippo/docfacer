import os
from flask import Flask, render_template, render_template_string, request, redirect, url_for, session, jsonify
from functools import wraps
from werkzeug.utils import secure_filename

import io
from PIL import Image
import face_recognition
from werkzeug.utils import secure_filename
import numpy as np
import urllib.request

import pyrebase

import cv2
import pickle

from face_detection import detect_faces

from rapidfuzz import process, fuzz
import easyocr
import re

app = Flask(__name__)

app.config.update(SECRET_KEY=os.urandom(24))

config = {
    'apiKey': "",
    'authDomain': "",
    'databaseURL': "",
    'projectId': "",
    'storageBucket': "",
    'messagingSenderId': "",
    'appId': ""
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
storage = firebase.storage()
db = firebase.database()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not password or not confirm_password:
            return render_template('signup.html', signup_error='Todos os campos são obrigatórios.')

        if password != confirm_password:
            return render_template('signup.html', signup_error='As senhas não coincidem.')
        
        try:
            user = auth.create_user_with_email_and_password(email, password)
            session['user'] = user
            return render_template('login.html', signup_success='Cadastro realizado com sucesso!')
        
        except Exception as e:
            app.logger.error(f'Erro ao criar conta: {e}')
            return render_template_string(open('templates/signup.html').read(), signup_error='Erro ao criar a conta, tente novamente', show_signup=True)

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = user
            return redirect(url_for('home'))
        except Exception as e:
            app.logger.error(f'Erro ao logar: {e}')
            return render_template('login.html', message='Login falhou. Verifique suas credenciais e tente novamente.', category='error')
    return render_template('login.html')

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

@app.route('/submit-your-form-handler', methods=['POST'])
def submit_form():
    nome_completo = request.form['nomeCompleto'].upper()
    cpf = request.form['cpf']
    rg = request.form['rg']
    data_expedicao = request.form['dataExpedicao']
    data_nascimento = request.form['dataNascimento']
    nome_mae = request.form['nomeMae'].upper()
    
    if 'input-file' not in request.files or 'input-file2' not in request.files:
        return "Arquivo necessário não foi enviado", 400

    frente_documento = request.files['input-file']
    verso_documento = request.files['input-file2']

    if not frente_documento or not verso_documento:
        return "Arquivo enviado está vazio", 400
    
    # Upload de arquivos
    frente_documento = request.files['input-file']
    verso_documento = request.files['input-file2']
    if frente_documento and verso_documento:
        frente_filename = secure_filename(f"{cpf}_frente.jpg")
        verso_filename = secure_filename(f"{cpf}_verso.jpg")
        frente_path = f"documentos/{frente_filename}"
        verso_path = f"documentos/{verso_filename}"
        storage.child(frente_path).put(frente_documento)
        storage.child(verso_path).put(verso_documento)
        frente_url = storage.child(frente_path).get_url(None)
        verso_url = storage.child(verso_path).get_url(None)
    else:
        return "Arquivo não encontrado", 400
    
    img_frente = Image.open(frente_documento)
    img_frente.save(f"{cpf}_frente_doc_temp.jpg")
    img_frente_np = np.array(img_frente)

    img_verso = Image.open(verso_documento)
    img_verso.save(f"{cpf}_verso_doc_temp.jpg")
    img_verso_np = np.array(img_verso)


    # Processamento OCR
    reader = easyocr.Reader(['pt', 'en'], gpu=False)
    result = reader.readtext(img_frente_np, detail=0)
    result = [remove_special_characters(text.upper()) for text in result]
    print(f"Resultado: {result}")

    os.remove(f"{cpf}_verso_doc_temp.jpg")

    # Função para verificar correspondência fuzzy
    def verificar_correspondencia(valor, lista):
        match = process.extractOne(valor, lista, scorer=fuzz.ratio)
        if match and match[1] >= 85:  # % de similaridade
            return match[0], round(match[1], 1)
        return None, 0
    
    nome_match, nome_score = verificar_correspondencia(nome_completo, result)
    data_nascimento_match, data_nascimento_score = verificar_correspondencia(data_nascimento.upper(), result)
    cpf_match, cpf_score = verificar_correspondencia(cpf, result)
    nome_mae_match, nome_mae_score = verificar_correspondencia(nome_mae, result)

    # Prepara os resultados para exibição
    resultados = {
        'nome': {'valor': nome_completo, 'match': nome_match, 'score': nome_score},
        'data_nascimento': {'valor': data_nascimento, 'match': data_nascimento_match, 'score': data_nascimento_score},
        'cpf': {'valor': cpf, 'match': cpf_match, 'score': cpf_score},
        'nome_mae': {'valor': nome_mae, 'match': nome_mae_match, 'score': nome_mae_score},
    }

    # Carregar a imagem da frente temporariamente
    img = Image.open(frente_documento)

    #try:
    #    img = img.rotate(90, expand=True)
    #except Exception as e:
    #    print(f"Erro ao tentar acessar EXIF: {e}")

    #img.save(f"{cpf}_temp.jpg")

    # Converter para numpy array para usar em face_recognition
    img_np = np.array(img)

    # Detecção de rosto
    face_locations = face_recognition.face_locations(img_np)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_image = img_np[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        buffer.seek(0)
        face_filename = secure_filename(f"{cpf}_face.jpg")
        face_path = f"documentos/{face_filename}"
        storage.child(face_path).put(buffer)
        face_url = storage.child(face_path).get_url(None)
    else:
        face_url = "No face detected"

    # Salva os dados no Firebase Database
    user_data = {
        "nomeCompleto": nome_completo,
        "cpf": cpf,
        "rg": rg,
        "dataExpedicao": data_expedicao,
        "dataNascimento": data_nascimento,
        "nomeMae": nome_mae,
        "frenteDocumentoUrl": frente_url,
        "versoDocumentoUrl": verso_url,
        "faceImageUrl": face_url

    }

    db.child("Users").child(cpf).set(user_data)

    
    def url_to_image(url):
        # Faz o download da imagem, converte para um array de bytes, e depois decodifica em um array NumPy
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    img_face = url_to_image(face_url)

    personId = cpf

    def findEncodings(img):
        encodeList = []
        img = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        return encodeList
    
    print('Encoding Started...')
    encodeListKnown = findEncodings(img_face)
    encodeListKnownWithId = [encodeListKnown, personId]
    print('Encoding Completed')

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithId, file)
    file.close()
    print('Encoding File Saved')

    session['form_submitted'] = True

    return render_template('ocr_result_fragment.html', resultados=resultados)

@app.route('/')
@login_required
def home():
    if session.get('form_submitted'):
        session.pop('form_submitted', None)
        return render_template('formRegisterDoc.html', show_continue_button=True)
    return render_template('formRegisterDoc.html')

@app.route('/face_detection')
@login_required
def pageFace_detection():
    return render_template('face_detection.html')

def run_detection():
    detect_faces()

@app.route('/start-recognition', methods=['POST'])
def start_recognition():
    run_detection()
    return render_template('conclusion.html')

@app.route('/finish', methods=['POST'])
def finish():
     return render_template('conclusion.html')

if __name__ == '__main__':
    app.run(debug=True)