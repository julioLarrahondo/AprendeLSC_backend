from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import joblib
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# === CARGAR MODELO ===
model = joblib.load("modelos/vocales/modelo_random_forest2.pkl")
with open("modelos/vocales/etiquetas.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1)

# === ESTADO POR USUARIO ===
usuarios = {}  # { sid: { 'letras': [], 'parrafo': "" } }

# === PROCESAMIENTO DE IMAGEN ===
def procesar_imagen(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        x_vals = [lm.x for lm in landmarks]
        y_vals = [lm.y for lm in landmarks]
        z_vals = [lm.z for lm in landmarks]

        ordered_keypoints = x_vals + y_vals + z_vals
        keypoints_flat = np.array(ordered_keypoints, dtype=np.float64).reshape(1, -1)

        prediction = model.predict(keypoints_flat)
        letra_detectada = prediction[0]
        return letra_detectada

    return None

# === ENDPOINT HTTP ===
@app.route('/api/translate', methods=['POST'])
def translate_image():
    sid = request.args.get('sid')
    if not sid or sid not in usuarios:
        return jsonify({'error': 'SID inválido o no registrado'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    image = request.files['image']
    npimg = np.frombuffer(image.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    letra = procesar_imagen(frame)
    if letra:
        usuarios[sid]['letras'].append(letra)
        palabra_actual = ''.join(usuarios[sid]['letras']).lower()

        socketio.emit('nueva_letra', {
            'letra': letra,
            'frase': palabra_actual,
            'arreglo': usuarios[sid]['letras'],
        }, to=sid)

        return jsonify({
            'success': True,
            'letra': letra,
            'frase': palabra_actual,
            'arreglo': usuarios[sid]['letras']
        })
    else:
        return jsonify({'success': False, 'message': 'No se detectó ninguna letra'})

# === SOCKETIO CONEXIÓN ===
@socketio.on('connect')
def manejar_conexion():
    sid = request.sid
    usuarios[sid] = { 'letras': [], 'parrafo': "" }
    print(f"🟢 Usuario conectado: {sid}")

@socketio.on('disconnect')
def manejar_desconexion():
    sid = request.sid
    usuarios.pop(sid, None)
    print(f"🔴 Usuario desconectado: {sid}")

@socketio.on('seleccion_palabra')
def manejar_seleccion(data):
    sid = request.sid
    if sid not in usuarios:
        return

    letras = usuarios[sid]['letras']
    if letras:
        palabra_final = ''.join(letras).lower()
        usuarios[sid]['parrafo'] += palabra_final + " "
        usuarios[sid]['letras'] = []

        socketio.emit('actualizar_parrafo', {
            'parrafo': usuarios[sid]['parrafo'].strip()
        }, to=sid)

# === PRUEBA ===
@app.route('/')
def index():
    return "✅ Servidor multisesión de reconocimiento de vocales activo"

# === EJECUCIÓN ===
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
