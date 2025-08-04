import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import mediapipe as mp
import pickle
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# === Modelo y etiquetas ===
model = TFSMLayer("modelos/colores/modelo_lsc_savedmodel", call_endpoint="serving_default")
with open("modelos/colores/label_encoder_colores.pkl", "rb") as f:
    le = pickle.load(f)

# === MediaPipe inicialización global ===
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)

# === Estado por usuario ===
usuarios = {}  # { sid: {'secuencia': [], 'letras': [], 'parrafo': ''} }

# === Función de reinicio por usuario cada 15s ===
def reiniciar_letras_cada_15s():
    while True:
        time.sleep(15)
        for sid in list(usuarios.keys()):
            if usuarios[sid]['letras']:
                print(f"🕒 Reiniciando letras de usuario {sid}")
                usuarios[sid]['letras'].clear()
                usuarios[sid]['secuencia'].clear()

# === Extraer características ===
def extraer_caracteristicas(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado_manos = mp_hands.process(rgb)
    resultado_pose = mp_pose.process(rgb)
    resultado_face = mp_face.process(rgb)
    fila = []

    # Manos
    for mano in resultado_manos.multi_hand_landmarks or []:
        for lm in mano.landmark:
            fila.extend([lm.x, lm.y, lm.z])
    while len(fila) < 126:
        fila.extend([0, 0, 0])

    # Pose
    if resultado_pose.pose_landmarks:
        for lm in resultado_pose.pose_landmarks.landmark[:33]:
            fila.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        fila.extend([0, 0, 0, 0] * 33)

    # Rostro
    if resultado_face.multi_face_landmarks:
        for lm in resultado_face.multi_face_landmarks[0].landmark[:468]:
            fila.extend([lm.x, lm.y, lm.z])
    else:
        fila.extend([0, 0, 0] * 468)

    # Ajuste final
    if len(fila) > 1662:
        fila = fila[:1662]
    elif len(fila) < 1662:
        fila.extend([0] * (1662 - len(fila)))

    return np.array(fila, dtype=np.float32)

# === Procesar imagen ===
def procesar_imagen(sid, frame):
    vector = extraer_caracteristicas(frame)
    usuarios[sid]['secuencia'].append(vector)

    if len(usuarios[sid]['secuencia']) > 30:
        usuarios[sid]['secuencia'] = usuarios[sid]['secuencia'][-30:]

    if len(usuarios[sid]['secuencia']) == 30:
        secuencia_array = np.array([usuarios[sid]['secuencia']], dtype=np.float32)
        outputs = model(secuencia_array, training=False)

        if isinstance(outputs, dict) and 'output_0' in outputs:
            pred = outputs['output_0'].numpy()
            etiqueta_idx = np.argmax(pred)
            letra = le.inverse_transform([etiqueta_idx])[0]
            prob = float(pred[0][etiqueta_idx])
            print(f"🧠 [{sid}] Letra: {letra} ({prob:.2f})")
            return letra, prob
    return None, None

# === Endpoint HTTP ===
@app.route('/api/translate', methods=['POST'])
def traducir_imagen():
    sid = request.args.get('sid')
    if not sid or sid not in usuarios:
        return jsonify({'error': 'SID inválido o no conectado'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    image = request.files['image']
    npimg = np.frombuffer(image.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    letra, prob = procesar_imagen(sid, frame)

    if letra:
        usuarios[sid]['letras'].append(letra)
        usuarios[sid]['parrafo'] += letra

        data = {
            'letra': letra,
            'frase': ''.join(usuarios[sid]['letras']),
            'parrafo': usuarios[sid]['parrafo'],
            'confianza': prob
        }

        socketio.emit('nueva_letra', data, to=sid)
        return jsonify({'success': True, **data})
    else:
        return jsonify({'success': False, 'message': 'Esperando completar 30 frames'})

# === WebSocket: Conexión / Desconexión ===
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    usuarios[sid] = {'secuencia': [], 'letras': [], 'parrafo': ''}
    print(f"🟢 Usuario conectado: {sid}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    usuarios.pop(sid, None)
    print(f"🔴 Usuario desconectado: {sid}")

# === Ruta raíz ===
@app.route('/')
def index():
    return "✅ Servidor multisesión de LSC activo"

# === Iniciar servidor ===
if __name__ == '__main__':
    threading.Thread(target=reiniciar_letras_cada_15s, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5002)
