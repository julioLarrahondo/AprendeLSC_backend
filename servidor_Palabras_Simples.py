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
model = TFSMLayer("modelos/Palabras_Simples/modelo_lsc_savedmodel", call_endpoint="serving_default")
with open("modelos/Palabras_Simples/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# === MediaPipe ===
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)

# === Usuarios por sesión ===
usuarios = {}  # sid: {"frames": [], "palabras": [], "parrafo": ""}

# === Limpieza periódica por sesión ===
def reiniciar_por_sesion():
    while True:
        time.sleep(15)
        for sid in list(usuarios.keys()):
            if usuarios[sid]["palabras"]:
                print(f"🕒 Limpieza para sesión {sid}")
                usuarios[sid]["frames"].clear()
                usuarios[sid]["palabras"].clear()

# === Extraer características ===
def extraer_caracteristicas(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado_manos = mp_hands.process(rgb)
    resultado_pose = mp_pose.process(rgb)
    resultado_face = mp_face.process(rgb)

    fila = []

    for mano in resultado_manos.multi_hand_landmarks or []:
        for lm in mano.landmark:
            fila.extend([lm.x, lm.y, lm.z])
    while len(fila) < 126:
        fila.extend([0, 0, 0])

    if resultado_pose.pose_landmarks:
        for lm in resultado_pose.pose_landmarks.landmark[:33]:
            fila.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        fila.extend([0, 0, 0, 0] * 33)

    if resultado_face.multi_face_landmarks:
        for lm in resultado_face.multi_face_landmarks[0].landmark[:468]:
            fila.extend([lm.x, lm.y, lm.z])
    else:
        fila.extend([0, 0, 0] * 468)

    return np.array(fila[:1662] + [0] * (1662 - len(fila)), dtype=np.float32)

# === Procesar imagen ===
def procesar_imagen(frame, sid):
    usuario = usuarios[sid]
    vector = extraer_caracteristicas(frame)
    usuario["frames"].append(vector)

    if len(usuario["frames"]) > 30:
        usuario["frames"] = usuario["frames"][-30:]

    if len(usuario["frames"]) == 30:
        secuencia_array = np.array([usuario["frames"]], dtype=np.float32)
        outputs = model(secuencia_array, training=False)

        if isinstance(outputs, dict) and 'output_0' in outputs:
            pred = outputs['output_0'].numpy()
            idx = np.argmax(pred)
            palabra = le.inverse_transform([idx])[0]
            prob = float(pred[0][idx])
            print(f"[{sid}] 🧠 Palabra: {palabra} ({prob:.2f})")
            return palabra, prob
    return None, None

# === API REST ===
@app.route('/api/translate', methods=['POST'])
def traducir_imagen():
    sid = request.args.get('sid')
    if not sid or sid not in usuarios:
        return jsonify({'error': 'Sesión inválida'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    npimg = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    palabra, prob = procesar_imagen(frame, sid)

    if palabra:
        usuario = usuarios[sid]
        usuario["palabras"].append(palabra)
        usuario["parrafo"] += palabra + " "

        data = {
            'palabra': palabra,
            'frase': ' '.join(usuario["palabras"]),
            'parrafo': usuario["parrafo"].strip(),
            'confianza': prob
        }

        socketio.emit('nueva_palabra', data, to=sid)
        return jsonify({'success': True, **data})
    else:
        return jsonify({'success': False, 'message': 'Esperando 30 frames'})

# === Conexiones WebSocket ===
@socketio.on('connect')
def on_connect():
    sid = request.sid
    usuarios[sid] = {"frames": [], "palabras": [], "parrafo": ""}
    print(f"✅ Conectado: {sid}")
    emit('sid', {'sid': sid})

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    usuarios.pop(sid, None)
    print(f"❌ Desconectado: {sid}")

# === Ruta raíz ===
@app.route('/')
def index():
    return "✅ Servidor para Palabras Simples corriendo"

# === Iniciar servidor ===
if __name__ == '__main__':
    threading.Thread(target=reiniciar_por_sesion, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5006)
