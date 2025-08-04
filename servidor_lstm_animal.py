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

# === Cargar modelo y etiquetas ===
model = TFSMLayer("modelos/animales/Modelo_animales", call_endpoint="serving_default")
with open("modelos/animales/label_encoder (1).pkl", "rb") as f:
    le = pickle.load(f)

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)

# === Estado de usuarios ===
usuarios = {}  # { sid: {secuencia_frames: [], letras_detectadas: [], parrafo: ""} }

# === Limpieza automática por sesión cada 15 segundos ===
def reiniciar_letras_periodicamente():
    while True:
        time.sleep(15)
        for sid in list(usuarios.keys()):
            if usuarios[sid]["letras_detectadas"]:
                print(f"🕒 Limpiando datos para {sid}")
                usuarios[sid]["letras_detectadas"].clear()
                usuarios[sid]["secuencia_frames"].clear()

# === Extraer características del frame ===
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

    # Face
    if resultado_face.multi_face_landmarks:
        for lm in resultado_face.multi_face_landmarks[0].landmark[:468]:
            fila.extend([lm.x, lm.y, lm.z])
    else:
        fila.extend([0, 0, 0] * 468)

    # Padding
    fila = (fila + [0] * (1662 - len(fila)))[:1662]
    return np.array(fila, dtype=np.float32)

# === Procesar imagen y predecir ===
def procesar_imagen(frame, sid):
    usuario = usuarios[sid]
    vector = extraer_caracteristicas(frame)
    usuario["secuencia_frames"].append(vector)

    if len(usuario["secuencia_frames"]) > 90:
        usuario["secuencia_frames"] = usuario["secuencia_frames"][-90:]

    if len(usuario["secuencia_frames"]) == 90:
        secuencia_array = np.array([usuario["secuencia_frames"]], dtype=np.float32)
        outputs = model(secuencia_array, training=False)

        if isinstance(outputs, dict) and 'output_0' in outputs:
            pred = outputs['output_0'].numpy()
            idx = np.argmax(pred)
            letra = le.inverse_transform([idx])[0]
            prob = float(pred[0][idx])
            print(f"[{sid}] 🧠 Predicción: {letra} ({prob:.2f})")
            return letra, prob
    return None, None

# === Endpoint de API ===
@app.route('/api/translate', methods=['POST'])
def traducir_imagen():
    sid = request.args.get('sid')
    if not sid or sid not in usuarios:
        return jsonify({'error': 'Sesión no encontrada o inválida'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    image = request.files['image']
    npimg = np.frombuffer(image.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    letra, prob = procesar_imagen(frame, sid)

    if letra:
        usuario = usuarios[sid]
        usuario["letras_detectadas"].append(letra)
        usuario["parrafo"] += letra

        data = {
            'letra': letra,
            'frase': ''.join(usuario["letras_detectadas"]),
            'parrafo': usuario["parrafo"].strip(),
            'confianza': float(prob)
        }

        socketio.emit('nueva_letra', data, to=sid)
        print(f"[{sid}] Letra enviada:", data)
        return jsonify({'success': True, **data})
    else:
        return jsonify({'success': False, 'message': 'Esperando completar 90 frames'})

# === Eventos Socket.IO ===
@socketio.on('connect')
def on_connect():
    sid = request.sid
    usuarios[sid] = {
        "secuencia_frames": [],
        "letras_detectadas": [],
        "parrafo": ""
    }
    print(f"✅ Usuario conectado: {sid}")
    emit('sid', {'sid': sid})

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    usuarios.pop(sid, None)
    print(f"❌ Usuario desconectado: {sid}")

# === Ruta principal ===
@app.route('/')
def index():
    return "✅ Servidor de lenguaje de señas (animales) corriendo y listo"

# === Ejecutar servidor ===
if __name__ == '__main__':
    threading.Thread(target=reiniciar_letras_periodicamente, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5001)
