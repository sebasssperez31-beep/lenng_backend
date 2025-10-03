# app.py
from flask import Flask, Response, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import os
import numpy as np
import tensorflow as tf
import zipfile
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
import json

# ------------------------
# CONFIG
# ------------------------
app = Flask(__name__)
CORS(app)

DATASET_DIR = "datasets"      # datasets/<categoria>/<label>/*.npy
MODEL_DIR = "models"         # models/<categoria>/model.h5 + model_classes.json
UPLOAD_FOLDER = "uploads"    # uploaded zip files
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALLOWED_MODEL_EXT = {"h5"}
MAX_SAMPLES_PER_LABEL = 30

# ------------------------
# Mediapipe Hands
# ------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ------------------------
# Cámara
# ------------------------
cap = cv2.VideoCapture(0)

current_label = None
current_category = None
collecting = False

# ------------------------
# Utilities
# ------------------------
def _now_ts():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

def allowed_model_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_MODEL_EXT

def save_landmarks_npy(category: str, label: str, landmarks, base_dir=DATASET_DIR):
    """
    Guarda un .npy 1D con shape (63,) -> 21 * 3
    Retorna la ruta si guardó, None si ya alcanzó el límite.
    """
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    label_dir = os.path.join(base_dir, category, label)
    os.makedirs(label_dir, exist_ok=True)

    existing = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    if len(existing) >= MAX_SAMPLES_PER_LABEL:
        return None

    fname = f"sample_{_now_ts()}.npy"
    path = os.path.join(label_dir, fname)
    np.save(path, arr)
    return path

def collect_npy_samples_from_folder(base_dir):
    """
    Recorre base_dir recursivamente, busca .npy y devuelve X_list, y_list, classes, stats
    Si encuentra imágenes (jpg/png) intentará extraer landmarks usando mediapipe.
    """
    X, y, classes, stats = [], [], [], {}
    if not os.path.exists(base_dir):
        return X, y, classes, stats

    for root, _, files in os.walk(base_dir):
        npy_files = [fn for fn in files if fn.lower().endswith(".npy")]
        img_files = [fn for fn in files if fn.lower().endswith((".jpg", ".jpeg", ".png"))]

        label = os.path.basename(root)
        if not npy_files and not img_files:
            continue

        # cargar .npy
        for fn in npy_files:
            try:
                arr = np.load(os.path.join(root, fn)).astype(np.float32).flatten()
                X.append(arr)
                y.append(label)
            except Exception:
                continue

        # intentar extraer landmarks de imágenes si las hay
        for fn in img_files:
            try:
                img_path = os.path.join(root, fn)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
                    X.append(arr)
                    y.append(label)
            except Exception:
                continue

        stats[label] = stats.get(label, 0) + len(npy_files) + len(img_files)
        if label not in classes:
            classes.append(label)

    return X, y, classes, stats

# ------------------------
# Stream (video feed + landmarks + optional prediction)
# ------------------------
def gen_frames(predict_mode=False, model=None, labels=None):
    global collecting, current_label, current_category

    while True:
        success, frame = cap.read()
        if not success:
            import time
            time.sleep(0.1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # guardar .npy si estamos colectando
                if collecting and current_label and current_category:
                    try:
                        saved = save_landmarks_npy(current_category, current_label, hand_landmarks.landmark, base_dir=DATASET_DIR)
                        if saved is None:
                            cv2.putText(frame, "Limite alcanzado", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    except Exception as e:
                        print("Error guardando landmarks:", e)

                # predicción en vivo
                if predict_mode and model is not None:
                    try:
                        lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32).flatten()
                        # adaptar al tamaño del modelo
                        try:
                            expected_len = int(model.inputs[0].shape[1])
                        except Exception:
                            expected_len = lm.shape[0]
                        if lm.shape[0] < expected_len:
                            lm = np.pad(lm, (0, expected_len - lm.shape[0]), mode="constant")
                        elif lm.shape[0] > expected_len:
                            lm = lm[:expected_len]
                        x = lm.reshape(1, -1)
                        probs = model.predict(x, verbose=0)[0]
                        idx = int(np.argmax(probs))
                        label_text = labels[idx] if 0 <= idx < len(labels) else str(idx)
                        conf = float(np.max(probs))
                        cv2.putText(frame, f"{label_text} ({conf:.2f})", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    except Exception as e:
                        print("Error predicción en stream:", e)

        # codificar y enviar
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------------
# Captura control
# ------------------------
@app.route("/start_capture", methods=["POST"])
def start_capture():
    global collecting, current_label, current_category
    data = request.json or {}
    current_label = data.get("label")
    current_category = data.get("category")
    if not current_label or not current_category:
        return jsonify({"error": "Faltan parámetros (label, category)"}), 400
    collecting = True
    return jsonify({"status": f"Capturando landmarks para {current_category}/{current_label}..."})

@app.route("/stop_capture", methods=["POST"])
def stop_capture():
    global collecting
    collecting = False
    return jsonify({"status": "Captura detenida"})

# ------------------------
# Descargar dataset por categoría (datasets/<category> -> category.zip)
# ------------------------
@app.route("/dataset_stats", methods=["GET"])
def dataset_stats():
    category = request.args.get("category")
    if not category:
        return jsonify({"error": "Debes indicar ?category=nombre"}), 400

    category_dir = os.path.join(DATASET_DIR, category)
    if not os.path.exists(category_dir):
        return jsonify({"error": f"No existe la categoría {category}"}), 404

    labels = []
    values = []
    for label in sorted(os.listdir(category_dir)):
        label_path = os.path.join(category_dir, label)
        if not os.path.isdir(label_path):
            continue
        count = len([f for f in os.listdir(label_path) if f.endswith(".npy")])
        labels.append(label)
        values.append(count)

    return jsonify({"labels": labels, "values": values})

# ------------------------
# Subir dataset (zip) -> datasets/<categoria>
#  - usa el nombre del zip para determinar la categoría (ej: vocales.zip -> vocales)
#  - guarda zip en uploads/
# ------------------------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Falta archivo"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Archivo vacío"}), 400

    if file and file.filename.lower().endswith(".zip"):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # categoría = nombre del archivo sin extensión
        category = os.path.splitext(filename)[0]

        category_dir = os.path.join(DATASET_DIR, category)
        if os.path.exists(category_dir):
            shutil.rmtree(category_dir)
        os.makedirs(category_dir, exist_ok=True)

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(category_dir)

        return jsonify({"status": f"Dataset {category} cargado con éxito", "category": category})
    
    return jsonify({"error": "Formato no permitido (usa .zip)"}), 400

# ------------------------
# Listar datasets (carpetas dentro de DATASET_DIR)
# Devuelve: [{ "category": name, "labels": {label:count}, "total_samples": N }, ...]
# ------------------------
@app.route("/list_datasets", methods=["GET"])
def list_datasets():
    out = []
    if not os.path.exists(DATASET_DIR):
        return jsonify(out)
    for cat in sorted(os.listdir(DATASET_DIR)):
        cat_path = os.path.join(DATASET_DIR, cat)
        if not os.path.isdir(cat_path):
            continue
        stats = {}
        total = 0
        for label in sorted(os.listdir(cat_path)):
            label_path = os.path.join(cat_path, label)
            if not os.path.isdir(label_path):
                continue
            count = len([f for f in os.listdir(label_path) if f.endswith(".npy")])
            stats[label] = count
            total += count
        out.append({"category": cat, "labels": stats, "total_samples": total})
    return jsonify(out)

# ------------------------
# Listar zips subidos en uploads/
# ------------------------
@app.route("/uploaded_zips", methods=["GET"])
def uploaded_zips():
    out = []
    for fn in sorted(os.listdir(UPLOAD_FOLDER)):
        if fn.lower().endswith(".zip"):
            path = os.path.join(UPLOAD_FOLDER, fn)
            size = os.path.getsize(path)
            out.append({"file": fn, "category": os.path.splitext(fn)[0], "size": size})
    return jsonify(out)

# ------------------------
# Entrenamiento por categoría
# ------------------------
@app.route("/train", methods=["POST"])
def train():
    data = request.get_json() or {}
    category = data.get("category")
    if not category:
        return jsonify({"error": "Debes indicar 'category'"}), 400
    base_dir = os.path.join(DATASET_DIR, category)

    X_list, y_list, classes, stats = collect_npy_samples_from_folder(base_dir)
    if not X_list:
        return jsonify({"error": "No se encontraron muestras para entrenar", "stats": stats}), 400

    lengths = [len(x) for x in X_list]
    max_len = max(lengths)
    X = np.zeros((len(X_list), max_len), dtype=np.float32)
    for i, arr in enumerate(X_list):
        ln = len(arr)
        if ln >= max_len:
            X[i, :] = arr[:max_len]
        else:
            X[i, :ln] = arr

    classes_ordered = sorted(list(set(classes)))
    label_to_idx = {lab: i for i, lab in enumerate(classes_ordered)}
    y_idx = np.array([label_to_idx[lbl] for lbl in y_list], dtype=np.int32)

    # modelo simple (ajusta a tu gusto)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_len,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(classes_ordered), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Entrenamiento (esto bloquea la petición; para producción usar job/cola)
    model.fit(X, y_idx, epochs=20, batch_size=16, verbose=1)

    # guardar modelo por categoría
    cat_model_dir = os.path.join(MODEL_DIR, category)
    os.makedirs(cat_model_dir, exist_ok=True)
    model.save(os.path.join(cat_model_dir, "model.h5"))
    with open(os.path.join(cat_model_dir, "model_classes.json"), "w", encoding="utf-8") as fh:
        json.dump(classes_ordered, fh, ensure_ascii=False)

    return jsonify({"status": f"Modelo '{category}' entrenado", "labels": classes_ordered, "stats": stats})

# ------------------------
# Listar modelos disponibles (categorías con modelos)
# ------------------------
@app.route("/list_models", methods=["GET"])
def list_models():
    out = []
    for cat in sorted(os.listdir(MODEL_DIR)):
        cat_path = os.path.join(MODEL_DIR, cat)
        if os.path.isdir(cat_path) and os.path.exists(os.path.join(cat_path, "model.h5")):
            out.append(cat)
    return jsonify(out)



@app.route("/list_categories")
def list_categories():
    if not os.path.exists(DATASET_DIR):
        return jsonify({"categories": []})
    cats = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return jsonify({"categories": cats})



# ------------------------
# Descargar modelo por categoría (models/<cat> -> cat_model.zip)
# ------------------------
@app.route("/download_model")
def download_model():
    category = request.args.get("category")
    if not category:
        return jsonify({"error": "Debes indicar ?category=nombre"}), 400
    folder = os.path.join(MODEL_DIR, category)
    if not os.path.exists(folder):
        return jsonify({"error": f"No existe modelo para la categoría {category}"}), 404

    zip_path = f"{category}_model.zip"
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception:
            pass
    shutil.make_archive(f"{category}_model", "zip", folder)
    return send_file(zip_path, as_attachment=True)

# ------------------------
# Predicción por categoría (stream)
# ------------------------
@app.route("/predict_feed")
def predict_feed():
    category = request.args.get("category")
    if not category:
        return jsonify({"error": "Debes indicar ?category=nombre"}), 400

    cat_model_dir = os.path.join(MODEL_DIR, category)
    model_path = os.path.join(cat_model_dir, "model.h5")
    classes_path = os.path.join(cat_model_dir, "model_classes.json")

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return jsonify({"error": f"No hay modelo entrenado para {category}"}), 400

    try:
        model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r", encoding="utf-8") as fh:
            labels = json.load(fh)
    except Exception as e:
        return jsonify({"error": f"Error cargando modelo: {e}"}), 500

    return Response(
        gen_frames(predict_mode=True, model=model, labels=labels),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ------------------------
# Graceful exit
# ------------------------
import atexit
@atexit.register
def _cleanup():
    try:
        if cap and cap.isOpened():
            cap.release()
    except Exception:
        pass

@app.route("/")
def home():
    return "Hola Render! El backend está corriendo."

# ------------------------
if __name__ == "__main__":
    print("Backend iniciado en http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
