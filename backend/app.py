"""
Agriculture Analysis — Backend API
-----------------------------------
Serves a trained CNN (chilli disease classifier) behind a simple
/predict endpoint that the frontend (index.html) calls.

Drop your trained model file into ./model/ and update MODEL_PATH
and CLASS_NAMES below to match your training setup.

Run locally:
    pip install -r requirements.txt
    python app.py

Then open index.html in a browser (or via Live Server) — the
frontend already points at http://localhost:5000/predict.
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # allow the frontend (served from a different origin/file) to call this API

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your trained model
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "vgg16_chili_final.keras"
)
IMG_SIZE = (224, 224)  # match whatever size you trained on (224x224 is VGG16 default)

# Order MUST match the order of your model's output classes (e.g. from
# train_generator.class_indices if you used Keras' ImageDataGenerator).
# Keys on the right map 1:1 to the CONDITIONS object in index.html's <script>.
CLASS_NAMES = [
    "healthy",        # index 0
    "leafCurl",       # index 1
    "anthracnose",    # index 2
    "bacterialSpot",  # index 3
    "powderyMildew",  # index 4
    "cercospora",     # index 5
]

# Rough severity score (0-100) shown on the frontend's severity bar,
# used only when the model doesn't output its own severity estimate.
BASE_SEVERITY = {
    "healthy": 8,
    "leafCurl": 70,
    "anthracnose": 75,
    "bacterialSpot": 55,
    "powderyMildew": 45,
    "cercospora": 50,
}

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

model = None
model_load_error = None

def load_model():
    """Loads the Keras/TensorFlow model. Swap this out if you're using
    PyTorch instead — just make predict_image() return a probability
    array in the same CLASS_NAMES order."""
    global model, model_load_error
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(MODEL_PATH)
        print(f"[startup] Loaded model from {MODEL_PATH}")
    except Exception as e:
        model_load_error = str(e)
        print(f"[startup] Could not load model ({e}). "
              f"/predict will return an error until a valid model is placed at {MODEL_PATH}")

load_model()


def preprocess_image(file_bytes):
    """Resize + normalize an uploaded image for the model."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model is not None else "model_not_loaded",
        "error": model_load_error,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send multipart/form-data with key 'image'."}), 400

    if model is None:
        return jsonify({
            "error": "Model not loaded on the server.",
            "detail": model_load_error,
        }), 503

    file = request.files["image"]
    try:
        file_bytes = file.read()
        input_arr = preprocess_image(file_bytes)
        predictions = model.predict(input_arr)[0]  # shape: (num_classes,)

        best_idx = int(np.argmax(predictions))
        label = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else "unknown"
        confidence = float(predictions[best_idx]) * 100
        severity = BASE_SEVERITY.get(label, 50)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 1),
            "severity": severity,
            "raw_scores": {CLASS_NAMES[i]: round(float(p) * 100, 1) for i, p in enumerate(predictions) if i < len(CLASS_NAMES)},
        })

    except Exception as e:
        return jsonify({"error": "Failed to process image", "detail": str(e)}), 500


if __name__ == "__main__":
    # debug=True is fine for local development; turn off in production
    app.run(host="0.0.0.0", port=5000, debug=True)
