import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
IMG_SIZE  = (128, 128)

models: dict = {}

def load_models():
    for name, filename in [
        ("xception", "xception_finetuned_best.keras"),
    ]:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            print(f"Loading {name} from {path} …")
            models[name] = tf.keras.models.load_model(path)
            print(f"  ✓ {name} loaded")
        else:
            print(f"  ✗ {name} not found at {path} — skipping")

load_models()

# ── Helpers ───────────────────────────────────────────────────────────────────
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES     = 10 * 1024 * 1024  # 10 MB

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)

def predict_one(model, arr: np.ndarray) -> dict:
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if label == "FAKE" else 1 - prob
    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "fake_prob":  round(prob * 100, 2),
        "real_prob":  round((1 - prob) * 100, 2),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WEBP.")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")

    if not models:
        raise HTTPException(503, "No models are loaded. Check MODEL_DIR.")

    try:
        arr = preprocess(raw)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    model = models["xception"]
    result = predict_one(model, arr)

    return JSONResponse({
        "filename": file.filename,
        "label":      result["label"],
        "confidence": result["confidence"],
        "fake_prob":  result["fake_prob"],
        "real_prob":  result["real_prob"],
    })
