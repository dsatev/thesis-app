import os
import io
import gdown
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Download model from Google Drive at startup ───────────────────────────────
MODEL_PATH   = "./models/xception_finetuned_best.keras"

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs("./models", exist_ok=True)
    print("Downloading model from Hugging Face...")
    path = hf_hub_download(
        repo_id="dsatev/thesis-models",
        filename="xception_finetuned_best.keras",
        repo_type="dataset"
    )
    import shutil
    shutil.copy(cached, MODEL_PATH)
    print("Model downloaded to", MODEL_PATH)

download_model()

# ── Load model ────────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)
models: dict = {}

if os.path.exists(MODEL_PATH):
    print("Loading Xception model ...")
    models["xception"] = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
else:
    print("WARNING: model file not found.")

# ── Helpers ───────────────────────────────────────────────────────────────────
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES     = 10 * 1024 * 1024

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_one(model, arr: np.ndarray) -> dict:
    prob       = float(model.predict(arr, verbose=0)[0][0])
    label      = "FAKE" if prob >= 0.5 else "REAL"
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
        raise HTTPException(400, "Unsupported file type. Use JPEG, PNG, or WEBP.")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")
    if not models:
        raise HTTPException(503, "Model not loaded yet. Try again in a moment.")
    try:
        arr = preprocess(raw)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    result = predict_one(models["xception"], arr)

    return JSONResponse({
        "filename": file.filename,
        "models":   {"xception": result},
        "ensemble": result
    })
