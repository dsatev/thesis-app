import os
import io
import shutil
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Download model from HF Hub at startup ─────────────────────────────────────
MODEL_PATH = "./models/xception_finetuned.onnx"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists, skipping download.")
        return
    os.makedirs("./models", exist_ok=True)
    print("Downloading ONNX model from Hugging Face Hub...")
    cached = hf_hub_download(
        repo_id="dsatev/thesis-models",
        filename="xception_finetuned.onnx",
        repo_type="dataset"
    )
    shutil.copy(cached, MODEL_PATH)
    print(f"Model downloaded ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")

download_model()

# ── Load ONNX model ───────────────────────────────────────────────────────────
session = None
input_name = None

try:
    print("Loading ONNX model...")
    session    = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model loaded. Input: {input_name} {session.get_inputs()[0].shape}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    import traceback
    traceback.print_exc()

# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_SIZE      = (128, 128)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES     = 10 * 1024 * 1024

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)

def predict_one(arr: np.ndarray) -> dict:
    outputs = session.run(None, {input_name: arr})
    prob    = float(outputs[0][0][0])
    label   = "FAKE" if prob >= 0.5 else "REAL"
    conf    = prob if label == "FAKE" else 1 - prob
    return {
        "label":      label,
        "confidence": round(conf * 100, 2),
        "fake_prob":  round(prob * 100, 2),
        "real_prob":  round((1 - prob) * 100, 2),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status":       "ok",
        "model_loaded": session is not None,
        "models_loaded": ["xception"] if session else []
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported file type. Use JPEG, PNG, or WEBP.")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")
    if session is None:
        raise HTTPException(503, "Model not loaded. Please try again in a moment.")
    try:
        arr = preprocess(raw)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    result = predict_one(arr)

    return JSONResponse({
        "filename": file.filename,
        "models":   {"xception": result},
        "ensemble": result
    })
