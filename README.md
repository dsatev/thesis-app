---
title: Deepfake Detection API
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Deepfake Detection API

FastAPI backend that detects AI-generated images using a fine-tuned Xception
model trained on the CIFAKE dataset.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + loaded models |
| POST | `/predict` | Upload an image, get detection results |

### POST `/predict`

**Request:** `multipart/form-data` with a `file` field (JPEG / PNG / WEBP, max 10 MB)

**Response:**
```json
{
  "filename": "photo.jpg",
  "models": {
    "xception": { "label": "FAKE", "confidence": 94.3, "fake_prob": 94.3, "real_prob": 5.7 },
    "resnet50":  { "label": "FAKE", "confidence": 88.1, "fake_prob": 88.1, "real_prob": 11.9 }
  },
  "ensemble": { "label": "FAKE", "confidence": 91.2, "fake_prob": 91.2, "real_prob": 8.8 }
}
```

## Uploading your models

### Option A — Upload files directly (simplest)

After creating the Space, upload your `.keras` files via the HF web UI or CLI:

```bash
pip install huggingface_hub
huggingface-cli login

# Upload both models
huggingface-cli upload YOUR_USERNAME/deepfake-detection-api \
    /path/to/xception_final.keras models/xception_final.keras

huggingface-cli upload YOUR_USERNAME/deepfake-detection-api \
    /path/to/resnet50_final.keras  models/resnet50_final.keras
```

### Option B — Load from HF Hub at startup (avoids repo size limits)

Upload models to a **dataset** repo (supports large files), then load in `app.py`:

```python
from huggingface_hub import hf_hub_download

for name, filename in [("xception", "xception_final.keras"), ("resnet50", "resnet50_final.keras")]:
    path = hf_hub_download(repo_id="YOUR_USERNAME/thesis-models",
                           filename=filename, repo_type="dataset")
    models[name] = tf.keras.models.load_model(path)
```

## Local development

```bash
# Put your models in ./models/
python -m uvicorn app:app --reload --port 7860
# Visit http://localhost:7860
```
