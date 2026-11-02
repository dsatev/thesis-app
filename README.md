---
title: Deepfake Detection API
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

<!-- The block above is Hugging Face Spaces metadata. Keep it if you deploy as a Space; remove it otherwise. -->

# Deepfake Detection API

A FastAPI backend that detects AI-generated images. It runs two fine-tuned convolutional neural networks, **Xception** and **ResNet50**, trained on the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset, and combines their outputs into an ensemble prediction.

Upload an image, get back a `REAL` / `FAKE` verdict with probabilities from each model and from the ensemble.

> The web interface for this API lives in [thesis-front](https://github.com/dsatev/thesis-front).

## Features

- Two independent models plus an ensemble result in a single response
- Simple REST API with one prediction endpoint
- Input validation (file type and size limits)
- Interactive API docs generated automatically by FastAPI
- Docker-ready, deployable as a Hugging Face Space

## How it works

1. The uploaded image is validated (JPEG, PNG or WEBP, max 10 MB).
2. It is preprocessed and passed through both the Xception and ResNet50 models.
3. Each model returns a probability that the image is fake or real.
4. The **ensemble** result is the simple average of the two models' probabilities (for example, 94.3% and 88.1% fake gives 91.2% fake).
5. The label is `FAKE` or `REAL`, depending on which class has the higher probability.

## Quick start

### Prerequisites

- Python 3.10 or newer
- The trained model files `xception_final.keras` and `resnet50_final.keras`

### Run locally

```bash
git clone https://github.com/dsatev/thesis-app.git
cd thesis-app

# Install dependencies
pip install -r requirements.txt

# Place your trained models in ./models/
#   models/xception_final.keras
#   models/resnet50_final.keras

# Start the server
python -m uvicorn app:app --reload --port 7860
```

The API is now available at <http://localhost:7860>, and interactive docs at <http://localhost:7860/docs>.

### Run with Docker

```bash
docker build -t deepfake-detection-api .
docker run -p 7860:7860 deepfake-detection-api
```

If the model files are not baked into the image, mount them as a volume:

```bash
docker run -p 7860:7860 -v $(pwd)/models:/app/models deepfake-detection-api
```

## API reference

| Method | Path       | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | Health check, also lists the loaded models   |
| POST   | `/predict` | Upload an image and get detection results    |

### `POST /predict`

**Request:** `multipart/form-data` with a single `file` field.

| Constraint | Value                |
|------------|----------------------|
| Formats    | JPEG, PNG, WEBP      |
| Max size   | 10 MB                |

**Example:**

```bash
curl -X POST \
  -F "file=@photo.jpg" \
  http://localhost:7860/predict
```

**Response:**

```json
{
  "filename": "photo.jpg",
  "models": {
    "xception": { "label": "FAKE", "confidence": 94.3, "fake_prob": 94.3, "real_prob": 5.7 },
    "resnet50": { "label": "FAKE", "confidence": 88.1, "fake_prob": 88.1, "real_prob": 11.9 }
  },
  "ensemble": { "label": "FAKE", "confidence": 91.2, "fake_prob": 91.2, "real_prob": 8.8 }
}
```

**Response fields**

| Field       | Description                                                          |
|-------------|----------------------------------------------------------------------|
| `label`     | Predicted class: `FAKE` or `REAL`                                    |
| `confidence`| Probability (in %) of the predicted class                            |
| `fake_prob` | Probability (in %) that the image is AI-generated                    |
| `real_prob` | Probability (in %) that the image is a real photograph               |

`fake_prob` and `real_prob` always sum to 100.

### `GET /`

Returns a simple status payload confirming the service is up and which models were loaded successfully.

## Project structure

```
thesis-app/
├── app.py             # FastAPI application and prediction logic
├── Dockerfile         # Container build for deployment
├── requirements.txt   # Python dependencies
├── index.html         # Simple built-in test page
├── style.css          # Styles for the test page
├── models/            # Trained .keras model files (not committed)
└── LICENSE
```

## Model files

The `.keras` files are large, so you have two options for providing them.

### Option A: upload them to the repository / Space

Upload the files into a `models/` folder, either through the Hugging Face web UI or the CLI:

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli upload <your-username>/<your-space> \
    /path/to/xception_final.keras models/xception_final.keras

huggingface-cli upload <your-username>/<your-space> \
    /path/to/resnet50_final.keras models/resnet50_final.keras
```

### Option B: download from the Hugging Face Hub at startup

To avoid repository size limits, store the models in a separate Hugging Face **dataset** repo and download them when the app starts:

```python
from huggingface_hub import hf_hub_download

for name, filename in [("xception", "xception_final.keras"),
                       ("resnet50", "resnet50_final.keras")]:
    path = hf_hub_download(
        repo_id="<your-username>/<your-models-repo>",
        filename=filename,
        repo_type="dataset",
    )
    models[name] = tf.keras.models.load_model(path)
```

## Deployment

The repository includes a `Dockerfile` configured for **Hugging Face Spaces** (Docker SDK, port `7860`). To deploy:

1. Create a new Space and choose **Docker** as the SDK.
2. Push this repository to the Space.
3. Provide the model files using one of the options above.

Any other container platform works too, since the app only needs to expose port 7860.

## Limitations

- The models were trained on CIFAKE, which consists of small, low-resolution images. Performance on high-resolution photos or images from newer generators may be noticeably worse.
- The output is a probabilistic estimate, not proof. Treat it as one signal among others, not a definitive judgment.

## Related

- [thesis-front](https://github.com/dsatev/thesis-front): web frontend for this API

## License

Released under the [MIT License](LICENSE).
