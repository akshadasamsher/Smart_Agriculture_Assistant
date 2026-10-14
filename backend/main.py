"""
FastAPI backend - Unit 6 (Web Application integration layer)

Serves:
  - POST /predict/crop      -> ANN/MLP crop recommendation
  - POST /predict/disease   -> leaf image -> disease class (EfficientNetV2-S)
  - GET  /health            -> readiness check (shows which models are loaded)

Run:
    uvicorn main:app --reload --port 8000

Expects model artifacts (produced by training/crop_recommendation_train.ipynb)
in backend/models/recommendation/:
    crop_mlp_model.joblib
    crop_scaler.joblib
    crop_label_encoder.joblib
Missing artifacts are handled gracefully -> /predict/crop returns a 503
with a clear message instead of crashing the whole app.

"""

import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from schemas import CropPrediction, DiseasePrediction, SoilInput

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "models"))
RECOMMENDATION_DIR = MODEL_DIR / "recommendation"

app = FastAPI(
    title="Smart Agriculture Assistant API",
    description="Crop recommendation (ANN/MLP) + plant disease detection (EfficientNetV2-S)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),  # set CORS_ORIGINS in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model registry - lazy-loaded once at startup, missing files are tolerated
# ---------------------------------------------------------------------------

_state: dict = {"crop": None, "disease": None}

LOW_CONFIDENCE = 0.70
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.on_event("startup")
def load_models():
    try:
        _state["crop"] = {
            "model": joblib.load(RECOMMENDATION_DIR / "crop_mlp_model.joblib"),
            "scaler": joblib.load(RECOMMENDATION_DIR / "crop_scaler.joblib"),
            "encoder": joblib.load(RECOMMENDATION_DIR / "crop_label_encoder.joblib"),
        }
        print("[startup] Crop recommendation model loaded.")
    except FileNotFoundError:
        print("[startup] Crop model artifacts not found - /predict/crop will 503 until trained.")

    try:
        from disease import load_disease_model
        _state["disease"] = load_disease_model()
        print(f"[startup] Disease model loaded ({len(_state['disease']['class_names'])} classes).")
    except FileNotFoundError:
        print("[startup] Disease checkpoint not found - /predict/disease will 503.")
    except ImportError as e:
        print(f"[startup] Disease model needs extra packages ({e}) - pip install torch timm opencv-python-headless")
    except Exception as e:  # corrupt checkpoint, key mismatch, etc.
        print(f"[startup] Disease model failed to load: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "crop_model_loaded": _state["crop"] is not None,
        "disease_model_loaded": _state["disease"] is not None,
    }


# ---------------------------------------------------------------------------
# Unit 2: Crop recommendation
# ---------------------------------------------------------------------------

@app.post("/predict/crop", response_model=CropPrediction)
def predict_crop(payload: SoilInput):
    if _state["crop"] is None:
        raise HTTPException(503, "Crop model not loaded. Run training/crop_recommendation_train.ipynb first.")

    bundle = _state["crop"]
    x = np.array([[payload.N, payload.P, payload.K, payload.temperature,
                    payload.humidity, payload.ph, payload.rainfall]])
    x_scaled = bundle["scaler"].transform(x)

    proba = bundle["model"].predict_proba(x_scaled)[0]
    classes = bundle["encoder"].classes_
    top_idx = np.argsort(proba)[::-1][:3]

    best_idx = top_idx[0]
    return CropPrediction(
        crop=classes[best_idx],
        confidence=float(proba[best_idx]),
        top_k=[{"crop": classes[i], "confidence": float(proba[i])} for i in top_idx],
    )


# ---------------------------------------------------------------------------
# Plant disease detection (image upload)
# ---------------------------------------------------------------------------

@app.post("/predict/disease", response_model=DiseasePrediction)
async def predict_disease(file: UploadFile = File(...)):
    if _state["disease"] is None:
        raise HTTPException(503, "Disease model not loaded. Check the server log and see backend/README.md.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (JPG/PNG).")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")

    from disease import predict
    try:
        result = predict(_state["disease"], data)
    except Exception:
        raise HTTPException(400, "Could not read that image. Try a JPG or PNG.")

    result["low_confidence"] = result["confidence"] < LOW_CONFIDENCE
    return DiseasePrediction(**result)


# ---------------------------------------------------------------------------
# Serve the browser UI from the same service (one deployable unit on the cloud)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
