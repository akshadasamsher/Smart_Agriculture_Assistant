# Smart Agriculture Assistant - Backend

FastAPI service for crop recommendation and leaf disease detection.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check whether the crop and disease models are loaded |
| `/predict/crop` | POST | ANN/MLP crop recommendation from soil/climate JSON |
| `/predict/disease` | POST | Leaf image upload (multipart `file`) -> disease class (EfficientNetV2-S, 29 classes) |
| `/docs` | GET | Auto-generated Swagger UI |
| `/` | GET | Browser UI (served from `../frontend`) |

## Setup and run

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

## Train the crop model (only if you need to regenerate the artifacts)

Run all cells of `training/crop_recommendation_train.ipynb`. It writes, inside
`backend/models/recommendation/`: `crop_mlp_model.joblib`, `crop_scaler.joblib`,
`crop_label_encoder.joblib`.

## Example requests

```bash
curl -X POST http://127.0.0.1:8000/predict/crop \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82,"ph":6.5,"rainfall":202.9}'

curl -X POST http://127.0.0.1:8000/predict/disease -F "file=@leaf.jpg"
```

## Disease model

`/predict/disease` loads `disease_detection_model.pth` from `backend/models/detection/`
(or `backend/models/`, or the project root); override with `DISEASE_MODEL_PATH`.
Needs `torch`, `timm`, `opencv-python-headless`, `python-multipart` (all in
requirements.txt). If they are missing, the rest of the API still runs.
