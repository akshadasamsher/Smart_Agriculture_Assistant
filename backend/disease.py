"""
Plant disease detection (EfficientNetV2-S, 29 PlantVillage classes).

Loads the checkpoint saved by training/disease_detection_train.ipynb:
    {'model_state': ..., 'class_names': [...], 'epoch': ..., 'val_acc': ...}

torch/timm are imported lazily so the crop-recommendation API still runs
on machines that don't have them installed.

Preprocessing matches the notebook's val/test pipeline:
    Resize(224, 224) (bilinear) -> Normalize(ImageNet mean/std)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_NAME = "tf_efficientnetv2_s"

_HERE = Path(__file__).parent
_CANDIDATES = [
    _HERE / "models" / "detection" / "disease_detection_model.pth",
    _HERE / "models" / "disease_detection_model.pth",
    _HERE.parent / "disease_detection_model.pth",
]


def find_checkpoint() -> Path | None:
    env = os.environ.get("DISEASE_MODEL_PATH")
    if env:
        return Path(env)
    for p in _CANDIDATES:
        if p.exists():
            return p
    return None


def _build_net(num_classes: int):
    import torch.nn as nn
    import timm

    class PlantDiseaseNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.backbone = timm.create_model(
                MODEL_NAME, pretrained=False, num_classes=0, global_pool="avg"
            )
            f = self.backbone.num_features
            self.head = nn.Sequential(
                nn.BatchNorm1d(f), nn.Dropout(0.3),
                nn.Linear(f, 512), nn.GELU(),
                nn.BatchNorm1d(512), nn.Dropout(0.15),
                nn.Linear(512, n),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    return PlantDiseaseNet(num_classes)


def load_disease_model(path: Path | None = None) -> dict:
    """Returns {'model', 'class_names', 'device'}. Raises FileNotFoundError / ImportError."""
    import torch

    path = path or find_checkpoint()
    if path is None or not Path(path).exists():
        raise FileNotFoundError("Disease model checkpoint not found")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older/odd checkpoints (e.g. with non-tensor objects) -- this is your own file.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    class_names = list(ckpt["class_names"])
    model = _build_net(len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"model": model, "class_names": class_names, "device": device}


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        import cv2  # same interpolation as the albumentations pipeline used in training
        arr = cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        arr = np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)  # CHW


def split_label(label: str) -> tuple[str, str]:
    """'Tomato - Late Blight' -> ('Tomato', 'Late Blight')"""
    plant, _, condition = label.partition(" - ")
    return plant.strip(), (condition.strip() or "Unknown")


def predict(bundle: dict, image_bytes: bytes, top_k: int = 5) -> dict:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(_preprocess(image_bytes)).unsqueeze(0).to(bundle["device"])
    with torch.no_grad():
        probs = F.softmax(bundle["model"](x), dim=1)[0].cpu().numpy()

    names = bundle["class_names"]
    idx = probs.argsort()[::-1][:top_k]
    plant, condition = split_label(names[idx[0]])
    return {
        "label": names[idx[0]],
        "plant": plant,
        "condition": condition,
        "healthy": condition.lower() == "healthy",
        "confidence": float(probs[idx[0]]),
        "top_k": [{"label": names[i], "confidence": float(probs[i])} for i in idx],
    }
