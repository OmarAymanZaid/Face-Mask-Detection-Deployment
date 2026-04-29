"""
Face Mask Detection API
Model: MobileNetV2 (transfer learning)
Classes: WithMask, WithoutMask
Export: best_model_NO_AUG.pth
"""

import io
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms, models
from contextlib import asynccontextmanager


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "best_model_NO_AUG.pth"))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["WithMask", "WithoutMask"]

ACTION_MAP = {
    "WithMask":    {"status": "mask_on",  "action": "Allow entry"},
    "WithoutMask": {"status": "mask_off", "action": "Deny entry"},
}

# ── DNN Face Detector paths (files must be in the same folder as app.py) ──
PROTOTXT_PATH   = os.path.join(BASE_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
FACE_CONFIDENCE_THRESHOLD = 0.5


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────
def load_model(path: str) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Image pre-processing (matches training transforms)
# ──────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ──────────────────────────────────────────────
# Face detection helper (DNN-based SSD)
# ──────────────────────────────────────────────
def detect_and_crop_face(img: Image.Image, face_net: cv2.dnn.Net) -> Image.Image:
    """
    Run the DNN face detector on a PIL image.
    Returns the cropped face as a PIL image if a face is found,
    otherwise returns the original image unchanged.
    """
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence_face = detections[0, 0, i, 2]
        if confidence_face > FACE_CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Return the first valid detected face
            return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    # No face found — return original image for inference
    return img


# ──────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────
model: nn.Module | None = None
face_net: cv2.dnn.Net | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, face_net

    # Load mask classifier
    print(f"Checking for model at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Load DNN face detector
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
        raise RuntimeError(
            f"DNN face detector files not found.\n"
            f"  prototxt : {PROTOTXT_PATH}\n"
            f"  caffemodel: {CAFFEMODEL_PATH}"
        )
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("DNN face detector loaded.")

    yield


app = FastAPI(
    title="Face Mask Detection API",
    description=(
        "Upload a face photo and receive a mask-wearing prediction "
        "with confidence score and recommended entry action."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Face Mask Detection API is running. POST an image to /predict"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "face_detector": face_net is not None,
        "device":       str(DEVICE),
        "classes":      CLASSES,
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Face image (JPEG/PNG)")):
    """
    Predict whether the person in the image is wearing a face mask.

    Returns:
    - **status**: `mask_on` | `mask_off`
    - **class**: `WithMask` | `WithoutMask`
    - **confidence**: float 0–1
    - **action**: recommended entry decision
    - **all_probabilities**: per-class confidence breakdown
    """

    # ── Validate content type ──
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPEG or PNG.",
        )

    # ── Read & decode image ──
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ── Face detection & cropping (DNN) ──
    img = detect_and_crop_face(img, face_net)

    # ── Inference ──
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    predicted_idx   = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    confidence      = float(probs[predicted_idx])

    all_probs = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
    business  = ACTION_MAP[predicted_class]

    return JSONResponse({
        "status":            business["status"],
        "class":             predicted_class,
        "confidence":        round(confidence, 4),
        "action":            business["action"],
        "all_probabilities": all_probs,
    })


# ──────────────────────────────────────────────
# Run directly:  python app.py
# Or:            uvicorn app:app --reload --port 8000
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)