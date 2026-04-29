"""
Face Mask Detection API — AWS-integrated, hybrid local/cloud mode.
Set USE_AWS=true in environment to enable S3, DynamoDB, CloudWatch.
"""

import io
import os
import time
import uuid
import boto3
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms, models
from contextlib import asynccontextmanager
from datetime import datetime, timezone

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

# ── AWS config — all driven by environment variables ──────────────
USE_AWS      = os.getenv("USE_AWS", "false").lower() == "true"
AWS_REGION   = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET    = os.getenv("S3_BUCKET", "face-mask-audit-trail-om-44")
DYNAMO_TABLE = os.getenv("DYNAMO_TABLE", "mask-detections")
CW_NAMESPACE = os.getenv("CW_NAMESPACE", "FaceMaskDetection")

# ── DNN face detector paths ───────────────────────────────────────
PROTOTXT_PATH   = os.path.join(BASE_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
FACE_CONFIDENCE_THRESHOLD = 0.5

# ──────────────────────────────────────────────
# AWS clients (lazy — only created when USE_AWS=true)
# Credentials come from the IAM Instance Role — no keys needed.
# ──────────────────────────────────────────────
def get_aws_clients():
    session = boto3.Session(region_name=AWS_REGION)
    return {
        "s3":        session.client("s3"),
        "dynamodb":  session.resource("dynamodb"),
        "cloudwatch": session.client("cloudwatch"),
    }

aws = {}

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
# Pre-processing
# ──────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ──────────────────────────────────────────────
# Face detection
# ──────────────────────────────────────────────
def detect_and_crop_face(img: Image.Image, face_net: cv2.dnn.Net) -> Image.Image:
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > FACE_CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return img

# ──────────────────────────────────────────────
# AWS integrations
# ──────────────────────────────────────────────
def save_frame_to_s3(raw_bytes: bytes, detection_id: str) -> str | None:
    """Upload the raw image to S3. Returns the S3 key or None if disabled."""
    if not USE_AWS:
        return None
    key = f"frames/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{detection_id}.jpg"
    aws["s3"].put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=raw_bytes,
        ContentType="image/jpeg",
    )
    return key


def log_to_dynamodb(detection_id: str, predicted_class: str,
                    confidence: float, s3_key: str | None, latency_ms: float):
    """Write inference metadata to DynamoDB. No-op locally."""
    if not USE_AWS:
        return
    table = aws["dynamodb"].Table(DYNAMO_TABLE)
    table.put_item(Item={
        "detection_id": detection_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "class":        predicted_class,
        "confidence":   str(round(confidence, 4)),   # Decimal not JSON-safe
        "action":       ACTION_MAP[predicted_class]["action"],
        "s3_key":       s3_key or "local",
        "latency_ms":   str(round(latency_ms, 2)),
        "device":       str(DEVICE),
    })


def push_cloudwatch_metrics(predicted_class: str, latency_ms: float):
    """Push latency and deny-entry count to CloudWatch. No-op locally."""
    if not USE_AWS:
        return
    aws["cloudwatch"].put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[
            {
                "MetricName": "InferenceLatencyMs",
                "Value": latency_ms,
                "Unit": "Milliseconds",
            },
            {
                "MetricName": "DenyEntryCount",
                "Value": 1 if predicted_class == "WithoutMask" else 0,
                "Unit": "Count",
            },
        ],
    )

# ──────────────────────────────────────────────
# App lifespan
# ──────────────────────────────────────────────
ml_model: nn.Module | None = None
face_net: cv2.dnn.Net | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, face_net, aws

    # Load mask classifier
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    ml_model = load_model(MODEL_PATH)

    # Load DNN face detector
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
        raise RuntimeError(
            f"DNN face detector files not found.\n"
            f"  prototxt:   {PROTOTXT_PATH}\n"
            f"  caffemodel: {CAFFEMODEL_PATH}"
        )
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("DNN face detector loaded.")

    # AWS clients (only when enabled)
    if USE_AWS:
        aws = get_aws_clients()
        print(f"AWS mode ON — region={AWS_REGION}, bucket={S3_BUCKET}, table={DYNAMO_TABLE}")
    else:
        print("AWS mode OFF — running in local mode.")

    yield


app = FastAPI(
    title="Face Mask Detection API",
    description="Mask detection with optional S3/DynamoDB/CloudWatch integration.",
    version="2.0.0",
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
        "status":         "ok",
        "model_loaded":   ml_model is not None,
        "face_detector":  face_net is not None,
        "device":         str(DEVICE),
        "aws_mode":       USE_AWS,
        "classes":        CLASSES,
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Face image (JPEG/PNG)")):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPEG or PNG.",
        )

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    detection_id = str(uuid.uuid4())
    t_start      = time.perf_counter()

    # Face detection & crop
    img = detect_and_crop_face(img, face_net)

    # Inference
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ml_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    predicted_idx   = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    confidence      = float(probs[predicted_idx])
    latency_ms      = (time.perf_counter() - t_start) * 1000

    all_probs = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
    business  = ACTION_MAP[predicted_class]

    # ── AWS side-effects (non-blocking best-effort) ───────────────
    s3_key = None
    try:
        s3_key = save_frame_to_s3(raw, detection_id)
        log_to_dynamodb(detection_id, predicted_class, confidence, s3_key, latency_ms)
        push_cloudwatch_metrics(predicted_class, latency_ms)
    except Exception as exc:
        # Never fail a prediction because of observability errors
        print(f"[WARN] AWS logging failed: {exc}")

    return JSONResponse({
        "detection_id":    detection_id,
        "status":          business["status"],
        "class":           predicted_class,
        "confidence":      round(confidence, 4),
        "action":          business["action"],
        "all_probabilities": all_probs,
        "latency_ms":      round(latency_ms, 2),
        "s3_key":          s3_key,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)