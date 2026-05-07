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

CLASSES              = ["WithMask", "WithoutMask"]
CONFIDENCE_THRESHOLD = 0.85

# ── AWS config — all driven by environment variables ──────────────
USE_AWS      = os.getenv("USE_AWS", "false").lower() == "true"
AWS_REGION   = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET    = os.getenv("S3_BUCKET", "face-mask-audit-trail-om-44")
DYNAMO_TABLE = os.getenv("DYNAMO_TABLE", "mask-detections")
CW_NAMESPACE = os.getenv("CW_NAMESPACE", "FaceMaskDetection")

# ── DNN face detector paths ───────────────────────────────────────
PROTOTXT_PATH             = os.path.join(BASE_DIR, "deploy.prototxt")
CAFFEMODEL_PATH           = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
FACE_CONFIDENCE_THRESHOLD = 0.5

# ──────────────────────────────────────────────
# AWS clients
# Credentials come from the IAM Instance Role — no keys needed.
# ──────────────────────────────────────────────
def get_aws_clients():
    session = boto3.Session(region_name=AWS_REGION)
    return {
        "s3":              session.client("s3"),
        "dynamodb_client": session.client("dynamodb"),   # low-level client — explicit types
        "cloudwatch":      session.client("cloudwatch"),
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


def log_to_dynamodb(item: dict):
    """Write inference metadata to DynamoDB using explicit type descriptors."""
    if not USE_AWS:
        print("[DEBUG] AWS disabled — skipping DynamoDB write.")
        return

    # Explicit Python-type casts before building the DynamoDB item.
    # This guarantees no silent type mismatch regardless of what
    # arrived in the dict (e.g. numpy float32 instead of Python float).
    detection_id  = str(item["detection_id"])
    timestamp     = str(item["timestamp"])
    cls           = str(item["class"])
    status        = str(item["status"])
    confidence    = float(item["confidence"])      # numpy float32 → Python float
    action        = str(item["action"])
    review_status = str(item["review_status"])
    needs_review  = bool(item["needs_review"])     # explicit bool cast
    s3_key        = str(item["s3_key"]) if item["s3_key"] else "local"
    latency_ms    = float(item["latency_ms"])
    device        = str(item["device"])

    dynamo_item = {
        "detection_id":  {"S": detection_id},
        "timestamp":     {"S": timestamp},
        "class":         {"S": cls},
        "status":        {"S": status},
        "confidence":    {"N": str(confidence)},   # N type must be a string
        "action":        {"S": action},
        "review_status": {"S": review_status},
        "needs_review":  {"BOOL": needs_review},   # native DynamoDB boolean
        "s3_key":        {"S": s3_key},
        "latency_ms":    {"N": str(latency_ms)},
        "device":        {"S": device},
    }

    print(f"[DEBUG] Writing to DynamoDB: {dynamo_item}")

    try:
        aws["dynamodb_client"].put_item(
            TableName=DYNAMO_TABLE,
            Item=dynamo_item,
        )
        print(f"[DEBUG] DynamoDB write succeeded for detection_id={detection_id}")
    except Exception as exc:
        print(f"[ERROR] DynamoDB put_item failed: {exc}")
        raise


def push_cloudwatch_metrics(predicted_class: str, latency_ms: float):
    """Push latency and deny-entry count to CloudWatch. No-op locally."""
    if not USE_AWS:
        return
    aws["cloudwatch"].put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[
            {
                "MetricName": "InferenceLatencyMs",
                "Value":      latency_ms,
                "Unit":       "Milliseconds",
            },
            {
                "MetricName": "DenyEntryCount",
                "Value":      1 if predicted_class == "WithoutMask" else 0,
                "Unit":       "Count",
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

    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    ml_model = load_model(MODEL_PATH)

    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
        raise RuntimeError(
            f"DNN face detector files not found.\n"
            f"  prototxt:   {PROTOTXT_PATH}\n"
            f"  caffemodel: {CAFFEMODEL_PATH}"
        )
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("DNN face detector loaded.")

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
        "status":        "ok",
        "model_loaded":  ml_model is not None,
        "face_detector": face_net is not None,
        "device":        str(DEVICE),
        "aws_mode":      USE_AWS,
        "classes":       CLASSES,
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

    img = detect_and_crop_face(img, face_net)

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ml_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    predicted_idx   = int(probs.argmax())
    predicted_class = str(CLASSES[predicted_idx])
    confidence      = float(probs[predicted_idx])   # force Python float — not numpy
    latency_ms      = float((time.perf_counter() - t_start) * 1000)
    timestamp       = datetime.now(timezone.utc).isoformat()

    all_probs = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}

    # ── Confidence-based routing ──────────────────────────────────
    # Read the module-level constant explicitly — no local shadowing possible
    threshold = float(CONFIDENCE_THRESHOLD)
    base_status = "mask_on" if predicted_class == "WithMask" else "mask_off"

    print(f"[DEBUG] predicted_class={predicted_class}, confidence={confidence:.4f}, threshold={threshold}")

    if confidence >= threshold:
        action        = "Allow entry"
        review_status = "automated"
        needs_review  = False
    else:
        action        = "Review Required"
        review_status = "pending_review"
        needs_review  = True

    # ── Build result dict BEFORE AWS calls ────────────────────────
    # s3_key starts as None and is updated in-place after the upload.
    # log_to_dynamodb is called AFTER save_frame_to_s3 so s3_key is
    # already in the dict when DynamoDB receives it.
    result = {
        "detection_id":      detection_id,
        "timestamp":         timestamp,
        "class":             predicted_class,
        "status":            base_status,
        "confidence":        round(confidence, 4),
        "action":            action,
        "review_status":     review_status,
        "needs_review":      needs_review,
        "all_probabilities": all_probs,
        "latency_ms":        round(latency_ms, 2),
        "s3_key":            None,
        "device":            str(DEVICE),
    }

    print(f"[DEBUG] Action={action}, Confidence={confidence:.4f}, needs_review={needs_review}")

    # ── AWS side-effects (sequential — s3_key must exist before DDB write)
    try:
        result["s3_key"] = save_frame_to_s3(raw, detection_id)  # updates dict in-place
        log_to_dynamodb(result)                                  # reads updated s3_key
        push_cloudwatch_metrics(predicted_class, latency_ms)
    except Exception as exc:
        print(f"[WARN] AWS logging failed: {exc}")

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)