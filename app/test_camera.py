import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)

model.load_state_dict(
    torch.load("models/best_model_AUG.pth", map_location=device)
)

model = model.to(device)
model.eval()

# Classes
classes = ["With Mask", "Without Mask"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# DNN Face Detector
base_dir = os.path.dirname(__file__)

prototxt_path = os.path.join(base_dir, "deploy.prototxt")
model_path = os.path.join(base_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Camera
cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare input for DNN
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):

        confidence_face = detections[0, 0, i, 2]

        if confidence_face > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clamp values
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Prediction
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            label = classes[pred.item()]
            conf = confidence.item() * 100

            # Drawing
            color = (0, 200, 0) if label == "With Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            text = f"{label}: {conf:.1f}%"

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )

            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Mask Detection (DNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()