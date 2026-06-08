from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

# ── App setup ──────────────────────────────────────────────
app = FastAPI(title="Road Damage Detector API")

# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load your trained model ────────────────────────────────
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

# Damage classes from your dataset
CLASS_NAMES = {
    0: "Alligator Crack",
    1: "Edge Cracking",
    2: "Lateral Crack",
    3: "Longitudinal Crack",
    4: "Ravelling",
    5: "Rutting",
    6: "Striping",
    7: "Pothole"
}

# Severity score per class (how bad is each damage type)
SEVERITY = {
    "Pothole":            {"score": 9, "level": "Critical"},
    "Alligator Crack":    {"score": 8, "level": "High"},
    "Rutting":            {"score": 7, "level": "High"},
    "Edge Cracking":      {"score": 6, "level": "Medium"},
    "Longitudinal Crack": {"score": 5, "level": "Medium"},
    "Lateral Crack":      {"score": 4, "level": "Medium"},
    "Ravelling":          {"score": 3, "level": "Low"},
    "Striping":           {"score": 2, "level": "Low"},
}

# Storage for GPS-tagged detections (heatmap data)
detections_log = []

# ── Helper: draw boxes on image ────────────────────────────
def draw_boxes(image_array, results):
    """Draw coloured bounding boxes on the image."""
    color_map = {
        "Critical": (0, 0, 255),    # Red
        "High":     (0, 128, 255),  # Orange
        "Medium":   (0, 255, 255),  # Yellow
        "Low":      (0, 255, 0),    # Green
    }

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        severity = SEVERITY.get(class_name, {"score": 0, "level": "Low"})
        color = color_map[severity["level"]]

        # Draw rectangle
        cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{class_name} {confidence:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_array, (x1, y1 - h - 8), (x1 + w, y1), color, -1)

        # Draw label text
        cv2.putText(image_array, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image_array

# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Road Damage Detector API is running!"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    latitude: float = None,
    longitude: float = None
):
    """
    Upload a road image → get back:
    - annotated image with bounding boxes
    - list of detected damages with severity scores
    - overall road condition score
    """
    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Run detection
    results = model(image_bgr, conf=0.25)

    # Build detections list
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        severity = SEVERITY.get(class_name, {"score": 0, "level": "Low"})
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class":      class_name,
            "confidence": round(confidence, 3),
            "severity":   severity["level"],
            "score":      severity["score"],
            "bbox":       [x1, y1, x2, y2]
        })

    # Overall road condition (average severity score)
    if detections:
        avg_score = sum(d["score"] for d in detections) / len(detections)
        if avg_score >= 7:
            condition = "Critical — Immediate repair needed"
        elif avg_score >= 5:
            condition = "Poor — Schedule repair soon"
        elif avg_score >= 3:
            condition = "Fair — Monitor regularly"
        else:
            condition = "Good — Minor issues only"
    else:
        avg_score = 0
        condition = "No damage detected"

    # Draw boxes on image
    annotated = draw_boxes(image_bgr.copy(), results)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Convert annotated image to base64 so frontend can display it
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Log GPS data if provided
    if latitude and longitude:
        detections_log.append({
            "lat":        latitude,
            "lng":        longitude,
            "score":      avg_score,
            "condition":  condition,
            "detections": detections
        })

    return JSONResponse({
        "detections":      detections,
        "total_damage":    len(detections),
        "avg_score":       round(avg_score, 2),
        "condition":       condition,
        "annotated_image": img_base64
    })


@app.get("/heatmap-data")
def heatmap_data():
    """Returns all GPS-tagged detections for the heatmap."""
    return JSONResponse({"points": detections_log})


@app.get("/stats")
def stats():
    """Quick summary of all detections so far."""
    if not detections_log:
        return {"message": "No detections yet"}

    all_detections = []
    for log in detections_log:
        all_detections.extend(log["detections"])

    class_counts = {}
    for d in all_detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    return {
        "total_images_analyzed": len(detections_log),
        "total_damages_found":   len(all_detections),
        "damage_breakdown":      class_counts
    }


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)