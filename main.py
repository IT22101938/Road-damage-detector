from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

# ── App setup ──────────────────────────────────────────────
app = FastAPI(title="Road Damage Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

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

detections_log = []

# ── Helper: draw boxes using Pillow (no opencv needed) ─────
def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)

    color_map = {
        "Critical": "#FF0000",
        "High":     "#FF8000",
        "Medium":   "#FFFF00",
        "Low":      "#00FF00",
    }

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        severity   = SEVERITY.get(class_name, {"score": 0, "level": "Low"})
        color      = color_map[severity["level"]]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        label = f"{class_name} {confidence:.0%}"
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 7, y1], fill=color)

        # Draw label text
        draw.text((x1 + 2, y1 - 18), label, fill="black")

    return image

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
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)

    # Run detection
    results = model(image_array, conf=0.25)

    # Build detections list
    detections = []
    for box in results[0].boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        severity   = SEVERITY.get(class_name, {"score": 0, "level": "Low"})
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class":      class_name,
            "confidence": round(confidence, 3),
            "severity":   severity["level"],
            "score":      severity["score"],
            "bbox":       [x1, y1, x2, y2]
        })

    # Overall condition
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

    # Draw boxes using Pillow
    annotated = draw_boxes(image.copy(), results)

    # Convert to base64
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Log GPS
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
    return JSONResponse({"points": detections_log})


@app.get("/stats")
def stats():
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)