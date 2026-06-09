# 🛣️ Road Damage Detector

An AI-powered road damage detection system fine-tuned on Sri Lankan road imagery.
Upload a photo and get instant damage detection with severity scores and GPS heatmaps.

## 🔴 Live Demo
👉 [Try it here](https://road-damage-detector-production.up.railway.app/docs)

## 🧠 What it detects
| Damage Type | Severity |
|---|---|
| Pothole | Critical |
| Alligator Crack | High |
| Rutting | High |
| Edge Cracking | Medium |
| Longitudinal Crack | Medium |
| Lateral Crack | Medium |

## 🛠️ Tech Stack
- **Model** — YOLOv8s fine-tuned on 4,915 labelled road images
- **Backend** — FastAPI + Uvicorn
- **Frontend** — Vanilla HTML/CSS/JS + Leaflet.js maps
- **Training** — Google Colab (free T4 GPU)
- **Data** — Roboflow (Road Damage Detection dataset)
- **Deployment** — Railway (free tier)

## 📊 Model Performance
- Training epochs: 50
- Best mAP50: 0.366
- Pothole detection accuracy: 67%
- Dataset: 3,604 train / 816 val / 495 test images

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/road-damage-detector
cd road-damage-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Then open `index.html` in your browser.

## 🗺️ Features
- Real-time damage detection from uploaded photos
- Bounding boxes with confidence scores
- Severity classification (Critical / High / Medium / Low)
- GPS-tagged damage logging
- Interactive Sri Lanka heatmap
- REST API with auto-generated docs at `/docs`

## 👩‍💻 Author
Rashini De Silva — [GitHub](https://github.com/IT22101938)