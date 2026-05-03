# Remote Sensing Detector — Django Web App

Multi-label scene classification (DenseNet169 + GCN) + object detection
(YOLOv8 + SAHI) on satellite and aerial imagery.

---

## Project Structure

```
remotesense_app/
├── manage.py
├── requirements.txt
├── model_assets/                 ← PUT YOUR 4 DOWNLOADED FILES HERE
│   ├── MLCGCN_DN169_best.pth
│   ├── categories.pkl
│   ├── best_thresh.npy
│   └── yolov8_best.pt
├── detector/
│   ├── model_src/
│   │   ├── mlcgcn.py             ← MLCGCN_DN169 architecture
│   │   └── util.py               ← gen_adj helper
│   ├── scene_predictor.py        ← GCN inference
│   ├── object_detector.py        ← YOLOv8 + SAHI inference
│   ├── pipeline.py               ← merges both models
│   └── views.py                  ← Django views
├── templates/
│   ├── index.html                ← upload page
│   └── results.html              ← results page
├── remotesense_app/
│   ├── settings.py
│   └── urls.py
└── media/                        ← auto-created, stores annotated images
```

---

## Setup Instructions

### Step 1 — Place your model files

Copy your 4 downloaded files into `model_assets/`:

```
model_assets/MLCGCN_DN169_best.pth
model_assets/categories.pkl
model_assets/best_thresh.npy
model_assets/yolov8_best.pt
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you have a GPU, install GPU PyTorch instead:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4 — Run the server

```bash
python manage.py runserver
```

### Step 5 — Open the app

Open your browser and go to:
```
http://127.0.0.1:8000
```

---

## How to Use

1. Upload any satellite or aerial image (JPG, PNG, TIFF)
2. Click **Analyse Image**
3. Wait 10–30 seconds (models load on first request)
4. Results page shows:
   - Annotated image with bounding boxes drawn by YOLOv8
   - Scene labels predicted by DenseNet169 + GCN
   - Per-class object counts and pixel areas
   - Full JSON output you can copy

---

## Notes

- First request after starting the server takes ~15–30 seconds to load models into memory
- Subsequent requests are much faster (models stay loaded)
- CPU inference is used by default — works without a GPU
- If you have a GPU, inference will be faster automatically
