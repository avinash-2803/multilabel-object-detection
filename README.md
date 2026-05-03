# Multi-Label Object Detection in Remote Sensing Imagery
### A Hybrid Deep Learning Approach Using DenseNet169 + GCN and YOLOv8

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Django](https://img.shields.io/badge/Django-4.2-092E20?style=for-the-badge&logo=django&logoColor=white)](https://djangoproject.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-8A2BE2?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>


</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Datasets](#4-datasets)
5. [Model Design](#5-model-design)
6. [Implementation](#6-implementation)
7. [Web Application](#7-web-application)
8. [Results and Analysis](#8-results-and-analysis)
9. [Project Structure](#9-project-structure)
10. [Installation and Setup](#10-installation-and-setup)
11. [Usage Guide](#11-usage-guide)
12. [Technologies Used](#12-technologies-used)
13. [Limitations](#13-limitations)
14. [Future Work](#14-future-work)
15. [Acknowledgements](#16-acknowledgements)
16. [References](#17-references)

---

## 1. Project Overview

This project presents a **hybrid deep learning pipeline** for simultaneous scene-level classification and instance-level object detection in high-resolution aerial and satellite imagery. Remote sensing images present unique challenges — objects appear at arbitrary orientations and scales, multiple semantic categories co-exist within a single frame, and spatial context is as informative as local appearance.

The system integrates two independently trained deep learning models into a unified Django-based web application:

| Component | Model | Dataset | Task |
|---|---|---|---|
| Scene Classifier | DenseNet169 + GCN | MLRSNet (46 classes) | Multi-label scene prediction |
| Object Detector | YOLOv8 + SAHI | DOTA (15 classes) | Bounding box detection + counting |

**Both models run on a single uploaded image and produce a combined structured JSON output alongside a visually annotated result image.**

---

## 2. Motivation and Problem Statement

### 2.1 Why Remote Sensing?

Satellite and aerial imagery is generated continuously by orbiting platforms, unmanned aerial vehicles (UAVs), and airborne sensors — producing petabytes of visual data that cannot be manually annotated or interpreted at scale. Automated scene understanding pipelines are essential for:

- **Disaster response coordination** — identifying damaged infrastructure and blocked roads
- **Urban planning** — monitoring land-use change and construction activity
- **Precision agriculture** — detecting crop types, irrigation, and field boundaries
- **Defence and surveillance** — tracking vehicles, vessels, and aircraft
- **Climate monitoring** — observing deforestation, glacier retreat, and flooding

### 2.2 Why Multi-Label Classification?

A single aerial image rarely belongs to only one semantic category. A coastal scene may simultaneously contain a port, residential buildings, a freeway interchange, storage tanks, and an airport — all within one frame. Standard single-label classifiers fail this scenario by design. Multi-label classification assigns all applicable labels simultaneously, providing richer scene understanding.

### 2.3 Research Gap Addressed

Existing multi-label classification systems typically treat labels as independent binary predictions, ignoring the statistical co-occurrence patterns between categories. For example, *airport* and *airplane* nearly always co-occur, while *beach* and *storage_tank* almost never do. This project addresses this gap by integrating a **Graph Convolutional Network (GCN)** that explicitly encodes label relationship structure from training data, improving classification accuracy over independent baseline classifiers.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Image                            │
│                  (Aerial / Satellite)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  DenseNet169     │       │   YOLOv8 + SAHI      │
│  Backbone        │       │   Object Detector     │
│  (Feature Ext.)  │       │   (Tiled Inference)  │
└────────┬─────────┘       └──────────┬───────────┘
         │                            │
         ▼                            ▼
┌──────────────────┐       ┌──────────────────────┐
│  Graph Conv.     │       │  Box Merging +        │
│  Network (GCN)   │       │  Class Filtering      │
│  Label Graph     │       │  (IoU-based NMS)      │
└────────┬─────────┘       └──────────┬───────────┘
         │                            │
         ▼                            ▼
┌──────────────────┐       ┌──────────────────────┐
│  Scene Labels    │       │  Bounding Boxes       │
│  (Multi-label)   │       │  + Pixel Counts       │
│  e.g. airport,  │       │  e.g. plane×9,        │
│  airplane,       │       │  storage-tank×20,     │
│  harbor&port     │       │  ship×5               │
└────────┬─────────┘       └──────────┬───────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
         ┌──────────────────┐
         │   pipeline.py    │
         │   (Merger)       │
         └──────────┬───────┘
                    │
         ┌──────────┴───────────┐
         │                      │
         ▼                      ▼
  Annotated Image          JSON Output
  (Bounding Boxes)    (Structured Result)
```

### 3.2 Data Flow

1. User uploads an aerial or satellite image through the Django web interface
2. `pipeline.py` receives the PIL Image and calls both model modules in sequence
3. `scene_predictor.py` runs DenseNet169+GCN inference and returns a ranked list of scene labels
4. `object_detector.py` runs YOLOv8 with SAHI sliced inference and returns bounding boxes with pixel areas
5. `pipeline.py` merges both outputs into a structured dictionary
6. `views.py` saves the annotated image to disk and renders the results page with JSON output

---

## 4. Datasets

###  DOTA — Object Detection Dataset

DOTA (Dataset for Object deTection in Aerial Images) is the largest and most widely cited benchmark for aerial object detection.

| Property | Value |
|---|---|
| **Total images** | 2,806 |
| **Total annotated instances** | 188,282 |
| **Object categories** | 15 |
| **Image resolution** | 800×800 to 4,000×4,000 pixels |
| **Annotation type** | Oriented bounding boxes (polygon) |
| **Image source** | Google Earth, GF-2, JL-1 satellites |

**15 DOTA object categories:**

`plane` · `ship` · `storage-tank` · `baseball-diamond` · `tennis-court` · `basketball-court` · `ground-track-field` · `harbor` · `bridge` · `large-vehicle` · `small-vehicle` · `helicopter` · `roundabout` · `soccer-ball-field` · `swimming-pool`

---

## 5. Model Design

### 5.1 Scene Classifier — MLCGCN_DN169

The scene classification model is a custom hybrid architecture combining a DenseNet169 convolutional backbone with a multi-level Graph Convolutional Network for label dependency modelling.

#### 5.1.1 Architecture Modules

**Module 1 — DenseNet169 Backbone**

DenseNet169 uses dense connectivity — each layer receives feature maps from all preceding layers via concatenation. This promotes feature reuse, mitigates vanishing gradients, and produces both fine-grained and high-level representations in a parameter-efficient manner. The backbone produces three feature taps at different spatial scales (F2, F3, F4) which are consumed by downstream modules.

**Module 2 — Context Information Module (cp2, cp3, cp4)**

Each feature tap passes through a Context Information Module consisting of:
- Four parallel dilated convolutional branches (dilation rates 1, 2, 3, 4) capturing multi-scale contextual information
- Convolutional position encoding on each branch output
- GroupNorm fusion with skip connection
- Masked self-attention for long-range spatial dependency modelling
- Final BatchNorm + LeakyReLU refinement

**Module 3 — Category Feature Extractor (cfm2, cfm3, cfm4)**

For each feature level, per-class attention maps are generated by a 1×1 convolution, producing soft spatial masks that highlight image regions relevant to each label. Attended feature vectors are pooled per class, yielding a (D × C) category feature matrix at each scale.

**Module 4 — Category Fusion Module**

Cross-scale category features are fused using a cross-attention mechanism where fine-scale (F2) features query medium-scale (F3) features, and the result is combined with coarse-scale (F4) features. This captures label-relevant context at multiple resolutions.

**Module 5 — Label Semantic Mining (GCN)**

A two-layer Graph Convolutional Network operates on a label co-occurrence graph derived from training set statistics. Node features are learnable label embeddings (300-dimensional). GCN message passing propagates semantic relationships between labels, producing context-aware classifier weight vectors of dimension 1664.

**Module 6 — Dual Graph Network (DGN)**

A dynamic graph refinement module that computes image-adaptive adjacency between label nodes based on current input features. Combines static GCN knowledge with input-specific label relationships for final score computation.

**Module 7 — Dual Classifier Head**

Final scores are computed via two parallel paths:
- `cls_r`: Conv1d classifier applied to GCN output
- `cls_m`: Global average pool + linear layer applied to F4 features

Final logit = 0.5 × (cls_r + cls_m)

#### 5.1.2 Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | DenseNet169 (ImageNet pretrained) |
| Embed dimension | 300 |
| GCN hidden dimension | 1024 |
| Feature dimension | 1664 |
| Loss function | Binary Cross-Entropy with Logits |
| Optimiser | Adam |
| Backbone LR | 1×10⁻⁵ |
| Head LR | 1×10⁻⁴ |
| Training epochs | 28 |
| Best validation mAP | **96.61%** |
| Platform | Google Colab (T4 GPU) |

### 5.2 Object Detector — YOLOv8 + SAHI

#### 5.2.1 YOLOv8 Architecture

YOLOv8 is a single-stage anchor-free object detector from Ultralytics. Key design choices:

- **CSPDarknet backbone** with C2f modules for multi-scale feature extraction
- **Feature Pyramid Network (FPN)** for handling objects at different scales
- **Decoupled detection head** — separate branches for classification and regression
- **Anchor-free design** — no pre-defined anchor boxes, simplifying training and improving generalisation

#### 5.2.2 SAHI — Sliced Aided Hyper Inference

Large aerial images contain many small objects (vehicles, small aircraft) that fall below the effective detection resolution of a standard single-pass inference. SAHI addresses this by:

1. Tiling the input image into overlapping chips (1024×1024 pixels with 0.2 overlap ratio)
2. Running YOLOv8 inference independently on each chip
3. Mapping detections back to original image coordinates
4. Merging duplicate detections using IoU-based Non-Maximum Suppression

This significantly improves recall for small objects without retraining.

#### 5.2.3 Post-Processing

A custom per-class confidence filtering step removes contextually implausible detections:

| Class | Minimum Confidence | Rationale |
|---|---|---|
| ship | 0.55 | Aircraft fuselages resemble ships from above |
| baseball-diamond | 0.65 | Taxiway markings resemble diamonds |
| tennis-court | 0.60 | Blue rooftops resemble courts |
| basketball-court | 0.60 | Rectangular rooftops trigger false positives |
| swimming-pool | 0.55 | Circular tank tops trigger false positives |
| plane, vehicle, harbor | 0.20 (base) | Detect freely — low false-positive rate |

#### 5.2.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | YOLOv8s (COCO pretrained) |
| Fine-tuned on | DOTA dataset |
| Input resolution | 640×640 (training), 1024×1024 (SAHI tiles) |
| Confidence threshold | 0.20 |
| IoU threshold (NMS) | 0.40 |
| Platform | Google Colab (T4 GPU) |

---

## 6. Implementation

### 6.1 Module Responsibilities

| File | Responsibility |
|---|---|
| `detector/model_src/mlcgcn.py` | Full MLCGCN_DN169 architecture definition |
| `detector/model_src/util.py` | Graph adjacency normalisation (gen_adj) |
| `detector/scene_predictor.py` | Load checkpoint, run inference, return scene labels |
| `detector/object_detector.py` | Load YOLOv8, run SAHI inference, draw bounding boxes |
| `detector/pipeline.py` | Call both models, aggregate results, return merged output |
| `detector/views.py` | Django HTTP handlers for upload and results |
| `detector/apps.py` | Pre-load both models at Django startup |
| `templates/index.html` | Upload interface |
| `templates/results.html` | Results display with annotated image and JSON |

### 6.2 Output JSON Format

```json
{
  "scene_labels": [
    "airport",
    "airplane",
    "harbor&port",
    "storage_tank",
    "roundabout"
  ],
  "objects": [
    {
      "label": "plane",
      "count": 9,
      "pixel_count": 41054,
      "bounding_boxes": [
        {
          "x1": 845.2,
          "y1": 612.7,
          "x2": 1023.4,
          "y2": 748.1,
          "confidence": 0.76,
          "seen_count": 2
        }
      ]
    },
    {
      "label": "storage-tank",
      "count": 20,
      "pixel_count": 115338,
      "bounding_boxes": ["..."]
    }
  ]
}
```

---

## 7. Web Application

The inference pipeline is served through a lightweight **Django 4.2** web application with two views:

- **Upload page** (`/`) — drag-and-drop image upload with file preview, supported formats: JPG, PNG, TIFF, WEBP up to 50 MB
- **Results page** (`/analyse/`) — split layout showing annotated image on the left and structured analysis panel on the right

Both models are loaded into memory once at server startup via `DetectorConfig.ready()` in `apps.py`, ensuring fast inference for all subsequent requests without reloading weights on every call. Static files are served by WhiteNoise.

---

## 8. Results and Analysis

### 8.1 Scene Classification Performance (MLRSNet Validation Set)

| Metric | Value |
|---|---|
| Mean Average Precision (mAP) | **96.61%** |
| Training epochs | 28 |
| Overall Precision (OP) | 98.28% |
| Overall Recall (OR) | 88.14% |
| Overall F1 (OF1) | 92.93% |
| Per-class Precision (CP) | 97.12% |
| Per-class Recall (CR) | 87.86% |
| Per-class F1 (CF1) | 92.26% |
| Best validation loss | 0.0137 |

### 8.2 Object Detection Performance (DOTA Test Images)

| Class | Detection Accuracy | Notes |
|---|---|---|
| plane | ✅ Excellent (9/10 detected) | High confidence 0.70–0.90 |
| storage-tank | ✅ Excellent | Circular shape highly distinctive |
| ship | ✅ Good (real ships only) | Per-class filter removes false positives |
| tennis-court | ✅ Good | Blue surface reliably detected |
| large-vehicle | ⚠️ Over-detected | Container stacks misclassified |
| harbor | ⚠️ Fragmented | Large dock area split into multiple boxes |

### 8.3 Qualitative Results

**Test Image 1 — Airport Scene**
- Scene labels predicted: `airplane`, `airport`
- Objects detected: 9 planes, 8 small-vehicles, 3 large-vehicles, 2 harbor
- All aircraft at gates correctly localised with confidence 0.66–0.76

**Test Image 2 — Mixed Urban + Port + Airport**
- Scene labels predicted: `roundabout`, `harbor&port`, `storage_tank`
- Objects detected: 9 planes, 90 storage-tanks, 5 ships, 2 tennis-courts, 112 large-vehicles
- Ships at real port correctly kept (confidence > 0.55 filter passed)
- Ships at airport correctly removed (confidence < 0.55 filter rejected)

---

## 9. Project Structure

```
remotesense-detection/
│
├── manage.py                          # Django management entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore
│
├── model_assets/                      # ⚠ Weights not in repo — see Drive link
│   └── .gitkeep                       # Placeholder — put your .pth/.pt files here
│
├── detector/
│   ├── model_src/
│   │   ├── __init__.py
│   │   ├── mlcgcn.py                  # MLCGCN_DN169 full architecture
│   │   └── util.py                    # gen_adj graph normalisation
│   │
│   ├── templatetags/
│   │   ├── __init__.py
│   │   └── detector_tags.py           # Custom Django template filters
│   │
│   ├── __init__.py
│   ├── apps.py                        # Model preloading at startup
│   ├── views.py                       # HTTP request handlers
│   ├── urls.py                        # URL routing
│   ├── pipeline.py                    # Merges both model outputs
│   ├── scene_predictor.py             # DenseNet169+GCN inference
│   └── object_detector.py             # YOLOv8+SAHI inference
│
├── templates/
│   ├── index.html                     # Upload page
│   └── results.html                   # Results display page
│
├── remotesense_app/
│   ├── __init__.py
│   ├── settings.py                    # Django configuration
│   ├── urls.py                        # Root URL configuration
│   └── wsgi.py                        # WSGI entry point
│
├── static/                            # Static assets (CSS, JS)
└── media/                             # Auto-created — stores annotated images
```

---

## 10. Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip
- 8 GB RAM recommended (4 GB minimum)
- GPU optional — CPU inference supported out of the box

### Step 1 — Clone the repository

```bash
git clone https://github.com/Avinash-2803/remotesense-detection.git
cd remotesense-detection
```

### Step 2 — Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Replace the torch line in requirements.txt with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4 — Download model weights

Download the following files from the shared Google Drive and place them inside `model_assets/`:

| File | Description | Size |
|---|---|---|
| `MLCGCN_DN169_best.pth` | Scene classifier trained weights | ~332 MB |
| `categories.pkl` | MLRSNet class name list | < 1 MB |
| `best_thresh.npy` | Per-class prediction thresholds | < 1 MB |
| `adj_matrix.pkl` | Label co-occurrence adjacency matrix | < 1 MB |
| `yolov8_best.pt` | YOLOv8 trained weights | ~64 MB |

> 📁 **Model weights download:** [Google Drive — Add your link here]

After downloading, your `model_assets/` folder must look like this:

```
model_assets/
├── MLCGCN_DN169_best.pth
├── categories.pkl
├── best_thresh.npy
├── adj_matrix.pkl
└── yolov8_best.pt
```

### Step 5 — Run the development server

```bash
python manage.py runserver
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

> **Note:** The first request after starting the server takes 20–40 seconds as both models load into memory. Subsequent requests are significantly faster.

---

## 11. Usage Guide

1. Open `http://127.0.0.1:8000` in your browser
2. Click **Upload Image** or drag and drop any aerial/satellite image (JPG, PNG, TIFF, WEBP — max 50 MB)
3. Click **Analyse Image** and wait for both models to process the image
4. The results page displays:
   - **Annotated image** — bounding boxes drawn directly on the image with class labels and confidence scores
   - **Scene labels** — multi-label scene classification from DenseNet169 + GCN
   - **Object table** — per-class instance count and total pixel area
   - **JSON output** — complete structured result ready to copy

### Supported Image Types

| Image Type | Scene Model | Detection Model |
|---|---|---|
| Google Earth top-down (medium zoom) | Excellent | Excellent |
| High-resolution satellite (< 1 m GSD) | Good | Good |
| Drone nadir-view imagery | Good | Good |
| Oblique aerial photography | Limited | Limited |

---

## 12. Technologies Used

| Category | Technology | Version |
|---|---|---|
| Deep Learning | PyTorch | 2.0+ |
| Scene backbone | DenseNet169 (torchvision) | ImageNet pretrained |
| Object detection | Ultralytics YOLOv8 | 8.2.0 |
| Sliced inference | SAHI | 0.11+ |
| Web framework | Django | 4.2 |
| Static files | WhiteNoise | 6.6+ |
| Image processing | Pillow | 10.0+ |
| Visualisation | Matplotlib | 3.7+ |
| Numerical computing | NumPy | 1.24+ |
| Serialisation | Pickle, PyTorch checkpoint | — |
| Training platform | Google Colab (NVIDIA T4 GPU) | — |
| Version control | Git + GitHub | — |

---

## 13. Limitations

- **Scene classification threshold sensitivity** — per-class thresholds tuned on the validation set may require recalibration when applied to out-of-distribution images (e.g. very high altitude, unusual sensor types)
- **Vehicle over-detection at ports** — shipping container stacks viewed from above are visually similar to large vehicle formations in DOTA training data, leading to inflated large-vehicle counts at container ports
- **Harbor fragmentation** — large dock structures are split into multiple harbor detections rather than one unified bounding box
- **CPU inference latency** — without GPU, full pipeline inference takes 15–45 seconds per image depending on resolution
- **No segmentation masks** — the deployed YOLOv8 model produces axis-aligned bounding boxes; pixel-level segmentation masks require the YOLOv8-seg variant
- **Fixed input resolution** — scene classifier input is fixed at 224×224 pixels after centre-crop from 256; very fine-grained details in large images may be lost

---

## 14. Future Work

- **Oriented bounding boxes** — integrate YOLOv8-OBB for rotation-invariant detection matching the full DOTA annotation format, improving precision on obliquely oriented objects such as aircraft and ships
- **Semantic segmentation** — replace bounding-box pixel estimation with a U-Net or DeepLab segmentation head for true pixel-level object area statistics
- **Transformer-based scene model** — replace DenseNet169 backbone with a Vision Transformer (ViT) or Swin Transformer for better long-range spatial dependency modelling
- **Semi-supervised learning** — leverage the large volume of unlabelled public satellite imagery to improve model generalisation through pseudo-labelling or contrastive pre-training
- **Dynamic threshold calibration** — learn scene-context-aware confidence thresholds to automatically suppress false positives (e.g. ships at airports) without hard-coded per-class rules
- **Temporal change detection** — extend the pipeline to multi-date image pairs for detecting land-use change, construction activity, and disaster impact
- **Edge deployment** — quantise and prune both models for deployment on UAV onboard processors using TensorRT or ONNX Runtime
- **Google OAuth authentication** — add user login, per-user analysis history, and export functionality to the web application

---


## 15. Acknowledgements

- **DOTA** dataset provided by Xia et al., CVPR 2018. We thank the DOTA team for the comprehensive aerial object detection benchmark.
- **Ultralytics** for the YOLOv8 framework and pretrained weights.
- **SAHI** (Sliced Aided Hyper Inference) by Akyon et al. for the small object detection framework.
- **Google Colab** for providing free GPU compute resources used for model training.
- The **ML-GCN** paper by Chen et al. (CVPR 2019) for the foundational GCN-based multi-label classification approach.

---

## 16. References

```
[1]  G. Huang, Z. Liu, L. Van der Maaten, and K. Q. Weinberger,
     "Densely Connected Convolutional Networks,"
     in Proc. IEEE CVPR, 2017, pp. 4700–4708.

[2]  T. N. Kipf and M. Welling,
     "Semi-Supervised Classification with Graph Convolutional Networks,"
     in Proc. ICLR, 2017.

[3]  Z. Chen, X. Wei, P. Wang, and Y. Guo,
     "Multi-Label Image Recognition with Graph Convolutional Networks,"
     in Proc. IEEE CVPR, 2019, pp. 5177–5186.

[4]  G.-S. Xia et al.,
     "DOTA: A Large-Scale Dataset for Object Detection in Aerial Images,"
     in Proc. IEEE CVPR, 2018, pp. 3974–3983.

[5]  X. Qi, L. Zhu, Y. Wang, L. Zhang, J. Qian, M. Le, et al.,
     "MLRSNet: A Multi-Label High Spatial Resolution Remote Sensing Dataset
     for Semantic Image Interpretation,"
     ISPRS Journal of Photogrammetry and Remote Sensing,
     vol. 172, pp. 337–350, 2021.

[6]  Ultralytics,
     "YOLOv8: A New State-of-the-Art Computer Vision Model,"
     GitHub, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[7]  F. C. Akyon, S. O. Altinuc, and A. Temizel,
     "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection,"
     in Proc. IEEE ICIP, 2022, pp. 966–970.

[8]  A. Paszke et al.,
     "PyTorch: An Imperative Style, High-Performance Deep Learning Library,"
     in Proc. NeurIPS, 2019, pp. 8024–8035.

[9]  A. Dosovitskiy et al.,
     "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,"
     in Proc. ICLR, 2021.

[10] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko,
     "End-to-End Object Detection with Transformers (DETR),"
     in Proc. ECCV, 2020, pp. 213–229.
```

---

<div align="center">

**Built with PyTorch · Django · YOLOv8 · Google Colab**

*This project was developed as part of an academic curriculum in Computer Vision and Deep Learning.*

⭐ If this project was helpful, please consider starring the repository.

</div>
