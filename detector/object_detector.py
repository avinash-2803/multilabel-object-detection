"""
object_detector.py
==================
YOLOv8 + SAHI inference module.
Fixes applied:
  1. confidence_threshold set to 0.20 — detects all planes including low conf
  2. Per-class high-confidence filter — removes false positives (ships etc.)
     but does NOT filter plane/vehicle classes so all real objects are detected
  3. postprocess_match_threshold set to 0.2 — less aggressive merging
     so individual planes are not collapsed into one detection
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "model_assets"

_yolo_model = None
_sahi_model = None

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

_CLASS_COLORS = {
    "plane":              "#E74C3C",
    "ship":               "#3498DB",
    "storage-tank":       "#2ECC71",
    "baseball-diamond":   "#F39C12",
    "tennis-court":       "#9B59B6",
    "basketball-court":   "#1ABC9C",
    "ground-track-field": "#E67E22",
    "harbor":             "#34495E",
    "bridge":             "#E91E63",
    "large-vehicle":      "#00BCD4",
    "small-vehicle":      "#FF5722",
    "helicopter":         "#8BC34A",
    "roundabout":         "#FF9800",
    "soccer-ball-field":  "#795548",
    "swimming-pool":      "#607D8B",
}

# Classes that need HIGHER confidence to avoid false positives.
# plane, large-vehicle, small-vehicle, harbor, bridge are NOT in this list
# so they use the base confidence threshold of 0.20 and get detected freely.
# ship/baseball-diamond etc. commonly appear as false positives at airports
# due to shape similarity — require higher confidence to be kept.
_HIGH_CONFIDENCE_CLASSES = {
    "ship":              0.55,
    "baseball-diamond":  0.65,
    "tennis-court":      0.60,
    "basketball-court":  0.60,
    "soccer-ball-field": 0.60,
    "swimming-pool":     0.55,
    "golf-course":       0.60,
    "roundabout":        0.50,
}


def _load():
    global _yolo_model, _sahi_model
    if _yolo_model is not None:
        return

    print("[ObjectDetector] Loading YOLOv8 ...")

    import torch
    from ultralytics import YOLO
    from sahi import AutoDetectionModel

    weights = str(ASSETS / "yolov8_best.pt")

    # Patch torch.load for PyTorch 2.6 compatibility
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    try:
        _yolo_model = YOLO(weights)
        _sahi_model = AutoDetectionModel.from_pretrained(
            model_type           = "ultralytics",
            model_path           = weights,
            confidence_threshold = 0.20,  # low base threshold — catches all planes
            device               = "cpu",
        )
    finally:
        torch.load = _original_load

    print("[ObjectDetector] Ready")


def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _merge_duplicate_boxes(predictions, iou_threshold=0.4):
    by_class = {}
    for pred in predictions:
        cls  = pred.category.name
        bbox = pred.bbox
        box  = [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
        conf = pred.score.value
        by_class.setdefault(cls, []).append({"conf": conf, "box": box})

    merged = []
    for cls_name, preds in by_class.items():
        preds = sorted(preds, key=lambda p: p["conf"], reverse=True)
        used  = [False] * len(preds)

        for i, anchor in enumerate(preds):
            if used[i]:
                continue
            used[i] = True
            group   = [anchor]

            for j in range(i + 1, len(preds)):
                if used[j]:
                    continue
                if _compute_iou(anchor["box"], preds[j]["box"]) >= iou_threshold:
                    group.append(preds[j])
                    used[j] = True

            avg_box = [
                sum(g["box"][k] for g in group) / len(group)
                for k in range(4)
            ]
            merged.append({
                "label":      cls_name,
                "confidence": round(group[0]["conf"], 4),
                "box":        avg_box,
                "seen_count": len(group),
            })

    return merged


def _apply_class_filters(detections: list) -> list:
    """
    Remove contextually wrong detections using per-class confidence thresholds.

    Logic:
    - plane, large-vehicle, small-vehicle, harbor, bridge have NO minimum
      so every detection of these classes above the base 0.20 is kept.
    - ship, baseball-diamond, tennis-court etc. require much higher confidence
      because they commonly appear as false positives at airports and
      industrial areas due to shape similarity with aircraft and buildings.

    Example: a ship detected at 0.23 confidence at an airport is almost
    certainly an aircraft fuselage. A ship at 0.75 confidence is likely real.
    """
    filtered = []
    removed  = []

    for det in detections:
        cls      = det["label"]
        conf     = det["confidence"]
        min_conf = _HIGH_CONFIDENCE_CLASSES.get(cls, 0.0)

        if conf >= min_conf:
            filtered.append(det)
        else:
            removed.append(f"{cls}({conf:.2f})")

    if removed:
        print(f"[ObjectDetector] Filtered out {len(removed)} false positives: "
              f"{', '.join(removed[:10])}"
              f"{'...' if len(removed) > 10 else ''}")

    return filtered


def _draw(pil_image: Image.Image, detections: list) -> np.ndarray:
    img_arr = np.array(pil_image.convert("RGB"))
    h, w    = img_arr.shape[:2]
    dpi     = 100
    fig_w   = max(8, w / dpi)
    fig_h   = max(6, h / dpi)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img_arr)
    ax.axis("off")

    for det in detections:
        cls          = det["label"]
        conf         = det["confidence"]
        x1,y1,x2,y2 = [int(v) for v in det["box"]]
        bw, bh       = x2 - x1, y2 - y1
        color        = _CLASS_COLORS.get(cls, "#FFFFFF")

        rect = patches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label_txt = f"{cls}: {conf:.2f}"
        ax.text(
            x1 + 2, y1 - 4, label_txt,
            fontsize   = max(5, min(9, int(bw / 12))),
            fontweight = "bold",
            color      = "white",
            va         = "bottom",
            bbox       = dict(facecolor=color, alpha=0.85, pad=1.5,
                              edgecolor="none", boxstyle="round,pad=0.2"),
        )

    n = len(detections)
    ax.set_title(
        f"{n} object{'s' if n != 1 else ''} detected",
        fontsize=11, fontweight="bold", pad=8
    )
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def detect(pil_image: Image.Image):
    _load()

    from sahi.predict import get_sliced_prediction

    img  = pil_image.convert("RGB")
    w, h = img.size

    tile_size = 1024 if max(w, h) > 2000 else 512

    sahi_result = get_sliced_prediction(
        img,
        _sahi_model,
        slice_height                = tile_size,
        slice_width                 = tile_size,
        overlap_height_ratio        = 0.2,
        overlap_width_ratio         = 0.2,
        postprocess_match_threshold = 0.2,   # less aggressive — keeps individual planes separate
        verbose                     = 0,
    )

    # Step 1 — merge overlapping boxes from SAHI tiles
    merged = _merge_duplicate_boxes(
        sahi_result.object_prediction_list,
        iou_threshold=0.4,
    )

    # Step 2 — remove contextually wrong low-confidence detections
    merged = _apply_class_filters(merged)

    # Step 3 — add pixel count (bounding box area in pixels)
    for det in merged:
        x1, y1, x2, y2 = det["box"]
        det["pixel_count"] = int((x2 - x1) * (y2 - y1))

    # Step 4 — draw annotated image with bounding boxes
    annotated_np = _draw(img, merged)

    return annotated_np, merged