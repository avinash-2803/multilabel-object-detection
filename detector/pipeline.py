"""
pipeline.py
===========
Calls both models and merges into final JSON-ready structure.
All indexing uses plain Python ints and dicts — no numpy indexing.
"""

from collections import defaultdict
from PIL import Image

from detector import scene_predictor
from detector import object_detector


def run(pil_image: Image.Image) -> dict:
    """
    Run full pipeline on a PIL image.

    Returns
    -------
    dict with keys:
        scene_labels   : list[str]
        objects        : list[dict]
        annotated_np   : np.ndarray
        raw_detections : list[dict]
    """
    img = pil_image.convert("RGB")

    # ── 1. Scene prediction ───────────────────────────────────────────────────
    scene_labels = scene_predictor.predict(img)

    # Ensure scene_labels is plain Python list of plain strings
    scene_labels = [str(s) for s in scene_labels]

    # ── 2. Object detection ───────────────────────────────────────────────────
    annotated_np, raw_detections = object_detector.detect(img)

    # ── 3. Aggregate per class using only plain Python types ──────────────────
    groups = {}

    for det in raw_detections:
        # Force all values to plain Python types
        cls   = str(det["label"])
        conf  = float(det["confidence"])
        box   = [float(v) for v in det["box"]]
        px    = int(det["pixel_count"])
        seen  = int(det.get("seen_count", 1))

        if cls not in groups:
            groups[cls] = {
                "label":        cls,
                "count":        0,
                "pixel_count":  0,
                "bounding_boxes": [],
            }

        groups[cls]["count"]       += 1
        groups[cls]["pixel_count"] += px
        groups[cls]["bounding_boxes"].append({
            "x1":         round(box[0], 1),
            "y1":         round(box[1], 1),
            "x2":         round(box[2], 1),
            "y2":         round(box[3], 1),
            "confidence": round(conf, 4),
            "seen_count": seen,
        })

    # Sort by count descending
    objects = sorted(
        groups.values(),
        key=lambda x: x["count"],
        reverse=True,
    )

    # Convert to plain list
    objects = list(objects)

    return {
        "scene_labels":   scene_labels,
        "objects":        objects,
        "annotated_np":   annotated_np,
        "raw_detections": raw_detections,
    }