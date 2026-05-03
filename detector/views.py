"""
views.py
========
Two views:
  GET  /        → render upload form (index.html)
  POST /analyse → receive image, run pipeline, return results page
"""

import json
import uuid
import numpy as np
from pathlib import Path
from PIL import Image

from django.shortcuts import render
from django.conf import settings
from django.views.decorators.http import require_http_methods

from detector.pipeline import run as run_pipeline


RESULTS_DIR = Path(settings.MEDIA_ROOT) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@require_http_methods(["GET"])
def index(request):
    return render(request, "index.html")


@require_http_methods(["POST"])
def analyse(request):
    # ── Validate upload ───────────────────────────────────────────────────────
    if "image" not in request.FILES:
        return render(request, "index.html",
                      {"error": "Please select an image file."})

    upload = request.FILES["image"]
    if not upload.content_type.startswith("image/"):
        return render(request, "index.html",
                      {"error": "Uploaded file is not an image."})

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        pil_img = Image.open(upload).convert("RGB")
    except Exception as e:
        return render(request, "index.html",
                      {"error": f"Could not open image: {e}"})

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_pipeline(pil_img)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, "index.html",
                      {"error": f"Model inference failed: {e}"})
    # ── Save annotated image to media/results/ ────────────────────────────────
    fname = f"{uuid.uuid4().hex}.png"
    save_path = RESULTS_DIR / fname

    from PIL import Image as PILImage
    PILImage.fromarray(result["annotated_np"]).save(str(save_path))

    annotated_url = settings.MEDIA_URL + "results/" + fname

    # ── Build JSON for display ────────────────────────────────────────────────
    result_json = json.dumps({
        "scene_labels": result["scene_labels"],
        "objects":      result["objects"],
    }, indent=2)

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_objects  = sum(o["count"] for o in result["objects"])
    unique_classes = len(result["objects"])

    CLASS_COLORS = {
        "plane": "#E74C3C", "ship": "#3498DB", "storage-tank": "#2ECC71",
        "baseball-diamond": "#F39C12", "tennis-court": "#9B59B6",
        "basketball-court": "#1ABC9C", "ground-track-field": "#E67E22",
        "harbor": "#34495E", "bridge": "#E91E63", "large-vehicle": "#00BCD4",
        "small-vehicle": "#FF5722", "helicopter": "#8BC34A",
        "roundabout": "#FF9800", "soccer-ball-field": "#795548",
        "swimming-pool": "#607D8B",
    }
    for obj in result["objects"]:
        obj["color"] = CLASS_COLORS.get(obj["label"], "#64748b")

    context = {
        "annotated_url":  annotated_url,
        "scene_labels":   result["scene_labels"],
        "objects":        result["objects"],
        "result_json":    result_json,
        "total_objects":  total_objects,
        "unique_classes": unique_classes,
        "original_name":  upload.name,
    }
    return render(request, "results.html", context)
