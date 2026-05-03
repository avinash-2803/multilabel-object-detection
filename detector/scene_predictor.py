"""
scene_predictor.py
==================
Loads MLCGCN_DN169 — full complex architecture.
Checkpoint: epoch=28, mAP=96.61%
Fix: thresholds scaled by 0.6 to correctly surface airport, airplane labels.
"""

import pickle
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

_model      = None
_categories = None
_thresholds = None
_device     = None

ASSETS = Path(__file__).resolve().parent.parent / "model_assets"

_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

MLRSNET_CLASSES = [
    'airplane','airport','bareland','baseball_diamond','basketball_court',
    'beach','bridge','chaparral','cloud','commercial_area',
    'dense_residential_area','desert','eroded_farmland','farmland','forest',
    'freeway','golf_course','ground_track_field','harbor&port','industrial_area',
    'intersection','island','lake','meadow','mobile_home_park',
    'mountain','overpass','park','parking_lot','parkway',
    'railway','railway_station','river','roundabout','shipping_yard',
    'snowberg','sparse_residential_area','stadium','storage_tank','swimmimg_pool',
    'tennis_court','terrace','transmission_tower','vegetable_greenhouse',
    'wetland','wind_turbine'
]


def _load():
    global _model, _categories, _thresholds, _device

    if _model is not None:
        return

    print("[ScenePredictor] Loading MLCGCN_DN169 (96.61% mAP) ...")
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _categories = MLRSNET_CLASSES[:]
    num_classes  = 46
    print(f"[ScenePredictor] {num_classes} classes confirmed")

    # ── Thresholds ────────────────────────────────────────────────────────────
    thresh_path = ASSETS / "best_thresh.npy"
    if thresh_path.exists():
        raw         = np.load(str(thresh_path))
        # Scale thresholds down by 40% so that correct scene labels
        # (airport, airplane etc.) surface even when model output
        # probabilities are slightly shifted from training distribution.
        # Original thresholds were tuned on validation set during training.
        # The 0.6 multiplier was determined empirically for this deployment.
        _thresholds = [float(t) * 0.6 for t in raw]
        print(f"[ScenePredictor] Thresholds loaded | "
              f"original mean={float(np.mean(raw)):.3f} | "
              f"scaled mean={float(np.mean(_thresholds)):.3f}")
    else:
        _thresholds = [0.3] * num_classes
        print("[ScenePredictor] best_thresh.npy not found — using 0.3")

    # ── Adjacency matrix ──────────────────────────────────────────────────────
    adj_matrix = None
    adj_path   = ASSETS / "adj_matrix.pkl"
    if adj_path.exists():
        with open(adj_path, "rb") as f:
            adj_raw = pickle.load(f)
        adj_matrix = (adj_raw if isinstance(adj_raw, np.ndarray)
                      else adj_raw.get("adj"))
        print(f"[ScenePredictor] adj_matrix loaded: "
              f"shape={np.array(adj_matrix).shape}")
    else:
        print("[ScenePredictor] adj_matrix.pkl not found — using identity")

    # ── Build model ───────────────────────────────────────────────────────────
    from detector.model_src.mlcgcn import MLCGCN_DN169

    _model = MLCGCN_DN169(
        num_classes = num_classes,
        embed_dim   = 300,
        feat_dim    = 1664,
        pretrained  = False,
        adj_matrix  = adj_matrix,
        t           = 0.4,
    ).to(_device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt  = torch.load(
        str(ASSETS / "MLCGCN_DN169_best.pth"),
        map_location  = _device,
        weights_only  = False,
    )
    state = ckpt["state_dict"]

    try:
        _model.load_state_dict(state, strict=True)
        print("[ScenePredictor] ✓ Weights loaded perfectly (strict=True)")
    except Exception as e:
        print(f"[ScenePredictor] strict=True failed: {e}")
        missing, unexpected = _model.load_state_dict(state, strict=False)
        print(f"  Missing    ({len(missing)}): {missing}")
        print(f"  Unexpected ({len(unexpected)} keys)")

    # CRITICAL — eval() before any inference
    # Ensures pool_bn uses stored running stats, not batch stats
    # This prevents the batch-size=1 crash during inference
    _model.eval()
    print(f"[ScenePredictor] Ready | device={_device}")


def predict(pil_image: Image.Image) -> list:
    _load()
    _model.eval()

    img    = pil_image.convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    # Plain Python dicts — zero numpy indexing issues
    idx_to_label  = {i: _categories[i] for i in range(46)}
    label_to_prob = {_categories[i]: float(probs[i]) for i in range(46)}

    # Collect all labels above scaled per-class threshold
    labels = [
        idx_to_label[i]
        for i in range(46)
        if float(probs[i]) >= float(_thresholds[i])
    ]

    # Always return at least top-1 label — never return empty list
    if not labels:
        top_idx = int(np.argmax(probs))
        labels  = [idx_to_label[top_idx]]
        print(f"[ScenePredictor] No labels above threshold — "
              f"returning top-1: {labels[0]} "
              f"(prob={float(probs[top_idx]):.3f})")

    # Sort by probability descending — highest confidence label first
    result = sorted(labels, key=lambda l: label_to_prob[l], reverse=True)

    print(f"[ScenePredictor] Predicted {len(result)} labels: {result}")
    return result