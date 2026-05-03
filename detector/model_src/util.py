"""
util.py  —  copied from /content/MLRSNet_MultiLabel/util.py (Colab)
Contains gen_adj used by models_zoo.py's GCNModel, plus
image-transform helpers used during training.
"""
import math
import torch
import numpy as np
import random
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms (used only during training — kept here for completeness)
# ─────────────────────────────────────────────────────────────────────────────

class Warp:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class MultiScaleCrop:
    def __init__(self, input_size, scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales        = scales if scales else [1, 0.875, 0.75, 0.66]
        self.max_distort   = max_distort
        self.fix_crop      = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size    = ([input_size, input_size]
                              if isinstance(input_size, int) else input_size)
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        cw, ch, ow, oh = self._sample_crop_size(img.size)
        return img.crop((ow, oh, ow + cw, oh + ch)).resize(
            (self.input_size[0], self.input_size[1]), self.interpolation)

    def _sample_crop_size(self, im_size):
        base = min(im_size)
        cs = [int(base * x) for x in self.scales]
        ch = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in cs]
        cw = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in cs]
        pairs = [(w, h) for i, h in enumerate(ch)
                 for j, w in enumerate(cw) if abs(i - j) <= self.max_distort]
        cp = random.choice(pairs)
        if not self.fix_crop:
            return (cp[0], cp[1],
                    random.randint(0, im_size[0] - cp[0]),
                    random.randint(0, im_size[1] - cp[1]))
        offsets = self._fill_fix_offset(im_size[0], im_size[1], cp[0], cp[1])
        wo, ho = random.choice(offsets)
        return cp[0], cp[1], wo, ho

    def _fill_fix_offset(self, iw, ih, cw, ch):
        ws = (iw - cw) // 4
        hs = (ih - ch) // 4
        ret = [(0, 0), (4*ws, 0), (0, 4*hs), (4*ws, 4*hs), (2*ws, 2*hs)]
        if self.more_fix_crop:
            ret += [(0, 2*hs), (4*ws, 2*hs), (2*ws, 4*hs), (2*ws, 0),
                    (ws, hs), (3*ws, hs), (ws, 3*hs), (3*ws, 3*hs)]
        return ret


# ─────────────────────────────────────────────────────────────────────────────
# Graph adjacency normalisation  — used by GCNModel in models_zoo.py
# ─────────────────────────────────────────────────────────────────────────────

def gen_adj(A):
    """Symmetric normalisation:  D^{-1/2} A D^{-1/2}"""
    D = torch.pow(A.sum(1).float().clamp(min=1e-6), -0.5)
    D = torch.diag(D)
    return torch.matmul(torch.matmul(A, D).t(), D)
