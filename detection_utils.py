"""
detection_utils.py — 3-Layer Fragment Detection Helpers
────────────────────────────────────────────────────────
Layer 1: Template Matching      — background-agnostic, spacing-agnostic (primary)
Layer 2: Background Subtraction — contour-based fallback for clean backgrounds
Layer 3: DBSCAN Clustering      — replaces broken chain-merge for FastSAM boxes
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


# ── LAYER 1 ────────────────────────────────────────────────────────────────────
def find_pieces_by_template(scattered_np: np.ndarray,
                             master_crops: dict,
                             confidence_threshold: float = 0.50) -> list:
    """
    For each master crop stored in memory, locate it in the scattered image using
    normalised cross-correlation (TM_CCOEFF_NORMED).

    Returns
    -------
    list of (x1, y1, x2, y2, crop_id, confidence)
    """
    found = []
    sh, sw = scattered_np.shape[:2]

    # Pre-compute grayscale scene once
    gray_scene = cv2.cvtColor(scattered_np, cv2.COLOR_RGB2GRAY)

    for crop_id, crop_np in master_crops.items():
        if crop_np is None or crop_np.size == 0:
            continue

        template = cv2.cvtColor(crop_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        th, tw = template.shape[:2]

        # Skip if template is as large as or larger than the scene
        if th >= sh or tw >= sw:
            continue

        result = cv2.matchTemplate(gray_scene, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= confidence_threshold:
            x1, y1 = max_loc
            found.append((x1, y1, x1 + tw, y1 + th, crop_id, round(float(max_val), 4)))

    return found


# ── LAYER 2 ────────────────────────────────────────────────────────────────────
def find_pieces_by_background_subtraction(scattered_np: np.ndarray,
                                           n_expected: int) -> list:
    """
    Estimate the table background colour from the image border, create a
    foreground mask, then extract bounding boxes from external contours.

    Returns
    -------
    list of (x1, y1, x2, y2)
    """
    h, w = scattered_np.shape[:2]
    border = max(10, min(h, w) // 20)

    # Median of border pixels is more robust than mean against corner fragments
    border_pixels = np.concatenate([
        scattered_np[:border, :].reshape(-1, 3),
        scattered_np[-border:, :].reshape(-1, 3),
        scattered_np[:, :border].reshape(-1, 3),
        scattered_np[:, -border:].reshape(-1, 3),
    ])
    bg_color = np.median(border_pixels, axis=0).astype(float)

    # Pixels whose total colour distance from background exceeds threshold
    diff = np.abs(scattered_np.astype(float) - bg_color)
    mask = (diff.sum(axis=2) > 60).astype(np.uint8) * 255

    # Close small gaps inside pieces, then open to remove tiny noise
    ks = max(5, int(min(h, w) / 60))
    kernel_close = np.ones((ks, ks), np.uint8)
    kernel_open  = np.ones((3, 3),   np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area     = w * h
    expected_area = img_area / max(n_expected, 1)
    min_area      = expected_area * 0.15
    max_area      = expected_area * 4.0

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw >= 20 and bh >= 20:
                boxes.append((x, y, x + bw, y + bh))

    return boxes


# ── LAYER 3 HELPER ─────────────────────────────────────────────────────────────
def cluster_boxes_dbscan(raw_boxes: list,
                          expected_piece_diagonal: float,
                          min_size: int = 25) -> list:
    """
    Group raw FastSAM bounding boxes using DBSCAN on their centres.
    Boxes in the same cluster (belonging to one split piece) are merged.
    eps = 40 % of expected piece diagonal — small enough to avoid
    bridging genuinely separate pieces.

    Returns
    -------
    list of merged (x1, y1, x2, y2)
    """
    if not raw_boxes:
        return []

    centers = np.array(
        [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in raw_boxes],
        dtype=float
    )
    eps = max(15.0, expected_piece_diagonal * 0.40)
    labels = DBSCAN(eps=eps, min_samples=1).fit(centers).labels_

    merged = []
    for label in set(labels):
        if label == -1:
            continue
        group = [raw_boxes[i] for i in range(len(raw_boxes)) if labels[i] == label]
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
            merged.append((x1, y1, x2, y2))

    return merged
