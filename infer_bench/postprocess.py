"""Post-processing for classification and YOLO-style detection outputs.

Detection decoding follows the YOLOv8 / YOLO11 export layout: one output of
shape (1, 4 + num_classes, num_anchors) where the first four rows are the box
centre x, centre y, width and height in input pixels and the remaining rows
are class scores (already passed through a sigmoid, no separate objectness).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- classification

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values of a 1-D array, best first."""
    scores = np.asarray(scores).ravel()
    k = min(k, scores.size)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


# --------------------------------------------------------------------- detection

@dataclass
class Detections:
    boxes: np.ndarray  # (n, 4) float32, x1 y1 x2 y2
    scores: np.ndarray  # (n,) float32
    class_ids: np.ndarray  # (n,) int64

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @staticmethod
    def empty() -> "Detections":
        return Detections(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union of two sets of xyxy boxes -> (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, best score first."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo(
    output: np.ndarray,
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    max_candidates: int = 3000,
    layout: str = "auto",
) -> Detections:
    """Turn a raw YOLOv8/YOLO11 output into detections (input-pixel coordinates).

    layout: "channels_first" for (4+nc, N), "channels_last" for (N, 4+nc), or
            "auto", which assumes the smaller dimension is the channel count
            (true for every real YOLO export: e.g. 84 channels vs 8400 anchors).
    """
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"expected a (1, 4+nc, N) or (4+nc, N) output, got shape {np.shape(output)}")
    if layout == "channels_first" or (layout == "auto" and pred.shape[0] < pred.shape[1]):
        pred = pred.T
    elif layout not in ("auto", "channels_last", "channels_first"):
        raise ValueError(f"unknown layout '{layout}'")
    if pred.shape[1] < 5:
        raise ValueError("a detection output needs at least 4 box values and 1 class score per anchor")

    xywh = pred[:, :4].astype(np.float32, copy=False)
    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids].astype(np.float32, copy=False)

    keep = np.flatnonzero(scores >= conf_threshold)
    if keep.size == 0:
        return Detections.empty()
    if keep.size > max_candidates:  # keep only the best candidates before the O(n^2) NMS
        keep = keep[np.argsort(-scores[keep])[:max_candidates]]
    xywh, scores, class_ids = xywh[keep], scores[keep], class_ids[keep].astype(np.int64)

    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    # Class-aware NMS: shift every class far away from the others so boxes of
    # different classes can never suppress each other.
    offset = class_ids.astype(np.float32)[:, None] * (float(np.abs(boxes).max()) + 1.0)
    kept = nms(boxes + offset, scores, iou_threshold)[:max_det]
    return Detections(boxes[kept], scores[kept], class_ids[kept])


def match_detections(
    reference: Detections, candidate: Detections, iou_threshold: float = 0.5
) -> tuple[int, int, int]:
    """Count (true positives, false positives, false negatives) of candidate vs reference.

    A candidate box is a true positive when it overlaps a not-yet-matched
    reference box of the same class with IoU >= ``iou_threshold``. Candidates
    are processed best score first. The reference plays the role of ground
    truth, so this measures *agreement*, not accuracy.
    """
    n_ref, n_cand = len(reference), len(candidate)
    if n_ref == 0 or n_cand == 0:
        return 0, n_cand, n_ref
    ious = iou_matrix(candidate.boxes, reference.boxes)
    same_class = candidate.class_ids[:, None] == reference.class_ids[None, :]
    ious = np.where(same_class, ious, 0.0)
    matched_ref = np.zeros(n_ref, dtype=bool)
    tp = 0
    for ci in np.argsort(-candidate.scores, kind="stable"):
        row = np.where(matched_ref, 0.0, ious[ci])
        best = int(row.argmax())
        if row[best] >= iou_threshold:
            matched_ref[best] = True
            tp += 1
    return tp, n_cand - tp, n_ref - tp
