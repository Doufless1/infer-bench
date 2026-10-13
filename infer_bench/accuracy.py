"""Accuracy of a candidate model *relative to a reference model*.

The usual question after quantising a model is "how much did I lose?". The
cleanest answer needs labelled data. Without labels, the next best thing is to
run the original (FP32) model and the candidate (for example INT8) on the same
inputs and measure how much their outputs agree. That is what this module
does, and it is deliberately reported under the name *agreement*, not
*accuracy*:

* cosine similarity and relative L2 error between the raw outputs;
* classification: how often both models pick the same top-1 class, and how
  often the reference's top-1 class is inside the candidate's top-5;
* detection (YOLO-style): precision / recall / F1 of the candidate's boxes
  when the reference's boxes are treated as ground truth (IoU >= 0.5, same
  class). This is **not** mAP.

Agreement with FP32 is a good detector of quantisation damage, but it cannot
tell you that the FP32 model itself is good.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends.base import Backend
from .postprocess import decode_yolo, match_detections, topk_indices

TASKS = ("auto", "none", "classification", "detection")
NEAR_ZERO = 1e-4  # reference outputs smaller than this on average carry no signal
SAME_OUTPUT = 0.9999  # consecutive reference outputs more similar than this: the model ignores its input


def flatten_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in outputs])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    diff = float(np.linalg.norm(reference - candidate))
    if denom == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / denom


def compare_models(
    reference: Backend,
    candidate: Backend,
    feeds: list[dict[str, np.ndarray]],
    *,
    task: str = "auto",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_source: str = "synthetic",
) -> dict[str, Any]:
    """Run both models on every feed (batch size 1) and summarise their agreement."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if not feeds:
        raise ValueError("need at least one input to compare on")

    cosines: list[float] = []
    rel_l2: list[float] = []
    max_abs: list[float] = []
    top1: list[bool] = []
    in_top5: list[bool] = []
    tp = fp = fn = n_ref_det = n_cand_det = 0
    nonfinite = 0
    ref_mean_abs: list[float] = []  # how big the reference outputs are
    ref_neighbour_cos: list[float] = []  # how similar the reference outputs of consecutive inputs are
    previous_ref: np.ndarray | None = None
    guessed_task = task == "auto"
    looks_like_detector = False

    for feed in feeds:
        ref_out = reference.run(feed)
        cand_out = candidate.run(feed)
        if task == "auto":
            first = np.asarray(ref_out[0])
            task = "classification" if first.ndim == 2 else "none"
            # (1, 4 + classes, anchors): fewer channels than anchors, and at least 4 boxes values + 1 score
            looks_like_detector = first.ndim == 3 and 5 <= first.shape[1] < first.shape[2]
        ref_vec, cand_vec = flatten_outputs(ref_out), flatten_outputs(cand_out)
        ref_mean_abs.append(float(np.mean(np.abs(ref_vec))))
        if previous_ref is not None and previous_ref.shape == ref_vec.shape:
            ref_neighbour_cos.append(cosine_similarity(previous_ref, ref_vec))
        previous_ref = ref_vec
        if not np.all(np.isfinite(cand_vec)):
            nonfinite += 1
            continue
        cosines.append(cosine_similarity(ref_vec, cand_vec))
        rel_l2.append(relative_l2_error(ref_vec, cand_vec))
        max_abs.append(float(np.max(np.abs(ref_vec - cand_vec))))

        if task == "classification":
            ref_logits = np.asarray(ref_out[0], dtype=np.float64).ravel()
            cand_logits = np.asarray(cand_out[0], dtype=np.float64).ravel()
            ref_top1 = int(ref_logits.argmax())
            top1.append(ref_top1 == int(cand_logits.argmax()))
            in_top5.append(ref_top1 in set(topk_indices(cand_logits, 5).tolist()))
        elif task == "detection":
            ref_det = decode_yolo(ref_out[0], conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            cand_det = decode_yolo(cand_out[0], conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            a, b, c = match_detections(ref_det, cand_det, 0.5)
            tp, fp, fn = tp + a, fp + b, fn + c
            n_ref_det += len(ref_det)
            n_cand_det += len(cand_det)

    n_used = len(cosines)
    notes = [
        "Agreement with the reference model, not ground-truth accuracy: no labels were used.",
    ]
    if input_source == "synthetic":
        notes.append("Evaluated on synthetic inputs: run with --images for a meaningful number.")
    if nonfinite:
        notes.append(f"{nonfinite} sample(s) produced NaN or infinity in the candidate output and were excluded.")
    if guessed_task and task == "none" and looks_like_detector:
        notes.append(
            "The output looks like a detector's (channels x anchors), but no task was given, so only cosine "
            "similarity and L2 error were computed. A detector can keep a cosine similarity near 1 and still lose "
            "all its detections: pass the detection task to also get box agreement (F1)."
        )
    mean_abs = float(np.mean(ref_mean_abs)) if ref_mean_abs else 0.0
    neighbour_cos = float(np.mean(ref_neighbour_cos)) if ref_neighbour_cos else None
    if mean_abs < NEAR_ZERO:
        notes.append(
            f"The reference output is almost zero on these inputs (mean |value| = {mean_abs:.1e}), so the agreement "
            "numbers are not meaningful. This happens when the inputs do not contain what the model looks for "
            "(for example a text detector shown pictures without text). Use real images with --images."
        )
    elif neighbour_cos is not None and len(ref_neighbour_cos) >= 4 and neighbour_cos > SAME_OUTPUT:
        notes.append(
            "The reference model gives almost the same output for every input, so high agreement is trivial. "
            "Use real, varied images with --images."
        )

    if task == "detection" and n_ref_det == 0:
        notes.append(
            "The reference model found no objects at all on these inputs, so detection agreement is undefined. "
            "Use real images that contain objects (--images)."
        )
    if task == "detection" and n_ref_det > 0 and n_cand_det == 0:
        notes.append(
            "The candidate produced no detections although the reference did. A frequent cause for a "
            "quantised detector: the output mixes pixel-scale box values with 0..1 class scores in one "
            "tensor, so they share one quantisation scale and the scores round to zero. Try quantising "
            "only the convolutions (--op-types Conv) or excluding the output nodes (--exclude-nodes)."
        )
    out: dict[str, Any] = {
        "task": task,
        "n_samples": len(feeds),
        "n_compared": n_used,
        "input_source": input_source,
        "nonfinite_outputs": nonfinite,
        "reference_mean_abs_output": mean_abs,
        "cosine_similarity_mean": float(np.mean(cosines)) if cosines else None,
        "cosine_similarity_min": float(np.min(cosines)) if cosines else None,
        "relative_l2_error_mean": float(np.mean(rel_l2)) if rel_l2 else None,
        "relative_l2_error_max": float(np.max(rel_l2)) if rel_l2 else None,
        "max_abs_error_max": float(np.max(max_abs)) if max_abs else None,
        "top1_agreement": float(np.mean(top1)) if top1 else None,
        "ref_top1_in_cand_top5": float(np.mean(in_top5)) if in_top5 else None,
        "detection": None,
        "notes": notes,
    }
    if task == "detection":
        denom = 2 * tp + fp + fn
        out["detection"] = {
            "iou_threshold_match": 0.5,
            "conf_threshold": conf_threshold,
            "nms_iou_threshold": iou_threshold,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "reference_detections": n_ref_det,
            "candidate_detections": n_cand_det,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "f1_iou50": (2 * tp / denom) if denom else None,
            "note": "Reference detections are treated as ground truth. This is agreement, not mAP.",
        }
    return out
