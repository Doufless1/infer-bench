"""Image pre-processing: load, resize (letterbox or stretch), normalise.

This is the "pre-processing" stage of the pipeline benchmark. It is kept in
plain numpy + Pillow so that its cost is measured honestly and can be compared
with the cost of the model itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class PreprocessConfig:
    """How a raw RGB image becomes a model input tensor.

    mode:      "letterbox" keeps the aspect ratio and pads (what YOLO models expect);
               "resize" stretches the image to the model size.
    scale:     multiply pixel values by this (1/255 maps 0..255 to 0..1).
    mean/std:  per-channel normalisation applied after scaling: (x - mean) / std.
    pad_value: grey value used for letterbox padding (114 is the YOLO convention).
    """

    height: int
    width: int
    mode: str = "letterbox"
    scale: float = 1.0 / 255.0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pad_value: int = 114


@dataclass(frozen=True)
class ResizeMeta:
    """What is needed to map model-space coordinates back to the original image."""

    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


def list_images(folder: str | Path, limit: int | None = None, seed: int | None = None) -> list[Path]:
    """Image files in a folder (not recursive), sorted by name.

    With ``limit``, only that many are returned: the first ones by name, or, when a ``seed`` is given, a random
    sample (the same for the same seed, returned in name order). A random sample matters for calibration:
    the first 100 files of a folder are often 100 consecutive video frames or a single class.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) in {folder}")
    if not limit or limit >= len(paths):
        return paths
    if seed is None:
        return paths[:limit]
    return sorted(random.Random(seed).sample(paths, limit))


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an H x W x 3 uint8 RGB array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def texture_batch(shape: tuple[int, int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Random smooth "images" in 0..1, shape (N, C, H, W), float32.

    Each sample is a random brightness plus a few random sine gratings and a
    little noise. White noise would make every sample look the same to a
    network (its features average out), so model outputs would hardly vary
    between samples and any agreement metric would be trivially perfect.
    These textures differ in brightness, orientation and frequency, which
    exercises a model far more like real pictures do - though they are still
    not real data.
    """
    n, c, h, w = shape
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, None, :, None]
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, None, None, :]
    img = np.full((n, c, h, w), 0.5, dtype=np.float32)
    img += rng.uniform(-0.25, 0.25, size=(n, c, 1, 1)).astype(np.float32)
    for _ in range(3):
        fx = rng.uniform(0.0, 10.0, size=(n, c, 1, 1)).astype(np.float32)
        fy = rng.uniform(0.0, 10.0, size=(n, c, 1, 1)).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, c, 1, 1)).astype(np.float32)
        amp = rng.uniform(0.05, 0.30, size=(n, c, 1, 1)).astype(np.float32)
        img += amp * np.cos(2.0 * np.pi * (fx * u + fy * v) + phase)
    img += rng.normal(0.0, 0.02, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def synthetic_image(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """A random textured H x W x 3 uint8 image, used when no image folder is given."""
    tex = texture_batch((1, 3, height, width), rng)[0]  # (3, H, W) in 0..1
    return np.ascontiguousarray((tex.transpose(1, 2, 0) * 255.0).astype(np.uint8))


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, ResizeMeta]:
    """H x W x 3 uint8 RGB image -> (1, 3, H', W') float32 tensor and resize metadata."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 image, got shape {image.shape}")
    h0, w0 = image.shape[:2]
    height, width = cfg.height, cfg.width
    pil = Image.fromarray(image)

    if cfg.mode == "letterbox":
        r = min(height / h0, width / w0)
        new_w = min(width, max(1, round(w0 * r)))
        new_h = min(height, max(1, round(h0 * r)))
        resized = np.asarray(pil.resize((new_w, new_h), Image.BILINEAR))
        canvas = np.full((height, width, 3), cfg.pad_value, dtype=np.uint8)
        left, top = (width - new_w) // 2, (height - new_h) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        meta = ResizeMeta(r, r, left, top, w0, h0)
    elif cfg.mode == "resize":
        canvas = np.asarray(pil.resize((width, height), Image.BILINEAR))
        meta = ResizeMeta(width / w0, height / h0, 0, 0, w0, h0)
    else:
        raise ValueError(f"unknown preprocess mode '{cfg.mode}' (use 'letterbox' or 'resize')")

    x = canvas.astype(np.float32) * np.float32(cfg.scale)
    x = (x - np.asarray(cfg.mean, dtype=np.float32)) / np.asarray(cfg.std, dtype=np.float32)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None, ...]
    return x.astype(np.float32, copy=False), meta


def unmap_boxes(boxes_xyxy: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    """Map boxes from model-input pixels back to the original image, clipped to it."""
    out = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    if out.size == 0:
        return out.reshape(0, 4)
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale_x
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale_y
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, meta.orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, meta.orig_h)
    return out
