"""Building model inputs: synthetic tensors or real images.

Synthetic inputs are fine for measuring speed (on most hardware the speed of
a dense network does not depend on the values). They are *not* fine for
judging accuracy or calibrating quantisation - use ``--images`` for that.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .backends.base import TensorSpec
from .preprocess import PreprocessConfig, load_image_rgb, preprocess, texture_batch


def _is_dynamic(dim: object) -> bool:
    return not (isinstance(dim, int) and dim > 0)


def resolve_shape(
    spec: TensorSpec,
    batch_size: int,
    *,
    override: tuple[int, ...] | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    """Concrete shape for one model input.

    The first dimension is the batch size. Other dynamic dimensions must be
    given with ``override`` (a full shape) or, for 4-D image inputs, with
    ``image_size`` = (height, width).
    """
    shape = tuple(spec.shape)
    if override is not None:
        if len(override) != len(shape):
            raise ValueError(f"input '{spec.name}' has {len(shape)} dimensions, override has {len(override)}")
        for i, (want, got) in enumerate(zip(shape, override)):
            if not _is_dynamic(want) and want != got:
                raise ValueError(f"input '{spec.name}' dimension {i} is fixed at {want}, override says {got}")
        if len(override) and override[0] != batch_size:
            raise ValueError(f"override batch dimension {override[0]} does not match --batch {batch_size}")
        return tuple(int(d) for d in override)

    resolved: list[int] = []
    for i, dim in enumerate(shape):
        if i == 0:
            if _is_dynamic(dim):
                resolved.append(batch_size)
            elif dim == batch_size:
                resolved.append(int(dim))
            else:
                raise ValueError(
                    f"input '{spec.name}' has a fixed batch dimension of {dim}, but --batch is {batch_size}. "
                    f"Use --batch {dim}, or export the model with a dynamic batch dimension."
                )
        elif not _is_dynamic(dim):
            resolved.append(int(dim))
        elif image_size is not None and len(shape) == 4 and i in (2, 3):
            resolved.append(int(image_size[i - 2]))
        else:
            raise ValueError(
                f"input '{spec.name}' has a dynamic dimension {i} ({dim!r}). "
                f"Give the size with --input-size HxW (image models) or --input-shape {spec.name}=<dims>."
            )
    if image_size is not None and len(shape) == 4:
        for i in (2, 3):
            if not _is_dynamic(shape[i]) and shape[i] != image_size[i - 2]:
                raise ValueError(
                    f"--input-size says {image_size[i - 2]} but input '{spec.name}' is fixed at {shape[i]}"
                )
    return tuple(resolved)


def _looks_like_images(shape: tuple[int, ...]) -> bool:
    return len(shape) == 4 and shape[1] in (1, 3) and min(shape[2:]) >= 8


def random_array(shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
    """Synthetic data of the right dtype.

    Float tensors shaped like NCHW images get smooth random textures in 0..1
    (see ``texture_batch``); other float tensors get uniform [0,1) values;
    bytes, small ints and bools are uniform.
    """
    if dtype in ("float32", "float16", "float64"):
        if _looks_like_images(shape):
            return texture_batch(shape, rng).astype(dtype)  # type: ignore[arg-type]
        return rng.random(shape, dtype=np.float32).astype(dtype)
    if dtype == "uint8":
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if dtype == "int8":
        return rng.integers(-128, 128, size=shape, dtype=np.int8)
    if dtype in ("int32", "int64"):
        return rng.integers(0, 1000, size=shape, dtype=np.dtype(dtype))
    if dtype == "bool":
        return rng.integers(0, 2, size=shape).astype(bool)
    raise ValueError(f"cannot generate synthetic data for dtype '{dtype}'")


def synthetic_feeds(
    specs: Iterable[TensorSpec],
    batch_size: int,
    *,
    count: int = 8,
    seed: int = 0,
    overrides: dict[str, tuple[int, ...]] | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[dict[str, np.ndarray]]:
    """``count`` different random input dicts (cycled through during the run)."""
    specs = list(specs)
    overrides = overrides or {}
    shapes = {
        s.name: resolve_shape(s, batch_size, override=overrides.get(s.name), image_size=image_size)
        for s in specs
    }
    rng = np.random.default_rng(seed)
    return [
        {s.name: random_array(shapes[s.name], s.dtype, rng) for s in specs}
        for _ in range(max(1, count))
    ]


def model_image_size(
    spec: TensorSpec, image_size: tuple[int, int] | None = None
) -> tuple[int, int]:
    """(height, width) a 4-D NCHW image input expects."""
    if len(spec.shape) != 4:
        raise ValueError(f"input '{spec.name}' is {len(spec.shape)}-D; image mode needs a 4-D NCHW input")
    dims = spec.shape[2:4]
    if all(not _is_dynamic(d) for d in dims):
        h, w = int(dims[0]), int(dims[1])
        if image_size is not None and tuple(image_size) != (h, w):
            raise ValueError(f"--input-size {image_size} conflicts with the model's fixed size {(h, w)}")
        return h, w
    if image_size is None:
        raise ValueError(f"input '{spec.name}' has dynamic spatial size: pass --input-size HxW")
    return int(image_size[0]), int(image_size[1])


def image_feeds(
    images: list[np.ndarray],
    spec: TensorSpec,
    cfg: PreprocessConfig,
    batch_size: int,
) -> list[dict[str, np.ndarray]]:
    """Pre-process real images into model feeds of ``batch_size`` images each."""
    if not images:
        raise ValueError("no images given")
    tensors = [preprocess(img, cfg)[0] for img in images]
    feeds: list[dict[str, np.ndarray]] = []
    n = len(tensors)
    n_feeds = max(1, -(-n // batch_size))  # ceil division
    for k in range(n_feeds):
        batch = [tensors[(k * batch_size + j) % n] for j in range(batch_size)]
        feeds.append({spec.name: np.concatenate(batch, axis=0).astype(spec.dtype, copy=False)})
    return feeds


def load_images(paths: Iterable) -> list[np.ndarray]:
    return [load_image_rgb(p) for p in paths]


def parse_size(text: str) -> tuple[int, int]:
    """'480x640' -> (480, 640). The order is height x width."""
    try:
        h, w = (int(part) for part in text.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"size must look like 640x640 (height x width), got '{text}'") from exc
    if h <= 0 or w <= 0:
        raise ValueError("size must be positive")
    return h, w


def parse_shape_override(text: str) -> tuple[str, tuple[int, ...]]:
    """'images=1,3,640,640' -> ('images', (1, 3, 640, 640))."""
    if "=" not in text:
        raise ValueError(f"--input-shape must look like NAME=1,3,224,224, got '{text}'")
    name, dims = text.split("=", 1)
    try:
        shape = tuple(int(d) for d in dims.split(","))
    except ValueError as exc:
        raise ValueError(f"bad shape '{dims}' in --input-shape") from exc
    return name.strip(), shape
