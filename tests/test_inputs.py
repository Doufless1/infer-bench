import numpy as np
import pytest
from PIL import Image

from infer_bench import inputs
from infer_bench.backends.base import TensorSpec
from infer_bench.preprocess import PreprocessConfig

IMG = TensorSpec("images", ("batch", 3, 224, 224), "float32")
DYN = TensorSpec("images", ("batch", 3, "height", "width"), "float32")


def test_dynamic_batch_dimension_takes_the_batch_size():
    assert inputs.resolve_shape(IMG, 4) == (4, 3, 224, 224)
    assert inputs.resolve_shape(TensorSpec("x", (None, 5), "float32"), 2) == (2, 5)


def test_fixed_batch_dimension_must_match():
    spec = TensorSpec("x", (1, 3, 8, 8), "float32")
    assert inputs.resolve_shape(spec, 1) == (1, 3, 8, 8)
    with pytest.raises(ValueError, match="fixed batch dimension of 1"):
        inputs.resolve_shape(spec, 8)


def test_dynamic_image_size_needs_a_hint():
    with pytest.raises(ValueError, match="--input-size"):
        inputs.resolve_shape(DYN, 1)
    assert inputs.resolve_shape(DYN, 2, image_size=(480, 640)) == (2, 3, 480, 640)


def test_image_size_must_agree_with_fixed_dimensions():
    with pytest.raises(ValueError, match="fixed at 224"):
        inputs.resolve_shape(IMG, 1, image_size=(320, 320))


def test_explicit_shape_override_is_validated():
    assert inputs.resolve_shape(DYN, 1, override=(1, 3, 100, 120)) == (1, 3, 100, 120)
    with pytest.raises(ValueError, match="dimensions"):
        inputs.resolve_shape(DYN, 1, override=(1, 3, 100))
    with pytest.raises(ValueError, match="fixed at 224"):
        inputs.resolve_shape(IMG, 1, override=(1, 3, 224, 100))
    with pytest.raises(ValueError, match="batch"):
        inputs.resolve_shape(DYN, 2, override=(1, 3, 100, 120))


@pytest.mark.parametrize("dtype", ["float32", "float16", "float64", "uint8", "int8", "int32", "int64", "bool"])
def test_random_array_has_the_requested_dtype(dtype):
    arr = inputs.random_array((2, 5), dtype, np.random.default_rng(0))
    assert arr.dtype == np.dtype(dtype) and arr.shape == (2, 5)


def test_random_array_rejects_unknown_dtypes():
    with pytest.raises(ValueError):
        inputs.random_array((2,), "complex64", np.random.default_rng(0))


def test_image_like_tensors_get_textures_that_differ_between_samples():
    arr = inputs.random_array((16, 3, 64, 64), "float32", np.random.default_rng(0))
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    sample_means = arr.mean(axis=(1, 2, 3))
    assert sample_means.std() > 0.03  # white noise would give ~0.002: every sample would look alike
    smooth = np.abs(np.diff(arr, axis=3)).mean()
    noise = np.abs(np.diff(np.random.default_rng(0).random(arr.shape, dtype=np.float32), axis=3)).mean()
    assert smooth < noise / 3  # neighbouring pixels are correlated, like in a picture


def test_non_image_float_tensors_are_uniform_noise():
    arr = inputs.random_array((1000, 4), "float32", np.random.default_rng(0))
    assert abs(arr.mean() - 0.5) < 0.03


def test_synthetic_feeds_are_reproducible_and_cycle_through_distinct_inputs():
    a = inputs.synthetic_feeds([IMG], 1, count=3, seed=5)
    b = inputs.synthetic_feeds([IMG], 1, count=3, seed=5)
    c = inputs.synthetic_feeds([IMG], 1, count=3, seed=6)
    assert len(a) == 3
    np.testing.assert_array_equal(a[0]["images"], b[0]["images"])
    assert not np.array_equal(a[0]["images"], c[0]["images"])
    assert not np.array_equal(a[0]["images"], a[1]["images"])


def test_synthetic_feeds_with_a_dynamic_size():
    feeds = inputs.synthetic_feeds([DYN], 2, count=1, image_size=(64, 96))
    assert feeds[0]["images"].shape == (2, 3, 64, 96)


def test_model_image_size():
    assert inputs.model_image_size(IMG) == (224, 224)
    assert inputs.model_image_size(DYN, (320, 480)) == (320, 480)
    with pytest.raises(ValueError, match="--input-size"):
        inputs.model_image_size(DYN)
    with pytest.raises(ValueError, match="conflicts"):
        inputs.model_image_size(IMG, (100, 100))
    with pytest.raises(ValueError, match="4-D"):
        inputs.model_image_size(TensorSpec("x", ("batch", 10), "float32"))


def test_image_feeds_group_images_into_batches_and_wrap_around():
    imgs = [np.full((40, 60, 3), v, dtype=np.uint8) for v in (10, 100, 200)]
    cfg = PreprocessConfig(height=16, width=16, mode="resize", scale=1.0)
    spec = TensorSpec("images", ("batch", 3, 16, 16), "float32")
    feeds = inputs.image_feeds(imgs, spec, cfg, batch_size=2)
    assert len(feeds) == 2  # ceil(3 / 2)
    assert feeds[0]["images"].shape == (2, 3, 16, 16)
    assert feeds[0]["images"][0].mean() == pytest.approx(10.0, abs=0.5)
    assert feeds[1]["images"][1].mean() == pytest.approx(10.0, abs=0.5)  # wrapped to the first image again
    with pytest.raises(ValueError):
        inputs.image_feeds([], spec, cfg, 1)


def test_parse_size():
    assert inputs.parse_size("480x640") == (480, 640)
    assert inputs.parse_size("32X64") == (32, 64)
    for bad in ("640", "axb", "0x10", "-1x5"):
        with pytest.raises(ValueError):
            inputs.parse_size(bad)


def test_parse_shape_override():
    assert inputs.parse_shape_override("images=1,3,640,640") == ("images", (1, 3, 640, 640))
    for bad in ("images", "images=1,x"):
        with pytest.raises(ValueError):
            inputs.parse_shape_override(bad)


def test_load_images_reads_files_from_disk(tmp_path):
    Image.fromarray(np.full((20, 30, 3), 77, dtype=np.uint8)).save(tmp_path / "a.png")
    from infer_bench.preprocess import list_images

    arrays = inputs.load_images(list_images(tmp_path))
    assert arrays[0].shape == (20, 30, 3) and int(arrays[0][0, 0, 0]) == 77
