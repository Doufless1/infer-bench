import numpy as np
import pytest
from PIL import Image

from infer_bench import preprocess as pp


def _image(h=480, w=640, value=200):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_letterbox_geometry_and_padding():
    cfg = pp.PreprocessConfig(height=320, width=320, mode="letterbox", scale=1.0)
    x, meta = pp.preprocess(_image(480, 640, 200), cfg)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert meta.scale_x == meta.scale_y == 0.5  # 640 -> 320
    assert (meta.pad_x, meta.pad_y) == (0, 40)  # 240 high image centred in 320: 40 rows on each side
    assert np.all(x[0, :, :40, :] == 114)  # padding rows
    assert np.all(x[0, :, 280:, :] == 114)
    assert np.allclose(x[0, :, 40:280, :], 200, atol=1)  # the picture itself


def test_resize_mode_stretches_without_padding():
    cfg = pp.PreprocessConfig(height=100, width=50, mode="resize", scale=1.0)
    x, meta = pp.preprocess(_image(200, 200, 90), cfg)
    assert x.shape == (1, 3, 100, 50)
    assert (meta.pad_x, meta.pad_y) == (0, 0)
    assert (meta.scale_x, meta.scale_y) == (0.25, 0.5)
    assert np.allclose(x, 90, atol=1)


def test_scale_mean_and_std_are_applied_per_channel():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 255, 0, 255
    cfg = pp.PreprocessConfig(8, 8, mode="resize", scale=1 / 255, mean=(0.5, 0.5, 0.0), std=(0.5, 0.25, 1.0))
    x, _ = pp.preprocess(img, cfg)
    assert np.allclose(x[0, 0], (1 - 0.5) / 0.5)
    assert np.allclose(x[0, 1], (0 - 0.5) / 0.25)
    assert np.allclose(x[0, 2], 1.0)


def test_unmap_boxes_inverts_the_letterbox():
    cfg = pp.PreprocessConfig(height=320, width=320, mode="letterbox")
    _, meta = pp.preprocess(_image(480, 640), cfg)
    original = np.array([[100.0, 60.0, 400.0, 300.0]], dtype=np.float32)
    in_model_space = original.copy()
    in_model_space[:, [0, 2]] = in_model_space[:, [0, 2]] * meta.scale_x + meta.pad_x
    in_model_space[:, [1, 3]] = in_model_space[:, [1, 3]] * meta.scale_y + meta.pad_y
    np.testing.assert_allclose(pp.unmap_boxes(in_model_space, meta), original, atol=1e-3)


def test_unmap_boxes_clips_to_the_original_image():
    cfg = pp.PreprocessConfig(height=320, width=320, mode="letterbox")
    _, meta = pp.preprocess(_image(480, 640), cfg)
    out = pp.unmap_boxes(np.array([[-50.0, -50.0, 1000.0, 1000.0]], dtype=np.float32), meta)
    assert out[0].tolist() == [0.0, 0.0, 640.0, 480.0]


def test_unmap_boxes_with_no_boxes():
    meta = pp.ResizeMeta(1.0, 1.0, 0, 0, 10, 10)
    assert pp.unmap_boxes(np.zeros((0, 4)), meta).shape == (0, 4)


def test_preprocess_rejects_bad_input():
    cfg = pp.PreprocessConfig(8, 8)
    with pytest.raises(ValueError, match="H x W x 3"):
        pp.preprocess(np.zeros((8, 8), dtype=np.uint8), cfg)
    with pytest.raises(ValueError, match="unknown preprocess mode"):
        pp.preprocess(_image(8, 8), pp.PreprocessConfig(8, 8, mode="crop"))


def test_tall_image_is_padded_left_and_right():
    cfg = pp.PreprocessConfig(height=100, width=100, mode="letterbox", scale=1.0)
    x, meta = pp.preprocess(_image(200, 100, 50), cfg)
    assert meta.scale_x == 0.5 and meta.pad_x == 25 and meta.pad_y == 0
    assert np.all(x[0, :, :, :25] == 114)


def test_list_and_load_images(tmp_path):
    Image.fromarray(_image(10, 12, 5)).save(tmp_path / "b.png")
    Image.fromarray(_image(10, 12, 6)).save(tmp_path / "a.jpg")
    (tmp_path / "notes.txt").write_text("not an image")
    names = [p.name for p in pp.list_images(tmp_path)]
    assert names == ["a.jpg", "b.png"]  # sorted, non-images skipped
    assert [p.name for p in pp.list_images(tmp_path, limit=1)] == ["a.jpg"]
    assert pp.load_image_rgb(tmp_path / "b.png").shape == (10, 12, 3)


def test_a_seeded_sample_of_a_folder_is_random_but_repeatable(tmp_path):
    for i in range(30):
        Image.fromarray(_image(6, 6, i)).save(tmp_path / f"frame_{i:02d}.png")
    first_five = [p.name for p in pp.list_images(tmp_path, limit=5)]
    assert first_five == [f"frame_{i:02d}.png" for i in range(5)]  # without a seed: the first ones by name

    sample = [p.name for p in pp.list_images(tmp_path, limit=5, seed=1)]
    assert len(set(sample)) == 5 and sample == sorted(sample)
    assert sample != first_five  # not just the start of the folder
    assert sample == [p.name for p in pp.list_images(tmp_path, limit=5, seed=1)]  # repeatable
    assert sample != [p.name for p in pp.list_images(tmp_path, limit=5, seed=2)]  # but depends on the seed
    assert len(pp.list_images(tmp_path, limit=100, seed=1)) == 30  # asking for more than exist returns all


def test_grayscale_and_rgba_images_are_converted_to_rgb(tmp_path):
    Image.fromarray(np.full((6, 6), 9, dtype=np.uint8)).save(tmp_path / "g.png")
    Image.fromarray(np.full((6, 6, 4), 9, dtype=np.uint8)).save(tmp_path / "rgba.png")
    assert pp.load_image_rgb(tmp_path / "g.png").shape == (6, 6, 3)
    assert pp.load_image_rgb(tmp_path / "rgba.png").shape == (6, 6, 3)


def test_list_images_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        pp.list_images(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="no images"):
        pp.list_images(tmp_path)


def test_synthetic_image_is_a_valid_uint8_picture():
    img = pp.synthetic_image(np.random.default_rng(0), width=64, height=48)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    assert img.std() > 5  # not a flat image
