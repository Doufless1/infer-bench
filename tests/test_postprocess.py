import numpy as np
import pytest

from infer_bench import postprocess as pp


def test_softmax_sums_to_one_and_is_stable_for_large_values():
    out = pp.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert out.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(out)) and out.argmax() == 1


def test_topk_indices_best_first_and_k_larger_than_size():
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    assert pp.topk_indices(scores, 2).tolist() == [1, 3]
    assert pp.topk_indices(scores, 10).tolist() == [1, 3, 2, 0]


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    iou = pp.iou_matrix(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(1 / 3)  # intersection 50, union 150
    assert iou[2] == 0.0


def test_iou_matrix_with_empty_inputs():
    assert pp.iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.5], dtype=np.float32)
    assert pp.nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_threshold_decides_what_counts_as_overlap():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=np.float32)  # IoU = 1/3
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert pp.nms(boxes, scores, 0.5).tolist() == [0, 1]
    assert pp.nms(boxes, scores, 0.2).tolist() == [0]


def _yolo_output(rows, n_classes=3, n_anchors=None):
    """Build a (1, 4+nc, N) tensor from rows of (cx, cy, w, h, class_id, score).

    N is padded to at least 50 anchors: like every real YOLO export, there are many more anchors
    than channels, which is what the automatic layout detection relies on.
    """
    n = n_anchors or max(len(rows), 50)
    out = np.zeros((1, 4 + n_classes, n), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_decode_yolo_converts_centre_boxes_and_filters_by_confidence():
    out = _yolo_output([(50, 40, 20, 10, 1, 0.9), (200, 200, 10, 10, 0, 0.1)])
    dets = pp.decode_yolo(out, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets.boxes[0].tolist() == [40.0, 35.0, 60.0, 45.0]  # xywh -> xyxy
    assert dets.class_ids.tolist() == [1] and dets.scores[0] == pytest.approx(0.9)


def test_decode_yolo_nms_is_class_aware():
    same_place = [(50, 50, 20, 20, 0, 0.9), (51, 51, 20, 20, 0, 0.8), (50, 50, 20, 20, 2, 0.7)]
    dets = pp.decode_yolo(_yolo_output(same_place), iou_threshold=0.45)
    assert sorted(dets.class_ids.tolist()) == [0, 2]  # the duplicate of class 0 is suppressed, class 2 survives


def test_decode_yolo_accepts_either_layout():
    out = _yolo_output([(50, 40, 20, 10, 1, 0.9)], n_anchors=100)
    a = pp.decode_yolo(out)
    b = pp.decode_yolo(out[0].T[None, ...])  # (1, N, 4+nc): the other layout, found automatically
    c = pp.decode_yolo(out[0].T, layout="channels_last")
    for other in (b, c):
        np.testing.assert_allclose(a.boxes, other.boxes)


def test_decode_yolo_respects_max_det_and_orders_by_score():
    rows = [(20 * i + 10, 10, 10, 10, 0, 0.5 + i * 0.01) for i in range(10)]
    dets = pp.decode_yolo(_yolo_output(rows), max_det=3)
    assert len(dets) == 3
    assert dets.scores.tolist() == sorted(dets.scores.tolist(), reverse=True)


def test_decode_yolo_with_nothing_above_the_threshold():
    dets = pp.decode_yolo(_yolo_output([(10, 10, 5, 5, 0, 0.1)]))
    assert len(dets) == 0 and dets.boxes.shape == (0, 4)


@pytest.mark.parametrize("bad", [np.zeros((2, 3, 4, 5)), np.zeros(10), np.zeros((1, 4, 10))])
def test_decode_yolo_rejects_malformed_outputs(bad):
    with pytest.raises(ValueError):
        pp.decode_yolo(bad)


def test_decode_yolo_rejects_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        pp.decode_yolo(_yolo_output([(1, 1, 1, 1, 0, 0.9)]), layout="sideways")


def _det(boxes, classes, scores=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores if scores is not None else [0.9] * len(boxes), dtype=np.float32)
    return pp.Detections(boxes, scores, np.asarray(classes, dtype=np.int64))


def test_match_detections_identical_sets():
    d = _det([[0, 0, 10, 10], [20, 20, 40, 40]], [0, 1])
    assert pp.match_detections(d, d) == (2, 0, 0)


def test_match_detections_counts_false_positives_and_negatives():
    ref = _det([[0, 0, 10, 10], [20, 20, 40, 40]], [0, 1])
    cand = _det([[0, 0, 10, 10], [100, 100, 120, 120]], [0, 1])
    assert pp.match_detections(ref, cand) == (1, 1, 1)


def test_match_detections_needs_the_same_class_and_enough_overlap():
    ref = _det([[0, 0, 10, 10]], [0])
    assert pp.match_detections(ref, _det([[0, 0, 10, 10]], [1])) == (0, 1, 1)  # wrong class
    assert pp.match_detections(ref, _det([[6, 0, 16, 10]], [0])) == (0, 1, 1)  # IoU 0.25 < 0.5


def test_match_detections_does_not_reuse_a_reference_box():
    ref = _det([[0, 0, 10, 10]], [0])
    cand = _det([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], scores=[0.9, 0.8])
    assert pp.match_detections(ref, cand) == (1, 1, 0)  # the duplicate is a false positive


def test_match_detections_with_empty_sets():
    empty = pp.Detections.empty()
    d = _det([[0, 0, 10, 10]], [0])
    assert pp.match_detections(empty, d) == (0, 1, 0)
    assert pp.match_detections(d, empty) == (0, 0, 1)
    assert pp.match_detections(empty, empty) == (0, 0, 0)
