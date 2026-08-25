from __future__ import annotations

import numpy as np
import pytest

from ocmask.changesim import MetricAccumulator, decode_target_array


@pytest.mark.parametrize(
    "target",
    [
        np.array([[256]], dtype=np.uint16),
        np.array([[-1]], dtype=np.int16),
        np.array([[1.0]], dtype=np.float32),
    ],
)
def test_scalar_target_validation_happens_before_uint8_narrowing(
    target: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        decode_target_array(target)


def test_add_rejects_shape_mismatch_without_mutating_accumulator() -> None:
    accumulator = MetricAccumulator()
    prediction = np.zeros((2, 3), dtype=np.uint8)
    target = np.zeros((3, 2), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match=r"prediction=\(2, 3\), target=\(3, 2\)",
    ):
        accumulator.add(prediction, target)

    assert accumulator.count == 0
    np.testing.assert_array_equal(accumulator.confusion, np.zeros((6, 6), dtype=np.int64))


def test_add_accumulates_confusion_for_exactly_matching_shapes() -> None:
    accumulator = MetricAccumulator()
    target = np.array([[0, 0, 1], [2, 3, 5]], dtype=np.uint8)
    prediction = np.array([[0, 1, 1], [2, 5, 0]], dtype=np.uint8)

    accumulator.add(prediction, target)

    expected = np.zeros((6, 6), dtype=np.int64)
    expected[0, 0] = 1
    expected[0, 1] = 1
    expected[1, 1] = 1
    expected[2, 2] = 1
    expected[3, 5] = 1
    expected[5, 0] = 1
    assert accumulator.count == 1
    np.testing.assert_array_equal(accumulator.confusion, expected)


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (np.array([[0, 255]], dtype=np.uint8), "outside the canonical 0..5"),
        (np.array([[0.0, 1.0]], dtype=np.float32), "integer dtype"),
    ],
)
def test_add_rejects_invalid_prediction_labels_without_ignoring_pixels(
    prediction: np.ndarray, message: str
) -> None:
    accumulator = MetricAccumulator()

    with pytest.raises(ValueError, match=message):
        accumulator.add(prediction, np.zeros((1, 2), dtype=np.uint8))

    assert accumulator.count == 0
    assert accumulator.confusion.sum() == 0


def test_compute_uses_aggregate_one_vs_rest_iou_definitions() -> None:
    accumulator = MetricAccumulator()
    target = np.array(
        [
            [0, 0, 1, 1, 2],
            [2, 3, 3, 5, 5],
        ],
        dtype=np.uint8,
    )
    prediction = np.array(
        [
            [0, 1, 1, 0, 2],
            [3, 3, 5, 5, 0],
        ],
        dtype=np.uint8,
    )
    accumulator.add(prediction[:, :3], target[:, :3])
    accumulator.add(prediction[:, 3:], target[:, 3:])

    metrics = accumulator.compute()

    assert accumulator.count == 2
    assert metrics["multiclass"]["unchanged"]["iou"] == pytest.approx(1 / 4)
    assert metrics["multiclass"]["added"]["iou"] == pytest.approx(1 / 3)
    assert metrics["multiclass"]["removed"]["iou"] == pytest.approx(1 / 2)
    assert metrics["multiclass"]["moved"]["iou"] == pytest.approx(1 / 3)
    assert metrics["multiclass"]["replaced"]["iou"] == pytest.approx(1 / 3)
    assert metrics["multiclass_miou"] == pytest.approx(7 / 20)
    assert metrics["multiclass_macro_f1"] == pytest.approx(77 / 150)

    assert metrics["binary"]["unchanged"]["iou"] == pytest.approx(1 / 4)
    assert metrics["binary"]["changed"]["iou"] == pytest.approx(2 / 3)
    assert metrics["binary_miou"] == pytest.approx(11 / 24)
    assert metrics["binary_macro_f1"] == pytest.approx(3 / 5)
