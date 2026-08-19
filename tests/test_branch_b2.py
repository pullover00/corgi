"""Tests for the consolidation logic kept in ocmask.stages.branch_b2.

The original branch_b2.py also carried a reciprocal cycle-consistency
matcher and an internal-feature rigid-pose estimator for a diagnostic that
never fed a shipped prediction; those (and their tests) were dropped when
this repository was extracted -- see the module docstring in
ocmask/stages/branch_b2.py.
"""

from __future__ import annotations

import numpy as np

from ocmask.stages.branch_b2 import consolidate_hypotheses
from ocmask.types import ObjectMask


def _mask(y0: int, x0: int, size: int) -> np.ndarray:
    mask = np.zeros((40, 40), bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


def _object(mask: np.ndarray, proposal_id: int) -> ObjectMask:
    return ObjectMask(mask=mask, metadata={"automatic_proposal_id": proposal_id})


def test_nested_fragments_consolidate_but_adjacent_objects_do_not() -> None:
    outer = _mask(5, 5, 15)
    inner = _mask(8, 8, 7)
    adjacent = _mask(5, 22, 10)
    hypotheses = consolidate_hypotheses(
        [_object(outer, 1), _object(inner, 2), _object(adjacent, 3)],
        [outer, None, adjacent],
        [
            {"rejection_reasons": []},
            {"rejection_reasons": ["object_absent"]},
            {"rejection_reasons": []},
        ],
    )
    assert len(hypotheses) == 2
    assert hypotheses[0].proposal_ids == (1, 2)
    assert hypotheses[0].track_status == "present"
    assert np.array_equal(hypotheses[0].mask, outer)
    assert hypotheses[1].proposal_ids == (3,)


def test_absent_requires_every_fragment_to_report_absence() -> None:
    outer = _mask(5, 5, 15)
    inner = _mask(8, 8, 7)
    ambiguous = consolidate_hypotheses(
        [_object(outer, 1), _object(inner, 2)],
        [None, None],
        [{"rejection_reasons": ["object_absent"]}, {"rejection_reasons": ["low_overlap"]}],
    )[0]
    absent = consolidate_hypotheses(
        [_object(outer, 1), _object(inner, 2)],
        [None, None],
        [{"rejection_reasons": ["object_absent"]}, {"rejection_reasons": ["object_absent"]}],
    )[0]
    assert ambiguous.track_status == "ambiguous"
    assert absent.track_status == "absent"
