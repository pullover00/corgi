from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from ocmask.adapters.sam2 import Sam2Adapter
from ocmask.numerics import (
    NUMERICAL_POLICY_VERSION,
    apply_post_reconstruction_numerics,
    capture_torch_numerical_state,
    close_sam3_autocast_context,
    finalize_pair_process,
    numerical_policy,
    restore_torch_numerical_state,
)
from ocmask.stages.sam3_identity_location import Sam3FeatureExtractor
from ocmask.stages.sam3_proposals import Sam3AutomaticMaskGenerator


class _FakeMatmul:
    """Model PyTorch's coupled TF32/float32-precision controls."""

    def __init__(self, owner: "_FakeTorch") -> None:
        self.owner = owner

    @property
    def allow_tf32(self) -> bool:
        return self.owner._matmul_allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, enabled: bool) -> None:
        self.owner._matmul_allow_tf32 = bool(enabled)
        # Directly setting this compatibility flag collapses ``medium`` to
        # ``high`` in real PyTorch. The precision API can retain that detail.
        self.owner._matmul_precision = "high" if enabled else "highest"


class _FakeCuda:
    def __init__(self) -> None:
        self.available = True
        self.major = 8
        self.empty_cache_calls = 0
        self.empty_cache_error = False
        self.manual_seed_all_calls: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def get_device_properties(self, _index: int):
        return SimpleNamespace(major=self.major)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1
        if self.empty_cache_error:
            raise RuntimeError("synthetic empty-cache failure")

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)


class _FakeContext:
    def __init__(self, *, exit_error: bool = False) -> None:
        self.exit_error = exit_error
        self.enter_calls = 0
        self.exit_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.exit_calls += 1
        if self.exit_error:
            raise RuntimeError("synthetic autocast-exit failure")


class _FakeTorch:
    def __init__(self) -> None:
        self._matmul_allow_tf32 = False
        self._matmul_precision = "highest"
        self._cuda_autocast_enabled = False
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=_FakeMatmul(self)),
            cudnn=SimpleNamespace(
                allow_tf32=True,
                benchmark=False,
                deterministic=False,
            ),
        )
        self.cuda = _FakeCuda()
        self.bfloat16 = object()
        self.autocast_calls: list[tuple[tuple, dict]] = []
        self.autocast_contexts: list[_FakeContext] = []
        self.manual_seed_calls: list[int] = []

    def is_autocast_enabled(self, device_type: str) -> bool:
        assert device_type == "cuda"
        return self._cuda_autocast_enabled

    def set_autocast_enabled(self, device_type: str, enabled: bool) -> None:
        assert device_type == "cuda"
        self._cuda_autocast_enabled = bool(enabled)

    def get_float32_matmul_precision(self) -> str:
        return self._matmul_precision

    def set_float32_matmul_precision(self, precision: str) -> None:
        assert precision in {"highest", "high", "medium"}
        self._matmul_precision = precision
        self._matmul_allow_tf32 = precision != "highest"

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def autocast(self, *args, **kwargs):
        self.autocast_calls.append((args, kwargs))
        context = _FakeContext()
        self.autocast_contexts.append(context)
        return context


def _set_distinct_initial_state(torch: _FakeTorch):
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_autocast_enabled("cuda", False)
    return capture_torch_numerical_state(torch)


def _mutate_like_sam3(torch: _FakeTorch) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_autocast_enabled("cuda", True)


def _make_sam3_owner(kind: str, source) -> object:
    if kind == "proposals":
        return Sam3AutomaticMaskGenerator("unused.pt", source=source)
    return Sam3FeatureExtractor(source, "unused.pt")


def test_restore_preserves_medium_matmul_precision() -> None:
    torch = _FakeTorch()
    initial = _set_distinct_initial_state(torch)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_autocast_enabled("cuda", True)

    restore_torch_numerical_state(initial, torch)

    assert capture_torch_numerical_state(torch) == initial


def test_post_reconstruction_policy_matches_fresh_stage1_without_other_changes() -> None:
    torch = _FakeTorch()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_autocast_enabled("cuda", False)

    state = apply_post_reconstruction_numerics(torch)

    assert state.cuda_matmul_allow_tf32 is True
    assert state.float32_matmul_precision == "high"
    assert state.cudnn_allow_tf32 is False
    assert state.cudnn_benchmark is True
    assert state.cudnn_deterministic is True
    assert state.cuda_autocast_enabled is False


def test_fingerprint_policy_records_both_reconstruction_boundary_states() -> None:
    policy = numerical_policy(2026)

    assert policy["version"] == NUMERICAL_POLICY_VERSION
    assert policy["cuda_matmul_allow_tf32_before_mast3r"] is False
    assert policy["float32_matmul_precision_before_mast3r"] == "highest"
    assert policy["cuda_matmul_allow_tf32_after_reconstruction"] is True
    assert policy["float32_matmul_precision_after_reconstruction"] == "high"


def test_finalize_pair_process_allows_sam2_tf32_delta_then_restores() -> None:
    torch = _FakeTorch()
    initial = _set_distinct_initial_state(torch)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    post_inference, restored = finalize_pair_process(initial, torch)

    assert post_inference != initial
    assert post_inference.cuda_autocast_enabled is False
    assert restored == initial
    assert capture_torch_numerical_state(torch) == initial


def test_finalize_pair_process_rejects_autocast_after_restoring() -> None:
    torch = _FakeTorch()
    initial = _set_distinct_initial_state(torch)
    torch.set_autocast_enabled("cuda", True)

    with pytest.raises(RuntimeError, match="ambient CUDA autocast"):
        finalize_pair_process(initial, torch)

    assert capture_torch_numerical_state(torch) == initial


def test_sam3_autocast_close_is_idempotent_after_exit_failure() -> None:
    context = _FakeContext(exit_error=True)
    model = SimpleNamespace(bf16_context=context)
    predictor = SimpleNamespace(model=model)

    with pytest.raises(RuntimeError, match="autocast-exit"):
        close_sam3_autocast_context(predictor)
    close_sam3_autocast_context(predictor)

    assert context.exit_calls == 1
    assert model.bf16_context is None


@pytest.mark.parametrize("kind", ["proposals", "features"])
@pytest.mark.parametrize("failure_point", ["context", "empty_cache"])
def test_sam3_release_restores_state_despite_cleanup_failure(
    monkeypatch, tmp_path, kind: str, failure_point: str
) -> None:
    torch = _FakeTorch()
    initial = _set_distinct_initial_state(torch)
    _mutate_like_sam3(torch)
    torch.cuda.empty_cache_error = failure_point == "empty_cache"
    monkeypatch.setitem(sys.modules, "torch", torch)

    owner = _make_sam3_owner(kind, tmp_path)
    context = _FakeContext(exit_error=failure_point == "context")
    model = SimpleNamespace(bf16_context=context)
    owner._model = object()
    owner._predictor = SimpleNamespace(model=model)
    owner._processor = object()
    owner._numerical_state = initial

    expected = "autocast-exit" if failure_point == "context" else "empty-cache"
    with pytest.raises(RuntimeError, match=expected):
        owner.release()

    assert capture_torch_numerical_state(torch) == initial
    assert owner._model is None
    assert owner._predictor is None
    assert owner._processor is None
    assert owner._numerical_state is None
    assert context.exit_calls == 1
    assert model.bf16_context is None
    assert torch.cuda.empty_cache_calls == 1

    # Resource ownership was detached before cleanup, so retrying cannot pop
    # the context or restore a potentially unrelated later numerical scope.
    owner.release()
    assert context.exit_calls == 1
    assert torch.cuda.empty_cache_calls == 1


@pytest.mark.parametrize("kind", ["proposals", "features"])
def test_sam3_import_failure_restores_state_and_source_path(
    monkeypatch, tmp_path, kind: str
) -> None:
    source = tmp_path / "sam3-checkout"
    (source / "sam3").mkdir(parents=True)
    torch = _FakeTorch()
    initial = _set_distinct_initial_state(torch)
    monkeypatch.setitem(sys.modules, "torch", torch)
    original_import = builtins.__import__

    def fail_sam3_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sam3.model.sam3_image_processor":
            _mutate_like_sam3(torch)
            raise RuntimeError("synthetic SAM3 import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_sam3_import)
    original_path = list(sys.path)
    owner = _make_sam3_owner(kind, source)

    with pytest.raises(RuntimeError, match="SAM3 import failure"):
        owner.load()

    assert capture_torch_numerical_state(torch) == initial
    assert sys.path == original_path
    assert owner._model is None
    assert owner._predictor is None
    assert owner._processor is None
    assert owner._numerical_state is None


def test_sam2_mixed_precision_context_is_scoped_bfloat16(monkeypatch) -> None:
    torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    adapter = Sam2Adapter(
        {
            "device": "cuda:0",
            "mixed_precision": False,
            "sam2": {"mixed_precision": True},
        }
    )

    context = adapter._inference_context()

    assert context is torch.autocast_contexts[0]
    assert torch.autocast_calls == [
        (("cuda",), {"dtype": torch.bfloat16}),
    ]

    cpu_adapter = Sam2Adapter(
        {"device": "cpu", "sam2": {"mixed_precision": True}}
    )
    with cpu_adapter._inference_context() as value:
        assert value is None
    assert len(torch.autocast_calls) == 1
