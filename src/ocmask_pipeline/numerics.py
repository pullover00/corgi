"""Numerical-state controls for reproducible model inference.

Several upstream model packages mutate process-global PyTorch state.  In
particular, MASt3R/CroCo enables CUDA matmul TF32 at import time, while the
SAM3 interactive predictor enters a CUDA autocast context without leaving it.
Those side effects must be explicit at stage boundaries and must never leak
from one image pair into the next.

The helpers here keep that state explicit and, importantly, do not import
PyTorch at module import time.  CPU-only tooling and unit tests can therefore
import the rest of :mod:`ocmask` without requiring a model environment.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


NUMERICAL_POLICY_VERSION = "clean-pair-post-reconstruction-scoped-sam3-v2"


@dataclass(frozen=True)
class TorchNumericalState:
    """The process-global PyTorch switches touched by this pipeline/SAM3."""

    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_autocast_enabled: bool
    float32_matmul_precision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_torch_numerical_state(torch_module=None) -> TorchNumericalState:
    """Capture numerical switches without changing them."""

    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    return TorchNumericalState(
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cuda_autocast_enabled=bool(torch.is_autocast_enabled("cuda")),
        float32_matmul_precision=str(torch.get_float32_matmul_precision()),
    )


def restore_torch_numerical_state(
    state: TorchNumericalState, torch_module=None
) -> None:
    """Restore a state captured by :func:`capture_torch_numerical_state`."""

    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    # Keep this assignment before ``set_float32_matmul_precision``. PyTorch's
    # precision setter also updates the CUDA matmul TF32 flag; applying the
    # captured precision last is what preserves its three-state
    # highest/high/medium policy rather than collapsing ``medium`` to ``high``.
    torch.backends.cuda.matmul.allow_tf32 = state.cuda_matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = state.cudnn_allow_tf32
    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.set_float32_matmul_precision(state.float32_matmul_precision)
    # SAM3 enters autocast manually and does not exit it. Setting this flag is
    # the final safety net after the owning context has been closed.
    torch.set_autocast_enabled("cuda", state.cuda_autocast_enabled)


def enable_sam3_numerics(torch_module=None) -> None:
    """Apply SAM3's intended Ampere+ TF32 policy within a scoped lifetime."""

    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def close_sam3_autocast_context(predictor: object | None) -> None:
    """Close the known leaked context owned by a SAM3 image predictor.

    ``SAM3InteractiveImagePredictor.model`` is the tracker object whose
    constructor stores and enters ``bf16_context``.  This deliberately uses
    only that public object relationship rather than recursively walking a
    large ``nn.Module`` graph.
    """

    model = getattr(predictor, "model", None)
    context = getattr(model, "bf16_context", None)
    if context is not None:
        # Detach first so cleanup is idempotent even when ``__exit__`` itself
        # raises. A second release must never pop a different nested autocast
        # scope from PyTorch's thread-local stack.
        model.bf16_context = None
        context.__exit__(None, None, None)


def exit_sam3_numerical_scope(
    predictor: object | None,
    state: TorchNumericalState | None,
    torch_module=None,
) -> None:
    """Close SAM3's leaked autocast scope and restore process-global state.

    Restoration is attempted even if the upstream context manager raises
    while exiting. If both operations fail, the restoration failure is
    raised with the context failure retained as its cause because the former
    means process-global numerical state may still be corrupted.
    """

    context_error: BaseException | None = None
    try:
        close_sam3_autocast_context(predictor)
    except BaseException as exc:  # cleanup must also survive cancellation
        context_error = exc

    try:
        if state is not None:
            restore_torch_numerical_state(state, torch_module)
    except BaseException as restore_error:
        if context_error is not None:
            raise restore_error from context_error
        raise

    if context_error is not None:
        raise context_error


def initialize_pair_process(seed: int, torch_module=None) -> TorchNumericalState:
    """Initialize the clean worker-entry state for one image pair.

    Before MASt3R/CroCo or SAM3 is imported, the pinned PyTorch environment
    has CUDA matmul TF32 disabled, cuDNN TF32 enabled, matmul precision
    ``highest``, and no ambient CUDA autocast context.  Pair workers also
    receive the same explicit seed, making results independent of manifest
    order/resume.  The post-reconstruction boundary below records CroCo's
    deliberate transition to TF32/high separately.
    """

    # Must be set before CUDA creates a cuBLAS workspace. It is harmless for
    # the historical kernels that do not require deterministic algorithms.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(int(seed))
    np.random.seed(int(seed))

    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    # This is PyTorch's historical default in the validated environment and
    # is intentionally not conflated with CUDA matmul TF32.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("highest")
    torch.set_autocast_enabled("cuda", False)
    return capture_torch_numerical_state(torch)


def apply_post_reconstruction_numerics(torch_module=None) -> TorchNumericalState:
    """Enter the fused-pipeline compatibility state used after stage 1.

    Importing MASt3R's CroCo implementation in a fresh stage-1 pass changes
    PyTorch's process-global CUDA matmul policy from
    ``allow_tf32=False``/``precision='highest'`` to
    ``allow_tf32=True``/``precision='high'``.  Reusing a cached
    reconstruction skips that side effect, which used to make every later
    model stage depend on whether stage 1 was loaded or computed.

    Apply that observed post-reconstruction state at the stage boundary for
    *both* paths.  This deliberately does not change the other numerical
    switches: the clean pair-process policy still controls MASt3R's starting
    state, while cuDNN/autocast behavior remains scoped to its owning stage.
    """

    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    # The precision setter is the authoritative PyTorch API and also enables
    # CUDA matmul TF32. Assign the compatibility flag as well so the intended
    # state remains explicit for supported PyTorch versions/backends.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    return capture_torch_numerical_state(torch)


def finalize_pair_process(
    initial_state: TorchNumericalState, torch_module=None
) -> tuple[TorchNumericalState, TorchNumericalState]:
    """Restore a disposable worker after inference and reject autocast leaks.

    Model imports/stages are allowed to change TF32/matmul flags after the
    clean worker-entry boundary; pair-process isolation prevents that later
    state from reaching the next pair. An ambient CUDA autocast scope is
    different: it indicates an unclosed context and is rejected. In every
    case restoration is attempted first so callers can safely inspect/tear
    down the process.
    """

    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    post_inference = capture_torch_numerical_state(torch)
    restore_torch_numerical_state(initial_state, torch)
    restored = capture_torch_numerical_state(torch)
    if restored != initial_state:
        raise RuntimeError(
            "Could not restore process-global numerical state after pair inference"
        )
    if post_inference.cuda_autocast_enabled:
        raise RuntimeError("A model stage leaked an ambient CUDA autocast context")
    return post_inference, restored


def numerical_policy(seed: int) -> dict[str, Any]:
    """Return the stable policy fields included in evaluation fingerprints."""

    return {
        "version": NUMERICAL_POLICY_VERSION,
        "seed": int(seed),
        "python_hash_seed": int(seed),
        "cublas_workspace_config": ":4096:8",
        "pair_process_isolation": True,
        "cuda_matmul_allow_tf32_before_mast3r": False,
        "cuda_matmul_allow_tf32_after_reconstruction": True,
        "cudnn_allow_tf32": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision_before_mast3r": "highest",
        "float32_matmul_precision_after_reconstruction": "high",
        "ambient_cuda_autocast_before_mast3r": False,
        "sam3_tf32_scoped": True,
        "sam3_autocast_closed_on_release": True,
    }
