"""Runtime device-selection utilities for CPU/GPU fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceResolution:
    """Resolved compute device and selection context."""

    selected_device: str
    requested_policy: str
    enable_gpu: bool
    gpu_available: bool
    fallback_used: bool
    reason: str


def resolve_torch_device(
    *,
    enable_gpu: bool = False,
    policy: str = "cpu",
) -> DeviceResolution:
    """Resolve torch device with safe CPU fallback.

    Parameters
    ----------
    enable_gpu:
        Global GPU switch. If ``False``, returns CPU regardless of availability.
    policy:
        One of ``cpu``, ``auto``, ``cuda``, ``mps``.

    Returns
    -------
    DeviceResolution
        Structured device selection details.
    """

    pol = str(policy).strip().lower() if policy else "cpu"
    if pol not in {"cpu", "auto", "cuda", "mps"}:
        pol = "cpu"

    if not enable_gpu:
        return DeviceResolution(
            selected_device="cpu",
            requested_policy=pol,
            enable_gpu=False,
            gpu_available=False,
            fallback_used=False,
            reason="GPU switch disabled; using CPU.",
        )

    try:
        import torch
    except Exception:
        return DeviceResolution(
            selected_device="cpu",
            requested_policy=pol,
            enable_gpu=True,
            gpu_available=False,
            fallback_used=True,
            reason="Torch GPU runtime not available; falling back to CPU.",
        )

    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if pol == "cpu":
        return DeviceResolution(
            selected_device="cpu",
            requested_policy=pol,
            enable_gpu=True,
            gpu_available=False,
            fallback_used=False,
            reason="Policy requested CPU.",
        )

    if pol == "cuda":
        if cuda_ok:
            return DeviceResolution(
                selected_device="cuda",
                requested_policy=pol,
                enable_gpu=True,
                gpu_available=True,
                fallback_used=False,
                reason="CUDA available and requested.",
            )
        return DeviceResolution(
            selected_device="cpu",
            requested_policy=pol,
            enable_gpu=True,
            gpu_available=False,
            fallback_used=True,
            reason="CUDA requested but unavailable; falling back to CPU.",
        )

    if pol == "mps":
        if mps_ok:
            return DeviceResolution(
                selected_device="mps",
                requested_policy=pol,
                enable_gpu=True,
                gpu_available=True,
                fallback_used=False,
                reason="MPS available and requested.",
            )
        return DeviceResolution(
            selected_device="cpu",
            requested_policy=pol,
            enable_gpu=True,
            gpu_available=False,
            fallback_used=True,
            reason="MPS requested but unavailable; falling back to CPU.",
        )

    # policy == auto
    if cuda_ok:
        return DeviceResolution(
            selected_device="cuda",
            requested_policy=pol,
            enable_gpu=True,
            gpu_available=True,
            fallback_used=False,
            reason="Auto policy selected CUDA.",
        )
    if mps_ok:
        return DeviceResolution(
            selected_device="mps",
            requested_policy=pol,
            enable_gpu=True,
            gpu_available=True,
            fallback_used=False,
            reason="Auto policy selected MPS.",
        )
    return DeviceResolution(
        selected_device="cpu",
        requested_policy=pol,
        enable_gpu=True,
        gpu_available=False,
        fallback_used=True,
        reason="No compatible GPU runtime found; falling back to CPU.",
    )
