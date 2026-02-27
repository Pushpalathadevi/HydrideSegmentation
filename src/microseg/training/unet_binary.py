"""UNet-style binary segmentation training with progress, tracking, and resume support."""

from __future__ import annotations

import html
import hashlib
import json
import logging
import os
import platform
import random
import socket
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.microseg.core import resolve_torch_device
from src.microseg.corrections.classes import binary_remapped_foreground_values, normalize_binary_index_mask
from src.microseg.data import InputPolicyConfig, apply_input_policy, resolve_collate_fn
from src.microseg.evaluation.hydride_metrics import scientific_distance_metrics
from src.microseg.plugins import (
    resolve_bundle_paths,
    resolve_pretrained_record,
    validate_pretrained_registry,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_logger() -> logging.Logger:
    logger = logging.getLogger("microseg.training.unet_binary")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    _append_jsonl_logged(path, payload, logger=None)


class _Heartbeat:
    def __init__(self, interval_seconds: float, logger: logging.Logger) -> None:
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.logger = logger
        self._last = time.perf_counter()

    def beat(self, message: str, *args: Any) -> None:
        now = time.perf_counter()
        if (now - self._last) >= self.interval_seconds:
            self.logger.info(message, *args)
            self._last = now

    def force(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)
        self._last = time.perf_counter()


def _append_jsonl_logged(
    path: Path,
    payload: dict[str, Any],
    *,
    logger: logging.Logger | None,
    context: str = "jsonl_append",
    retries: int = 2,
    retry_backoff_seconds: float = 0.25,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    for attempt in range(1, max(1, int(retries)) + 2):
        try:
            open_start = time.perf_counter()
            if logger is not None:
                logger.info("%s | JSONL_OPEN_START path=%s attempt=%d", context, path, attempt)
            with path.open("a", encoding="utf-8") as f:
                if logger is not None:
                    logger.info(
                        "%s | JSONL_OPEN_END path=%s elapsed=%s",
                        context,
                        path,
                        _format_seconds(time.perf_counter() - open_start),
                    )
                write_start = time.perf_counter()
                if logger is not None:
                    logger.info("%s | JSONL_WRITE_START path=%s bytes=%d", context, path, len(line.encode("utf-8")))
                f.write(line)
                f.flush()
                if logger is not None:
                    logger.info("%s | JSONL_FSYNC_START path=%s", context, path)
                os.fsync(f.fileno())
                if logger is not None:
                    logger.info(
                        "%s | JSONL_WRITE_END path=%s elapsed=%s",
                        context,
                        path,
                        _format_seconds(time.perf_counter() - write_start),
                    )
            return
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "%s | JSONL_APPEND_RETRY path=%s attempt=%d error=%s",
                    context,
                    path,
                    attempt,
                    exc,
                )
            if attempt > int(retries):
                raise
            time.sleep(float(retry_backoff_seconds) * attempt)


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _code_version() -> str:
    try:
        from src.microseg.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _config_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"missing images/masks under split dir: {split_dir}")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*")):
        if not img.is_file():
            continue
        msk = masks_dir / img.name
        if msk.exists() and msk.is_file():
            pairs.append((img, msk))
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found in {split_dir}")
    return pairs


def _warn_binary_mask_remap_values(
    pairs: list[tuple[Path, Path]],
    *,
    binary_mask_normalization: str,
    logger: logging.Logger,
) -> None:
    mode = str(binary_mask_normalization).strip().lower()
    if mode not in {"two_value_zero_background", "nonzero_foreground"}:
        return
    remapped: set[int] = set()
    for _img_path, mask_path in pairs:
        arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        remapped.update(binary_remapped_foreground_values(arr, mode=mode))
    if remapped:
        logger.warning(
            "binary_mask_normalization=%s remapped non-zero mask values %s to foreground class 1; "
            "zero remains background.",
            mode,
            sorted(remapped),
        )


def _to_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _normalize_fixed_samples(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    sep = "|" if "|" in text else ","
    return [item.strip() for item in text.split(sep) if item.strip()]


def _select_tracking_pairs(
    all_pairs: list[tuple[Path, Path]],
    *,
    fixed_names: list[str],
    total_samples: int,
    seed: int,
    epoch: int,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    if total_samples <= 0 or not all_pairs:
        return [], []

    pair_by_name = {img.name: pair for pair in all_pairs for img in [pair[0]]}
    selected: list[tuple[Path, Path]] = []
    missing_fixed: list[str] = []

    for name in fixed_names:
        pair = pair_by_name.get(name)
        if pair is None:
            missing_fixed.append(name)
            continue
        selected.append(pair)

    remaining = [pair for pair in all_pairs if pair not in selected]
    need_random = max(0, total_samples - len(selected))
    if need_random > 0 and remaining:
        rng = random.Random(int(seed) + int(epoch))
        selected.extend(rng.sample(remaining, k=min(need_random, len(remaining))))
    return selected[:total_samples], missing_fixed


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _unwrap_torch_module(model_obj: Any) -> Any:
    return getattr(model_obj, "model", model_obj)


def _model_parameter_counts(model_obj: Any) -> tuple[int | None, int | None]:
    total = 0
    trainable = 0
    try:
        iterator = model_obj.parameters()
    except Exception:
        return None, None
    for param in iterator:
        try:
            count = int(param.numel())
        except Exception:
            continue
        total += count
        if bool(getattr(param, "requires_grad", False)):
            trainable += count
    if total <= 0:
        return None, None
    return total, trainable


def _state_dict_tensor_stats(state_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}
    total = 0.0
    total_sq = 0.0
    value_count = 0
    tensor_count = 0
    min_value: float | None = None
    max_value: float | None = None
    for value in state_dict.values():
        if not isinstance(value, torch.Tensor):
            continue
        if value.numel() == 0:
            continue
        vec = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if vec.numel() == 0:
            continue
        tensor_count += 1
        value_count += int(vec.numel())
        total += float(vec.sum().item())
        total_sq += float((vec * vec).sum().item())
        vmin = float(vec.min().item())
        vmax = float(vec.max().item())
        min_value = vmin if min_value is None else min(min_value, vmin)
        max_value = vmax if max_value is None else max(max_value, vmax)
    if value_count <= 0:
        return {}
    mean = total / float(value_count)
    variance = max(0.0, (total_sq / float(value_count)) - (mean * mean))
    return {
        "tensor_count": int(tensor_count),
        "value_count": int(value_count),
        "mean": float(mean),
        "std": float(sqrt(variance)),
        "min": float(min_value if min_value is not None else 0.0),
        "max": float(max_value if max_value is not None else 0.0),
    }


def _model_weight_statistics(model_obj: Any) -> dict[str, Any]:
    try:
        state_dict = model_obj.state_dict()
    except Exception:
        return {}
    if not isinstance(state_dict, dict):
        return {}
    return _state_dict_tensor_stats(state_dict)


def _runtime_hardware_profile(device: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": int(os.cpu_count() or 0),
        "selected_device": str(device),
    }
    try:
        import torch

        profile["torch_version"] = str(torch.__version__)
        profile["torch_cuda_available"] = bool(torch.cuda.is_available())
        profile["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if str(device).startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(device)
            index = int(dev.index or 0)
            props = torch.cuda.get_device_properties(index)
            profile["gpu_index"] = index
            profile["gpu_name"] = str(props.name)
            profile["gpu_total_memory_mb"] = float(props.total_memory) / (1024.0 * 1024.0)
            profile["gpu_multiprocessor_count"] = int(props.multi_processor_count)
            profile["gpu_compute_capability"] = f"{int(props.major)}.{int(props.minor)}"
    except Exception:
        return profile

    if str(device).startswith("cuda"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode == 0:
                selected = profile.get("gpu_index")
                for line in proc.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 6:
                        continue
                    if selected is not None and str(parts[0]) != str(selected):
                        continue
                    profile["nvidia_smi_driver_version"] = parts[2]
                    try:
                        profile["nvidia_smi_memory_total_mb"] = float(parts[3])
                        profile["nvidia_smi_memory_used_mb"] = float(parts[4])
                        profile["nvidia_smi_gpu_utilization_pct"] = float(parts[5])
                    except Exception:
                        pass
                    break
        except Exception:
            pass
    return profile


def _current_gpu_peak_memory_mb(device: str) -> float | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        dev = torch.device(device)
        index = int(dev.index or 0)
        peak_bytes = float(torch.cuda.max_memory_allocated(index))
        if peak_bytes <= 0:
            return 0.0
        return peak_bytes / (1024.0 * 1024.0)
    except Exception:
        return None


def _output_numel(output: object) -> int | None:
    try:
        import torch
    except Exception:
        return None
    if isinstance(output, torch.Tensor):
        return int(output.numel())
    if isinstance(output, (list, tuple)):
        for item in output:
            n = _output_numel(item)
            if n is not None:
                return n
        return None
    if isinstance(output, dict):
        for item in output.values():
            n = _output_numel(item)
            if n is not None:
                return n
        return None
    return None


def _estimate_forward_flops_per_sample(
    model_obj: Any,
    *,
    image_height: int,
    image_width: int,
    device: str,
) -> int | None:
    try:
        import torch
    except Exception:
        return None
    if image_height <= 0 or image_width <= 0:
        return None
    module = _unwrap_torch_module(model_obj)
    if not isinstance(module, torch.nn.Module):
        return None

    total_flops = 0

    def conv_hook(mod: Any, _inputs: tuple[Any, ...], output: object) -> None:
        nonlocal total_flops
        out_numel = _output_numel(output)
        if out_numel is None or out_numel <= 0:
            return
        kernel_size = getattr(mod, "kernel_size", (1, 1))
        k_h = int(kernel_size[0]) if isinstance(kernel_size, tuple) else int(kernel_size)
        k_w = int(kernel_size[1]) if isinstance(kernel_size, tuple) else int(kernel_size)
        groups = max(1, int(getattr(mod, "groups", 1)))
        in_ch = int(getattr(mod, "in_channels", 1))
        kernel_ops = (k_h * k_w * max(1, in_ch // groups) * 2) + (1 if getattr(mod, "bias", None) is not None else 0)
        total_flops += int(out_numel * kernel_ops)

    def linear_hook(mod: Any, _inputs: tuple[Any, ...], output: object) -> None:
        nonlocal total_flops
        out_numel = _output_numel(output)
        if out_numel is None or out_numel <= 0:
            return
        in_features = max(1, int(getattr(mod, "in_features", 1)))
        total_flops += int(out_numel * (2 * in_features))

    def norm_hook(_mod: Any, _inputs: tuple[Any, ...], output: object) -> None:
        nonlocal total_flops
        out_numel = _output_numel(output)
        if out_numel is None or out_numel <= 0:
            return
        total_flops += int(out_numel * 2)

    hooks: list[Any] = []
    try:
        for submodule in module.modules():
            if isinstance(submodule, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                hooks.append(submodule.register_forward_hook(conv_hook))
            elif isinstance(submodule, torch.nn.Linear):
                hooks.append(submodule.register_forward_hook(linear_hook))
            elif isinstance(submodule, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                hooks.append(submodule.register_forward_hook(norm_hook))

        x = torch.zeros((1, 3, int(image_height), int(image_width)), dtype=torch.float32, device=device)
        was_training = bool(module.training)
        module.eval()
        with torch.no_grad():
            _ = model_obj(x)
        if was_training:
            module.train()
    except Exception:
        total_flops = 0
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except Exception:
                pass

    if total_flops <= 0:
        return None
    return int(total_flops)


@dataclass(frozen=True)
class _PretrainedInitBundle:
    """Resolved local pretrained-initialization bundle paths and metadata."""

    model_id: str
    architecture: str
    framework: str
    source: str
    source_revision: str
    source_url: str
    license: str
    citation_key: str
    citation: str
    citation_url: str
    bundle_dir: Path
    weights_path: Path
    weights_format: str
    metadata_path: Path | None = None


def _resolve_pretrained_bundle(
    *,
    init_mode: str,
    model_id: str,
    bundle_dir: str,
    registry_path: str,
    architecture: str,
    verify_sha256: bool,
) -> _PretrainedInitBundle | None:
    mode = str(init_mode).strip().lower()
    if mode in {"", "scratch", "none", "off"}:
        return None
    if mode not in {"local", "local_pretrained"}:
        raise ValueError(f"unsupported pretrained_init_mode={init_mode!r}; expected scratch|local")

    reg_path = str(registry_path).strip() or "pre_trained_weights/registry.json"
    if model_id:
        if bool(verify_sha256):
            report = validate_pretrained_registry(reg_path, verify_sha256=True)
            if not report.ok:
                joined = "; ".join(report.errors[:5])
                raise RuntimeError(f"pretrained registry validation failed: {joined}")
        rec = resolve_pretrained_record(model_id=str(model_id).strip(), registry_path=reg_path)
        if str(rec.architecture).strip() and str(rec.architecture).strip().lower() != str(architecture).strip().lower():
            raise ValueError(
                f"pretrained model '{rec.model_id}' architecture={rec.architecture!r} is incompatible with "
                f"requested training architecture={architecture!r}"
            )
        bundle_abs, weights_abs, metadata_abs = resolve_bundle_paths(rec, registry_path=reg_path)
        return _PretrainedInitBundle(
            model_id=rec.model_id,
            architecture=rec.architecture,
            framework=rec.framework,
            source=rec.source,
            source_revision=rec.source_revision,
            source_url=rec.source_url,
            license=rec.license,
            citation_key=rec.citation_key,
            citation=rec.citation,
            citation_url=rec.citation_url,
            bundle_dir=bundle_abs,
            weights_path=weights_abs,
            weights_format=rec.weights_format,
            metadata_path=metadata_abs,
        )

    bdir = Path(str(bundle_dir).strip()).expanduser()
    if not bdir.exists():
        raise FileNotFoundError(f"pretrained_bundle_dir does not exist: {bdir}")
    if not bdir.is_absolute():
        bdir = bdir.resolve()
    return _PretrainedInitBundle(
        model_id="ad_hoc_local_bundle",
        architecture=str(architecture).strip().lower(),
        framework="local",
        source="local_bundle",
        source_revision="n/a",
        source_url="",
        license="",
        citation_key="",
        citation="",
        citation_url="",
        bundle_dir=bdir,
        weights_path=bdir,
        weights_format="directory",
        metadata_path=None,
    )


class _SegPairDataset:
    """Dataset of image/mask path pairs for binary segmentation."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        binary_mask_normalization: str,
        input_policy_cfg: InputPolicyConfig,
        seed: int,
        is_train: bool,
        logger: logging.Logger,
    ) -> None:
        self.pairs = pairs
        self.binary_mask_normalization = str(binary_mask_normalization)
        self.input_policy_cfg = input_policy_cfg
        self.seed = int(seed)
        self.is_train = bool(is_train)
        self.logger = logger

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):  # noqa: ANN001
        import torch

        img_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask_idx = normalize_binary_index_mask(
            np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8),
            mode=self.binary_mask_normalization,
        )
        mask = (mask_idx > 0).astype(np.float32)

        x = torch.from_numpy(image.transpose(2, 0, 1))
        y = torch.from_numpy(mask[None, ...])
        orig_hw = tuple(int(v) for v in x.shape[-2:])
        rng = torch.Generator().manual_seed(self.seed + int(index)) if self.is_train else None
        x, y = apply_input_policy(x, y, self.input_policy_cfg, rng=rng, is_train=self.is_train)
        self.logger.debug(
            "input policy sample=%s idx=%d train=%s policy=%s original_hw=%s final_hw=%s",
            img_path.name,
            int(index),
            self.is_train,
            self.input_policy_cfg.input_policy,
            orig_hw,
            tuple(int(v) for v in x.shape[-2:]),
        )
        return x, y


class _DoubleConv:
    def __init__(self, in_ch: int, out_ch: int) -> None:
        import torch

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )


class _UNetBinaryModel:
    """Compact UNet-style model for binary segmentation."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        import torch

        self.torch = torch
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.inc = _DoubleConv(in_channels, base_channels).module
        self.down1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            _DoubleConv(base_channels, base_channels * 2).module,
        )
        self.down2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            _DoubleConv(base_channels * 2, base_channels * 4).module,
        )

        self.up1_t = torch.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1_c = _DoubleConv(base_channels * 4, base_channels * 2).module

        self.up2_t = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up2_c = _DoubleConv(base_channels * 2, base_channels).module

        self.out = torch.nn.Conv2d(base_channels, 1, kernel_size=1)

        self.model = torch.nn.ModuleDict(
            {
                "inc": self.inc,
                "down1": self.down1,
                "down2": self.down2,
                "up1_t": self.up1_t,
                "up1_c": self.up1_c,
                "up2_t": self.up2_t,
                "up2_c": self.up2_c,
                "out": self.out,
            }
        )

    def __call__(self, x):  # noqa: ANN001
        m = self.model
        x1 = m["inc"](x)
        x2 = m["down1"](x1)
        x3 = m["down2"](x2)

        u1 = m["up1_t"](x3)
        if u1.shape[-2:] != x2.shape[-2:]:
            u1 = self.torch.nn.functional.interpolate(u1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.torch.cat([u1, x2], dim=1)
        u1 = m["up1_c"](u1)

        u2 = m["up2_t"](u1)
        if u2.shape[-2:] != x1.shape[-2:]:
            u2 = self.torch.nn.functional.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.torch.cat([u2, x1], dim=1)
        u2 = m["up2_c"](u2)

        return m["out"](u2)

    def to(self, device: str) -> _UNetBinaryModel:
        self.model.to(device)
        return self

    def train(self) -> _UNetBinaryModel:
        self.model.train()
        return self

    def eval(self) -> _UNetBinaryModel:
        self.model.eval()
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True):
        return self.model.load_state_dict(state, strict=bool(strict))

    def parameters(self):
        return self.model.parameters()


class _TransUNetTinyModel:
    """UNet-like model with transformer encoder at bottleneck."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 16,
        transformer_depth: int = 2,
        transformer_num_heads: int = 4,
        transformer_mlp_ratio: float = 2.0,
        transformer_dropout: float = 0.0,
    ) -> None:
        import torch

        self.torch = torch
        self.in_channels = in_channels
        self.base_channels = base_channels

        bottleneck_channels = base_channels * 4
        self.inc = _DoubleConv(in_channels, base_channels).module
        self.down1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            _DoubleConv(base_channels, base_channels * 2).module,
        )
        self.down2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            _DoubleConv(base_channels * 2, bottleneck_channels).module,
        )

        ff = max(32, int(round(float(transformer_mlp_ratio) * bottleneck_channels)))
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=bottleneck_channels,
            nhead=transformer_num_heads,
            dim_feedforward=ff,
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=max(1, int(transformer_depth)))

        self.up1_t = torch.nn.ConvTranspose2d(bottleneck_channels, base_channels * 2, kernel_size=2, stride=2)
        self.up1_c = _DoubleConv(base_channels * 4, base_channels * 2).module
        self.up2_t = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up2_c = _DoubleConv(base_channels * 2, base_channels).module
        self.out = torch.nn.Conv2d(base_channels, 1, kernel_size=1)

        self.model = torch.nn.ModuleDict(
            {
                "inc": self.inc,
                "down1": self.down1,
                "down2": self.down2,
                "transformer": self.transformer,
                "up1_t": self.up1_t,
                "up1_c": self.up1_c,
                "up2_t": self.up2_t,
                "up2_c": self.up2_c,
                "out": self.out,
            }
        )

    def __call__(self, x):  # noqa: ANN001
        m = self.model
        x1 = m["inc"](x)
        x2 = m["down1"](x1)
        x3 = m["down2"](x2)

        b, c, h, w = x3.shape
        t = x3.flatten(2).transpose(1, 2)
        t = m["transformer"](t)
        x3 = t.transpose(1, 2).reshape(b, c, h, w)

        u1 = m["up1_t"](x3)
        if u1.shape[-2:] != x2.shape[-2:]:
            u1 = self.torch.nn.functional.interpolate(u1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.torch.cat([u1, x2], dim=1)
        u1 = m["up1_c"](u1)

        u2 = m["up2_t"](u1)
        if u2.shape[-2:] != x1.shape[-2:]:
            u2 = self.torch.nn.functional.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.torch.cat([u2, x1], dim=1)
        u2 = m["up2_c"](u2)

        return m["out"](u2)

    def to(self, device: str) -> _TransUNetTinyModel:
        self.model.to(device)
        return self

    def train(self) -> _TransUNetTinyModel:
        self.model.train()
        return self

    def eval(self) -> _TransUNetTinyModel:
        self.model.eval()
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True):
        return self.model.load_state_dict(state, strict=bool(strict))

    def parameters(self):
        return self.model.parameters()


class _SegFormerMiniModel:
    """Small patch-transformer segmentation model for binary masks."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 16,
        transformer_depth: int = 2,
        transformer_num_heads: int = 4,
        transformer_mlp_ratio: float = 2.0,
        transformer_dropout: float = 0.0,
        patch_size: int = 4,
    ) -> None:
        import torch

        self.torch = torch
        embed_dim = max(16, base_channels * 4)
        p = max(2, int(patch_size))

        self.patch_embed = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=p, stride=p, bias=False)
        self.patch_norm = torch.nn.BatchNorm2d(embed_dim)
        self.patch_act = torch.nn.ReLU(inplace=True)

        ff = max(32, int(round(float(transformer_mlp_ratio) * embed_dim)))
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_num_heads,
            dim_feedforward=ff,
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=max(1, int(transformer_depth)))

        self.decode = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, base_channels * 2, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(base_channels * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.out = torch.nn.Conv2d(base_channels, 1, kernel_size=1)
        self.model = torch.nn.ModuleDict(
            {
                "patch_embed": self.patch_embed,
                "patch_norm": self.patch_norm,
                "patch_act": self.patch_act,
                "transformer": self.transformer,
                "decode": self.decode,
                "out": self.out,
            }
        )

    def __call__(self, x):  # noqa: ANN001
        m = self.model
        feat = m["patch_embed"](x)
        feat = m["patch_norm"](feat)
        feat = m["patch_act"](feat)

        b, c, h, w = feat.shape
        t = feat.flatten(2).transpose(1, 2)
        t = m["transformer"](t)
        feat = t.transpose(1, 2).reshape(b, c, h, w)

        feat = self.torch.nn.functional.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feat = m["decode"](feat)
        return m["out"](feat)

    def to(self, device: str) -> _SegFormerMiniModel:
        self.model.to(device)
        return self

    def train(self) -> _SegFormerMiniModel:
        self.model.train()
        return self

    def eval(self) -> _SegFormerMiniModel:
        self.model.eval()
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True):
        return self.model.load_state_dict(state, strict=bool(strict))

    def parameters(self):
        return self.model.parameters()


def _hf_segformer_config_for_variant(variant: str):
    from transformers import SegformerConfig

    name = str(variant).strip().lower()
    presets: dict[str, dict[str, Any]] = {
        "b0": {
            "hidden_sizes": [32, 64, 160, 256],
            "depths": [2, 2, 2, 2],
            "num_attention_heads": [1, 2, 5, 8],
            "decoder_hidden_size": 256,
            "drop_path_rate": 0.1,
        },
        "b2": {
            "hidden_sizes": [64, 128, 320, 512],
            "depths": [3, 4, 6, 3],
            "num_attention_heads": [1, 2, 5, 8],
            "decoder_hidden_size": 768,
            "drop_path_rate": 0.1,
        },
        "b5": {
            "hidden_sizes": [64, 128, 320, 512],
            "depths": [3, 6, 40, 3],
            "num_attention_heads": [1, 2, 5, 8],
            "decoder_hidden_size": 768,
            "drop_path_rate": 0.1,
        },
    }
    if name not in presets:
        raise ValueError(f"unsupported hf segformer variant: {variant}")

    p = presets[name]
    return SegformerConfig(
        num_labels=2,
        num_channels=3,
        hidden_sizes=p["hidden_sizes"],
        depths=p["depths"],
        num_attention_heads=p["num_attention_heads"],
        decoder_hidden_size=p["decoder_hidden_size"],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=float(p["drop_path_rate"]),
        reshape_last_stage=True,
    )


def _extract_state_dict(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint payload type: {type(payload).__name__}")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return dict(payload["model_state_dict"])
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return dict(payload["state_dict"])
    if payload and all(isinstance(key, str) for key in payload.keys()):
        return dict(payload)
    raise ValueError("could not determine model state_dict from checkpoint payload")


def _load_local_torch_pretrained(
    *,
    model_obj: Any,
    pretrained_bundle: _PretrainedInitBundle,
    strict: bool,
) -> dict[str, Any]:
    """Load local torch checkpoint/state_dict into a model wrapper."""

    import torch

    path = Path(pretrained_bundle.weights_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            "local pretrained weights must point to an existing checkpoint/state_dict file; "
            f"got: {path}"
        )
    payload = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    load_notes: dict[str, Any] = {"missing_keys": [], "unexpected_keys": []}
    try:
        result = model_obj.load_state_dict(state_dict, strict=bool(strict))
    except TypeError:
        result = model_obj.load_state_dict(state_dict)
    load_notes["missing_keys"] = list(getattr(result, "missing_keys", []))
    load_notes["unexpected_keys"] = list(getattr(result, "unexpected_keys", []))
    return load_notes


class _SmpUNetBinaryModel:
    """SMP U-Net model wrapper supporting local pretrained-state initialization."""

    def __init__(
        self,
        *,
        encoder_name: str = "resnet18",
        pretrained_bundle: _PretrainedInitBundle | None = None,
        strict: bool = False,
    ) -> None:
        import torch

        try:
            import segmentation_models_pytorch as smp
        except Exception as exc:
            raise RuntimeError(
                "segmentation_models_pytorch is required for smp_unet_* backends. "
                "Install with `pip install segmentation-models-pytorch`."
            ) from exc

        self.torch = torch
        self.model = smp.Unet(
            encoder_name=str(encoder_name),
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        self.load_notes: dict[str, Any] = {"missing_keys": [], "unexpected_keys": []}

        if pretrained_bundle is not None:
            path = Path(pretrained_bundle.weights_path)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(
                    "smp_unet_resnet18 requires pretrained weights_path to be an existing file; "
                    f"got: {path}"
                )
            payload = torch.load(path, map_location="cpu")
            state_dict = _extract_state_dict(payload)
            load_result = self.model.load_state_dict(state_dict, strict=bool(strict))
            self.load_notes["missing_keys"] = list(getattr(load_result, "missing_keys", []))
            self.load_notes["unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []))

    def __call__(self, x):  # noqa: ANN001
        return self.model(x)

    def to(self, device: str) -> _SmpUNetBinaryModel:
        self.model.to(device)
        return self

    def train(self) -> _SmpUNetBinaryModel:
        self.model.train()
        return self

    def eval(self) -> _SmpUNetBinaryModel:
        self.model.eval()
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True):
        return self.model.load_state_dict(state, strict=bool(strict))

    def parameters(self):
        return self.model.parameters()


class _HfSegFormerBinaryModel:
    """Hugging Face SegFormer binary model with optional local pretrained init."""

    def __init__(
        self,
        *,
        variant: str = "b0",
        pretrained_bundle: _PretrainedInitBundle | None = None,
        ignore_mismatched_sizes: bool = True,
    ) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation

        self.torch = torch
        self.variant = str(variant).strip().lower()
        if pretrained_bundle is None:
            self.model = SegformerForSemanticSegmentation(_hf_segformer_config_for_variant(self.variant))
        else:
            pretrained_dir = Path(pretrained_bundle.weights_path)
            if not pretrained_dir.exists():
                raise FileNotFoundError(f"local pretrained transformer path does not exist: {pretrained_dir}")
            if pretrained_dir.is_file():
                pretrained_dir = pretrained_dir.parent
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                str(pretrained_dir),
                local_files_only=True,
                ignore_mismatched_sizes=bool(ignore_mismatched_sizes),
                num_labels=2,
            )
            self.model.config.num_labels = 2

    def __call__(self, x):  # noqa: ANN001
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = self.torch.nn.functional.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if logits.shape[1] == 1:
            return logits
        # Convert 2-class logits to one-logit binary form for BCE workflow.
        return logits[:, 1:2, ...] - logits[:, 0:1, ...]

    def to(self, device: str) -> _HfSegFormerBinaryModel:
        self.model.to(device)
        return self

    def train(self) -> _HfSegFormerBinaryModel:
        self.model.train()
        return self

    def eval(self) -> _HfSegFormerBinaryModel:
        self.model.eval()
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True):
        return self.model.load_state_dict(state, strict=bool(strict))

    def parameters(self):
        return self.model.parameters()


def _build_binary_model(
    *,
    architecture: str,
    base_channels: int,
    transformer_depth: int,
    transformer_num_heads: int,
    transformer_mlp_ratio: float,
    transformer_dropout: float,
    segformer_patch_size: int,
    pretrained_bundle: _PretrainedInitBundle | None = None,
    pretrained_strict: bool = False,
    pretrained_ignore_mismatched_sizes: bool = True,
):
    arch = str(architecture).strip().lower()
    if arch == "unet_binary":
        model = _UNetBinaryModel(base_channels=max(4, int(base_channels)))
        if pretrained_bundle is not None:
            if str(pretrained_bundle.weights_format).strip().lower() == "hf_model_dir":
                raise ValueError(
                    "unet_binary local-pretrained expects a torch checkpoint/state_dict file, "
                    f"got weights_format={pretrained_bundle.weights_format!r}"
                )
            _load_local_torch_pretrained(
                model_obj=model,
                pretrained_bundle=pretrained_bundle,
                strict=bool(pretrained_strict),
            )
        return model
    if arch.startswith("smp_unet_"):
        encoder = arch[len("smp_unet_") :]
        if not encoder:
            raise ValueError(f"invalid smp_unet architecture name: {architecture!r}")
        return _SmpUNetBinaryModel(
            encoder_name=encoder,
            pretrained_bundle=pretrained_bundle,
            strict=bool(pretrained_strict),
        )
    if arch == "transunet_tiny":
        channels = max(4, int(base_channels)) * 4
        if channels % max(1, int(transformer_num_heads)) != 0:
            raise ValueError(
                "transunet_tiny requires (model_base_channels*4) divisible by transformer_num_heads; "
                f"got base={base_channels}, heads={transformer_num_heads}"
            )
        model = _TransUNetTinyModel(
            base_channels=max(4, int(base_channels)),
            transformer_depth=max(1, int(transformer_depth)),
            transformer_num_heads=max(1, int(transformer_num_heads)),
            transformer_mlp_ratio=max(1.0, float(transformer_mlp_ratio)),
            transformer_dropout=max(0.0, float(transformer_dropout)),
        )
        if pretrained_bundle is not None:
            if str(pretrained_bundle.weights_format).strip().lower() == "hf_model_dir":
                raise ValueError(
                    "transunet_tiny local-pretrained expects a torch checkpoint/state_dict file, "
                    f"got weights_format={pretrained_bundle.weights_format!r}"
                )
            _load_local_torch_pretrained(
                model_obj=model,
                pretrained_bundle=pretrained_bundle,
                strict=bool(pretrained_strict),
            )
        return model
    if arch == "segformer_mini":
        embed_dim = max(16, max(4, int(base_channels)) * 4)
        if embed_dim % max(1, int(transformer_num_heads)) != 0:
            raise ValueError(
                "segformer_mini requires embed_dim divisible by transformer_num_heads; "
                f"got embed_dim={embed_dim}, heads={transformer_num_heads}"
            )
        model = _SegFormerMiniModel(
            base_channels=max(4, int(base_channels)),
            transformer_depth=max(1, int(transformer_depth)),
            transformer_num_heads=max(1, int(transformer_num_heads)),
            transformer_mlp_ratio=max(1.0, float(transformer_mlp_ratio)),
            transformer_dropout=max(0.0, float(transformer_dropout)),
            patch_size=max(2, int(segformer_patch_size)),
        )
        if pretrained_bundle is not None:
            if str(pretrained_bundle.weights_format).strip().lower() == "hf_model_dir":
                raise ValueError(
                    "segformer_mini local-pretrained expects a torch checkpoint/state_dict file, "
                    f"got weights_format={pretrained_bundle.weights_format!r}"
                )
            _load_local_torch_pretrained(
                model_obj=model,
                pretrained_bundle=pretrained_bundle,
                strict=bool(pretrained_strict),
            )
        return model
    if arch == "hf_segformer_b0":
        return _HfSegFormerBinaryModel(
            variant="b0",
            pretrained_bundle=pretrained_bundle,
            ignore_mismatched_sizes=bool(pretrained_ignore_mismatched_sizes),
        )
    if arch == "hf_segformer_b2":
        return _HfSegFormerBinaryModel(
            variant="b2",
            pretrained_bundle=pretrained_bundle,
            ignore_mismatched_sizes=bool(pretrained_ignore_mismatched_sizes),
        )
    if arch == "hf_segformer_b5":
        return _HfSegFormerBinaryModel(
            variant="b5",
            pretrained_bundle=pretrained_bundle,
            ignore_mismatched_sizes=bool(pretrained_ignore_mismatched_sizes),
        )
    raise ValueError(f"unsupported model_architecture: {architecture}")


def _checkpoint_schema_for_architecture(architecture: str) -> str:
    arch = str(architecture).strip().lower()
    if arch == "unet_binary":
        return "microseg.torch_unet_binary.v1"
    if arch.startswith("hf_segformer_"):
        return "microseg.hf_transformer_segmentation.v1"
    return "microseg.torch_segmentation_binary.v2"


@dataclass(frozen=True)
class UNetBinaryTrainingConfig:
    """Configuration for UNet binary segmentation training."""

    dataset_dir: str
    output_dir: str
    train_split: str = "train"
    val_split: str = "val"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    enable_gpu: bool = False
    device_policy: str = "cpu"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    checkpoint_every: int = 1
    resume_checkpoint: str = ""
    val_tracking_samples: int = 6
    val_tracking_fixed_samples: tuple[str, ...] = ()
    val_tracking_seed: int = 17
    write_html_report: bool = True
    progress_log_interval_pct: int = 10
    post_epoch_heartbeat_seconds: float = 30.0
    val_progress_log_every_batches: int = 25
    model_architecture: str = "unet_binary"
    model_base_channels: int = 16
    transformer_depth: int = 2
    transformer_num_heads: int = 4
    transformer_mlp_ratio: float = 2.0
    transformer_dropout: float = 0.0
    segformer_patch_size: int = 4
    backend_label: str = "unet_binary"
    pretrained_init_mode: str = "scratch"
    pretrained_model_id: str = ""
    pretrained_bundle_dir: str = ""
    pretrained_registry_path: str = "pre_trained_weights/registry.json"
    pretrained_strict: bool = False
    pretrained_ignore_mismatched_sizes: bool = True
    pretrained_verify_sha256: bool = True
    amp_enabled: bool = False
    grad_accum_steps: int = 1
    torch_compile: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    deterministic: bool = True
    binary_mask_normalization: str = "off"
    input_hw: tuple[int, int] = (512, 512)
    input_policy: str = "random_crop"
    val_input_policy: str = "letterbox"
    keep_aspect: bool = True
    pad_value_image: float = 0.0
    pad_value_mask: int = 0
    image_interpolation: str = "bilinear"
    mask_interpolation: str = "nearest"
    require_divisible_by: int = 32
    dataloader_collate: str = "default"


def _binary_iou_from_logits(logits, targets) -> float:  # noqa: ANN001
    import torch

    pred = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
    inter = torch.sum(pred * targets).item()
    union = torch.sum((pred + targets) > 0).item()
    if union == 0:
        return 1.0
    return float(inter / union)


def _binary_accuracy_from_logits(logits, targets) -> float:  # noqa: ANN001
    import torch

    pred = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
    return float(torch.mean((pred == targets).to(torch.float32)).item())


def _binary_iou_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    inter = int(np.count_nonzero((y_true > 0) & (y_pred > 0)))
    union = int(np.count_nonzero((y_true > 0) | (y_pred > 0)))
    if union == 0:
        return 1.0
    return float(inter / union)


def _binary_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    gt = (y_true > 0).astype(np.uint8).reshape(-1)
    pred = (y_pred > 0).astype(np.uint8).reshape(-1)
    total = float(gt.size)

    tp = float(np.count_nonzero((gt == 1) & (pred == 1)))
    tn = float(np.count_nonzero((gt == 0) & (pred == 0)))
    fp = float(np.count_nonzero((gt == 0) & (pred == 1)))
    fn = float(np.count_nonzero((gt == 1) & (pred == 0)))

    fg_precision = _safe_div(tp, tp + fp)
    fg_recall = _safe_div(tp, tp + fn)
    fg_f1 = _safe_div(2.0 * fg_precision * fg_recall, fg_precision + fg_recall)
    fg_iou = _safe_div(tp, tp + fp + fn)

    bg_precision = _safe_div(tn, tn + fn)
    bg_recall = _safe_div(tn, tn + fp)
    bg_f1 = _safe_div(2.0 * bg_precision * bg_recall, bg_precision + bg_recall)
    bg_iou = _safe_div(tn, tn + fp + fn)

    macro_precision = 0.5 * (fg_precision + bg_precision)
    macro_recall = 0.5 * (fg_recall + bg_recall)
    macro_f1 = 0.5 * (fg_f1 + bg_f1)
    weighted_f1 = _safe_div(((tn + fp) * bg_f1) + ((tp + fn) * fg_f1), total)
    balanced_accuracy = macro_recall
    frequency_weighted_iou = _safe_div(((tn + fp) * bg_iou) + ((tp + fn) * fg_iou), total)
    pixel_accuracy = _safe_div(tp + tn, total)

    fg_specificity = _safe_div(tn, tn + fp)
    fg_dice = _safe_div(2.0 * tp, (2.0 * tp) + fp + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0 if mcc_denom <= 0.0 else _safe_div((tp * tn) - (fp * fn), sqrt(mcc_denom))

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mean_iou": float(0.5 * (fg_iou + bg_iou)),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "frequency_weighted_iou": float(frequency_weighted_iou),
        "foreground_precision": float(fg_precision),
        "foreground_recall": float(fg_recall),
        "foreground_specificity": float(fg_specificity),
        "foreground_iou": float(fg_iou),
        "foreground_dice": float(fg_dice),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "matthews_corrcoef": float(mcc),
    }


def _sample_metrics_block(sample: dict[str, Any], metric_order: list[str]) -> str:
    metrics_lines: list[str] = []
    for key in metric_order:
        if key not in sample:
            continue
        value = sample.get(key)
        try:
            value_text = f"{float(value):.6f}"
        except Exception:
            continue
        label = html.escape(key.replace("_", " "))
        metrics_lines.append(f"<li><b>{label}</b>: {value_text}</li>")
    if not metrics_lines:
        return ""
    return "<ul style='margin:8px 0 0 18px;'>" + "".join(metrics_lines) + "</ul>"


def _tracking_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_u8 = (gt.astype(np.uint8) * 255)
    pred_u8 = (pred.astype(np.uint8) * 255)
    diff_u8 = ((gt.astype(np.uint8) != pred.astype(np.uint8)).astype(np.uint8) * 255)
    gt_rgb = np.stack([gt_u8, gt_u8, gt_u8], axis=2)
    pred_rgb = np.stack([pred_u8, pred_u8, pred_u8], axis=2)
    diff_rgb = np.stack([diff_u8, diff_u8, diff_u8], axis=2)
    return np.concatenate([image, gt_rgb, pred_rgb, diff_rgb], axis=1).astype(np.uint8)


def _write_training_html(payload: dict[str, Any], output_path: Path) -> None:
    history_raw = payload.get("history", [])
    history = history_raw if isinstance(history_raw, list) else []
    rows: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{int(item.get('epoch', 0))}</td>"
            f"<td>{float(item.get('train_loss', 0.0)):.6f}</td>"
            f"<td>{float(item.get('train_accuracy', 0.0)):.4f}</td>"
            f"<td>{float(item.get('train_iou', 0.0)):.4f}</td>"
            f"<td>{float(item.get('val_loss', 0.0)):.6f}</td>"
            f"<td>{float(item.get('val_accuracy', 0.0)):.4f}</td>"
            f"<td>{float(item.get('val_iou', 0.0)):.4f}</td>"
            "</tr>"
        )

    samples_raw = payload.get("latest_tracked_samples", [])
    samples = samples_raw if isinstance(samples_raw, list) else []
    sample_metric_order = [
        "iou",
        "pixel_accuracy",
        "macro_f1",
        "mean_iou",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "balanced_accuracy",
        "frequency_weighted_iou",
        "foreground_precision",
        "foreground_recall",
        "foreground_specificity",
        "foreground_iou",
        "foreground_dice",
        "false_positive_rate",
        "false_negative_rate",
        "matthews_corrcoef",
        "mask_area_fraction_abs_error",
        "hydride_count_abs_error",
        "hydride_size_wasserstein",
        "hydride_orientation_wasserstein",
    ]

    def _sample_card(sample: dict[str, Any]) -> str:
        panel = html.escape(str(sample.get("panel", "")))
        name = html.escape(str(sample.get("sample_name", "")))
        iou_raw = sample.get("iou")
        try:
            iou = float(iou_raw)
            iou_text = f" | IoU={iou:.4f}"
        except Exception:
            iou_text = ""
        metrics_html = _sample_metrics_block(sample, sample_metric_order)
        return (
            "<div style='margin:10px 0;padding:10px;border:1px solid #ddd;'>"
            f"<div><b>{name}</b>{iou_text}</div>"
            f"<img src='{panel}' style='max-width:100%;border:1px solid #333;' alt='{name}'>"
            + metrics_html
            + "</div>"
        )

    latest_gallery: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        latest_gallery.append(_sample_card(sample))

    epoch_gallery: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        tracked_raw = item.get("tracked_samples", [])
        tracked = tracked_raw if isinstance(tracked_raw, list) else []
        cards: list[str] = []
        for sample in tracked:
            if not isinstance(sample, dict):
                continue
            cards.append(_sample_card(sample))
        if not cards:
            continue
        epoch_num = int(item.get("epoch", 0))
        epoch_gallery.append(f"<h3>Epoch {epoch_num}</h3>")
        epoch_gallery.append(
            "<p><b>Train Acc:</b> "
            + f"{float(item.get('train_accuracy', 0.0)):.4f}"
            + " | <b>Val Acc:</b> "
            + f"{float(item.get('val_accuracy', 0.0)):.4f}"
            + " | <b>Train IoU:</b> "
            + f"{float(item.get('train_iou', 0.0)):.4f}"
            + " | <b>Val IoU:</b> "
            + f"{float(item.get('val_iou', 0.0)):.4f}"
            + "</p>"
        )
        epoch_gallery.extend(cards)

    progress = payload.get("progress", {})
    html_text = (
        "<html><head><meta charset='utf-8'><title>MicroSeg Training Report</title></head><body>"
        "<h1>MicroSeg Training Report</h1>"
        f"<p><b>Status:</b> {html.escape(str(payload.get('status', 'unknown')))}</p>"
        f"<p><b>Started (UTC):</b> {html.escape(str(payload.get('started_utc', '')))}</p>"
        f"<p><b>Updated (UTC):</b> {html.escape(str(payload.get('updated_utc', '')))}</p>"
        f"<p><b>Runtime:</b> {html.escape(str(payload.get('runtime_human', '')))}</p>"
        f"<p><b>Progress:</b> {float(progress.get('total_percent', 0.0)):.1f}%"
        f" (epoch {int(progress.get('epochs_completed', 0))}/{int(progress.get('epochs_total', 0))})</p>"
        "<h2>Epoch Metrics</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Epoch</th><th>Train Loss</th><th>Train Acc</th><th>Train IoU</th><th>Val Loss</th><th>Val Acc</th><th>Val IoU</th></tr>"
        + "".join(rows)
        + "</table>"
        "<h2>Tracked Validation Samples (Latest Epoch: Input | GT | Pred | Diff)</h2>"
        + ("".join(latest_gallery) if latest_gallery else "<p>No tracked samples recorded yet.</p>")
        + "<h2>Tracked Validation Samples By Epoch (Input | GT | Pred | Diff)</h2>"
        + ("".join(epoch_gallery) if epoch_gallery else "<p>No per-epoch tracked samples recorded.</p>")
        + "</body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


class UNetBinaryTrainer:
    """Train UNet binary segmentation model with checkpoints and early stopping."""

    def train(self, config: UNetBinaryTrainingConfig) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        logger = _ensure_logger()
        _set_seed(int(config.seed))

        dataset_root = Path(config.dataset_dir)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "report.json"
        html_report_path = output_dir / "training_report.html"
        history_jsonl_path = output_dir / "epoch_history.jsonl"
        events_path = output_dir / "events.jsonl"
        samples_root = output_dir / "eval_samples"
        config_payload = asdict(config)
        config_sha256 = _config_hash(config_payload)

        train_pairs = _collect_pairs(dataset_root / config.train_split)
        val_pairs = _collect_pairs(dataset_root / config.val_split)
        _warn_binary_mask_remap_values(
            train_pairs + val_pairs,
            binary_mask_normalization=str(config.binary_mask_normalization),
            logger=logger,
        )
        fixed_sample_names = _normalize_fixed_samples(config.val_tracking_fixed_samples)

        workers = max(0, int(config.num_workers))
        use_persistent_workers = bool(config.persistent_workers) and workers > 0
        pin_memory = bool(config.pin_memory)
        collate_fn = resolve_collate_fn(config.dataloader_collate)
        input_hw = tuple(int(v) for v in config.input_hw)
        if len(input_hw) != 2:
            raise ValueError(f"input_hw must be (height, width), got {config.input_hw!r}")
        if str(config.mask_interpolation).strip().lower() != "nearest":
            raise ValueError("mask_interpolation must be 'nearest' for segmentation masks")
        train_policy_cfg = InputPolicyConfig(
            input_hw=input_hw,
            input_policy=str(config.input_policy).strip().lower() or "random_crop",
            keep_aspect=bool(config.keep_aspect),
            pad_value_image=float(config.pad_value_image),
            pad_value_mask=int(config.pad_value_mask),
            image_interpolation=str(config.image_interpolation).strip().lower() or "bilinear",
            require_divisible_by=max(1, int(config.require_divisible_by)),
        )
        val_policy_cfg = InputPolicyConfig(
            input_hw=input_hw,
            input_policy=str(config.val_input_policy).strip().lower() or "letterbox",
            keep_aspect=bool(config.keep_aspect),
            pad_value_image=float(config.pad_value_image),
            pad_value_mask=int(config.pad_value_mask),
            image_interpolation=str(config.image_interpolation).strip().lower() or "bilinear",
            require_divisible_by=max(1, int(config.require_divisible_by)),
        )
        logger.info(
            "input size policy train=%s val=%s input_hw=%s require_divisible_by=%d collate=%s",
            train_policy_cfg.input_policy,
            val_policy_cfg.input_policy,
            train_policy_cfg.input_hw,
            int(train_policy_cfg.require_divisible_by),
            str(config.dataloader_collate),
        )

        train_loader = DataLoader(
            _SegPairDataset(
                train_pairs,
                binary_mask_normalization=config.binary_mask_normalization,
                input_policy_cfg=train_policy_cfg,
                seed=int(config.seed),
                is_train=True,
                logger=logger,
            ),
            batch_size=max(1, int(config.batch_size)),
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            _SegPairDataset(
                val_pairs,
                binary_mask_normalization=config.binary_mask_normalization,
                input_policy_cfg=val_policy_cfg,
                seed=int(config.seed),
                is_train=False,
                logger=logger,
            ),
            batch_size=max(1, int(config.batch_size)),
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent_workers,
            collate_fn=collate_fn,
        )

        resolved = resolve_torch_device(enable_gpu=bool(config.enable_gpu), policy=str(config.device_policy))
        device = resolved.selected_device
        architecture = str(config.model_architecture).strip().lower() or "unet_binary"
        backend_label = str(config.backend_label).strip().lower() or architecture
        pretrained_bundle = _resolve_pretrained_bundle(
            init_mode=str(config.pretrained_init_mode),
            model_id=str(config.pretrained_model_id),
            bundle_dir=str(config.pretrained_bundle_dir),
            registry_path=str(config.pretrained_registry_path),
            architecture=architecture,
            verify_sha256=bool(config.pretrained_verify_sha256),
        )
        if pretrained_bundle is not None:
            init_mode = "local_pretrained"
        elif architecture.startswith("hf_segformer_") or architecture.startswith("smp_unet_"):
            init_mode = "scratch"
        else:
            init_mode = "native"
        pretrained_payload: dict[str, Any] = {}
        if pretrained_bundle is not None:
            pretrained_payload = {
                "model_id": pretrained_bundle.model_id,
                "architecture": pretrained_bundle.architecture,
                "framework": pretrained_bundle.framework,
                "source": pretrained_bundle.source,
                "source_revision": pretrained_bundle.source_revision,
                "source_url": pretrained_bundle.source_url,
                "weights_format": pretrained_bundle.weights_format,
                "license": pretrained_bundle.license,
                "citation_key": pretrained_bundle.citation_key,
                "citation": pretrained_bundle.citation,
                "citation_url": pretrained_bundle.citation_url,
                "bundle_dir": str(pretrained_bundle.bundle_dir),
                "weights_path": str(pretrained_bundle.weights_path),
                "metadata_path": str(pretrained_bundle.metadata_path) if pretrained_bundle.metadata_path else "",
            }
        model = _build_binary_model(
            architecture=architecture,
            base_channels=int(config.model_base_channels),
            transformer_depth=int(config.transformer_depth),
            transformer_num_heads=int(config.transformer_num_heads),
            transformer_mlp_ratio=float(config.transformer_mlp_ratio),
            transformer_dropout=float(config.transformer_dropout),
            segformer_patch_size=int(config.segformer_patch_size),
            pretrained_bundle=pretrained_bundle,
            pretrained_strict=bool(config.pretrained_strict),
            pretrained_ignore_mismatched_sizes=bool(config.pretrained_ignore_mismatched_sizes),
        ).to(device)
        model_parameter_count, model_trainable_parameter_count = _model_parameter_counts(model)
        model_weight_stats: dict[str, Any] = {}
        runtime_hardware = _runtime_hardware_profile(str(device))
        first_img = np.asarray(Image.open(train_pairs[0][0]).convert("RGB"), dtype=np.uint8)
        flops_per_sample_forward = _estimate_forward_flops_per_sample(
            model,
            image_height=int(first_img.shape[0]),
            image_width=int(first_img.shape[1]),
            device=str(device),
        )
        if bool(config.torch_compile) and hasattr(torch, "compile") and hasattr(model, "model"):
            try:
                model.model = torch.compile(model.model)  # type: ignore[assignment]
                logger.info("torch.compile enabled for model backend")
            except Exception as exc:
                logger.warning("torch.compile requested but unavailable: %s", exc)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        if bool(config.deterministic):
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        amp_pref = bool(config.amp_enabled)
        if not amp_pref and architecture.startswith("hf_segformer_"):
            amp_pref = True
        use_amp = amp_pref and str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        grad_accum_steps = max(1, int(config.grad_accum_steps))
        if str(device).startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            except Exception:
                pass

        start_epoch = 1
        best_val_loss = float("inf")
        best_path = output_dir / "best_checkpoint.pt"
        last_path = output_dir / "last_checkpoint.pt"

        if config.resume_checkpoint:
            resume_path = Path(config.resume_checkpoint)
            if resume_path.exists():
                ckpt = torch.load(resume_path, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
                logger.info("Resuming training from checkpoint: %s (next epoch=%d)", resume_path, start_epoch)

        started_utc = _utc_now()
        run_start = time.perf_counter()
        status = "running"
        interrupted = False
        history: list[dict[str, Any]] = []
        latest_samples: list[dict[str, Any]] = []
        no_improve = 0
        checkpoint_files: list[str] = []
        train_samples_processed = 0
        val_samples_processed = 0
        tracking_samples_processed = 0

        def write_report(
            *,
            current_epoch: int,
            epoch_percent: float,
            eta_seconds: float,
            status_value: str,
            error: dict[str, Any] | None = None,
        ) -> None:
            runtime_seconds = time.perf_counter() - run_start
            total_epochs = max(1, int(config.epochs))
            total_percent = min(100.0, max(0.0, (len(history) / total_epochs) * 100.0))
            selected_ckpt = best_path if best_path.exists() else last_path
            checkpoint_size_bytes = (
                int(selected_ckpt.stat().st_size) if selected_ckpt.exists() and selected_ckpt.is_file() else None
            )
            estimated_total_flops = None
            if flops_per_sample_forward is not None:
                estimated_total_flops = int(
                    int(flops_per_sample_forward)
                    * (
                        (3 * int(train_samples_processed))
                        + int(val_samples_processed)
                        + int(tracking_samples_processed)
                    )
                )
            hardware_payload = dict(runtime_hardware)
            peak_mb = _current_gpu_peak_memory_mb(str(device))
            if peak_mb is not None:
                hardware_payload["gpu_peak_memory_allocated_mb"] = float(peak_mb)
            payload: dict[str, Any] = {
                "schema_version": "microseg.training_report.v1",
                "backend": backend_label,
                "status": status_value,
                "started_utc": started_utc,
                "updated_utc": _utc_now(),
                "runtime_seconds": runtime_seconds,
                "runtime_human": _format_seconds(runtime_seconds),
                "code_version": _code_version(),
                "config": config_payload,
                "config_sha256": config_sha256,
                "model_architecture": architecture,
                "model_initialization": init_mode,
                "pretrained_init": pretrained_payload,
                "device": device,
                "device_reason": resolved.reason,
                "runtime_hardware": hardware_payload,
                "model_parameter_count": model_parameter_count,
                "model_trainable_parameter_count": model_trainable_parameter_count,
                "model_weight_statistics": model_weight_stats,
                "model_checkpoint_size_bytes": checkpoint_size_bytes,
                "model_checkpoint_size_mb": (
                    float(checkpoint_size_bytes) / (1024.0 * 1024.0) if checkpoint_size_bytes is not None else None
                ),
                "compute_effort": {
                    "train_samples_processed": int(train_samples_processed),
                    "val_samples_processed": int(val_samples_processed),
                    "tracking_samples_processed": int(tracking_samples_processed),
                    "estimated_forward_flops_per_sample": flops_per_sample_forward,
                    "estimated_total_flops": estimated_total_flops,
                    "estimated_total_tflops": (
                        float(estimated_total_flops) / 1_000_000_000_000.0
                        if estimated_total_flops is not None
                        else None
                    ),
                    "flops_estimate_method": (
                        "forward_hook_approximate; total ~= 3*train + 1*val + 1*tracking samples"
                        if flops_per_sample_forward is not None
                        else "unavailable"
                    ),
                },
                "progress": {
                    "epochs_total": int(config.epochs),
                    "epochs_completed": len(history),
                    "current_epoch": current_epoch,
                    "epoch_percent": max(0.0, min(100.0, epoch_percent)),
                    "total_percent": total_percent,
                    "eta_seconds": max(0.0, eta_seconds),
                    "eta_human": _format_seconds(max(0.0, eta_seconds)),
                },
                "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
                "best_checkpoint": _to_rel(best_path, output_dir) if best_path.exists() else "",
                "last_checkpoint": _to_rel(last_path, output_dir) if last_path.exists() else "",
                "history": history,
                "latest_tracked_samples": latest_samples,
                "checkpoint_files": sorted(checkpoint_files),
            }
            if error:
                payload["error"] = error
            if status_value in {"completed", "interrupted", "failed"}:
                payload["finished_utc"] = _utc_now()
            _write_json(report_path, payload)
            if bool(config.write_html_report):
                _write_training_html(payload, html_report_path)

        logger.info(
            "Segmentation training started | backend=%s arch=%s init=%s | dataset=%s | output=%s | device=%s (%s) | epochs=%d | batch=%d",
            backend_label,
            architecture,
            init_mode,
            dataset_root,
            output_dir,
            device,
            resolved.reason,
            int(config.epochs),
            int(config.batch_size),
        )
        _append_jsonl_logged(
            events_path,
            {
                "event": "run_started",
                "ts_utc": _utc_now(),
                "dataset_dir": str(dataset_root),
                "output_dir": str(output_dir),
                "device": device,
                "device_reason": resolved.reason,
                "model_initialization": init_mode,
                "pretrained_init": pretrained_payload,
            },
            logger=logger,
            context="RUN_STARTED_EVENT",
        )
        write_report(current_epoch=start_epoch, epoch_percent=0.0, eta_seconds=0.0, status_value=status)

        def export_tracking_samples(epoch: int) -> tuple[list[dict[str, Any]], list[str], int]:
            selected_pairs, missing = _select_tracking_pairs(
                val_pairs,
                fixed_names=fixed_sample_names,
                total_samples=max(0, int(config.val_tracking_samples)),
                seed=int(config.val_tracking_seed),
                epoch=epoch,
            )
            logger.info("TRACK_EXPORT_START | epoch=%d selected=%d", epoch, len(selected_pairs))
            if not selected_pairs:
                logger.info("TRACK_EXPORT_END | epoch=%d selected=0 elapsed=%s", epoch, _format_seconds(0.0))
                return [], missing, 0

            out_epoch = samples_root / f"epoch_{epoch:03d}"
            out_epoch.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, Any]] = []
            export_start = time.perf_counter()
            heartbeat = _Heartbeat(float(config.post_epoch_heartbeat_seconds), logger)

            model.eval()
            with torch.no_grad():
                for sample_idx, (img_path, mask_path) in enumerate(selected_pairs, start=1):
                    sample_start = time.perf_counter()
                    logger.info("TRACK_SAMPLE_START | epoch=%d sample=%d/%d name=%s", epoch, sample_idx, len(selected_pairs), img_path.name)
                    io_start = time.perf_counter()
                    image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                    gt = normalize_binary_index_mask(
                        np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8),
                        mode=config.binary_mask_normalization,
                    )
                    gt_bin = (gt > 0).astype(np.uint8)
                    logger.info("TRACK_SAMPLE_IMAGE_LOAD_END | sample=%s elapsed=%s", img_path.name, _format_seconds(time.perf_counter() - io_start))

                    prep_start = time.perf_counter()
                    x = torch.from_numpy((image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]).to(device)
                    logger.info("TRACK_SAMPLE_TENSOR_PREP_END | sample=%s elapsed=%s", img_path.name, _format_seconds(time.perf_counter() - prep_start))

                    fw_start = time.perf_counter()
                    logits = model(x)
                    logger.info("TRACK_SAMPLE_FORWARD_END | sample=%s elapsed=%s", img_path.name, _format_seconds(time.perf_counter() - fw_start))

                    post_start = time.perf_counter()
                    pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()[0, 0].astype(np.uint8)
                    sample_metrics = _binary_sample_metrics(gt_bin, pred)
                    scientific_metrics = scientific_distance_metrics(gt_bin, pred)
                    logger.info("TRACK_SAMPLE_POSTPROC_END | sample=%s elapsed=%s", img_path.name, _format_seconds(time.perf_counter() - post_start))

                    stem = img_path.stem
                    panel_path = out_epoch / f"{stem}_panel.png"
                    pred_path = out_epoch / f"{stem}_pred.png"
                    gt_path = out_epoch / f"{stem}_gt.png"

                    write_start = time.perf_counter()
                    logger.info("TRACK_SAMPLE_FILE_WRITE_START | sample=%s gt=%s pred=%s panel=%s", img_path.name, gt_path, pred_path, panel_path)
                    Image.fromarray(_tracking_panel(image, gt_bin, pred)).save(panel_path)
                    Image.fromarray((pred * 255).astype(np.uint8)).save(pred_path)
                    Image.fromarray((gt_bin * 255).astype(np.uint8)).save(gt_path)
                    logger.info("TRACK_SAMPLE_FILE_WRITE_END | sample=%s elapsed=%s", img_path.name, _format_seconds(time.perf_counter() - write_start))

                    records.append(
                        {
                            "sample_name": img_path.name,
                            "iou": float(sample_metrics.get("foreground_iou", _binary_iou_from_arrays(gt_bin, pred))),
                            "panel": _to_rel(panel_path, output_dir),
                            "pred": _to_rel(pred_path, output_dir),
                            "gt": _to_rel(gt_path, output_dir),
                            "is_fixed": img_path.name in fixed_sample_names,
                            **sample_metrics,
                            **scientific_metrics,
                        }
                    )
                    logger.info("TRACK_SAMPLE_END | sample=%s elapsed=%s", img_path.name, _format_seconds(time.perf_counter() - sample_start))
                    heartbeat.beat("HEARTBEAT | phase=tracking_export epoch=%d sample=%d/%d elapsed=%s", epoch, sample_idx, len(selected_pairs), _format_seconds(time.perf_counter() - export_start))
            logger.info("TRACK_EXPORT_END | epoch=%d selected=%d elapsed=%s", epoch, len(selected_pairs), _format_seconds(time.perf_counter() - export_start))
            return records, missing, len(selected_pairs)

        try:
            logger.info(
                "POST_EPOCH_CONTEXT | output_dir=%s pid=%d host=%s cwd=%s",
                output_dir,
                os.getpid(),
                socket.gethostname(),
                Path.cwd(),
            )
            for epoch in range(start_epoch, int(config.epochs) + 1):
                epoch_timings: dict[str, float] = {}
                epoch_start = time.perf_counter()
                model.train()
                train_loss_sum = 0.0
                train_acc_sum = 0.0
                train_iou_sum = 0.0
                train_steps = 0

                total_train_steps = max(1, len(train_loader))
                log_every = max(1, int(total_train_steps * max(1, int(config.progress_log_interval_pct)) / 100))

                optimizer.zero_grad(set_to_none=True)
                for step, (x, y) in enumerate(train_loader, start=1):
                    x = x.to(device, non_blocking=pin_memory)
                    y = y.to(device, non_blocking=pin_memory)
                    train_samples_processed += int(x.shape[0])

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(x)
                        loss_raw = criterion(logits, y)
                        loss = loss_raw / grad_accum_steps

                    scaler.scale(loss).backward()

                    should_step = (step % grad_accum_steps == 0) or (step == total_train_steps)
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    train_loss_sum += float(loss_raw.item())
                    train_acc_sum += _binary_accuracy_from_logits(logits.detach(), y)
                    train_iou_sum += _binary_iou_from_logits(logits.detach(), y)
                    train_steps += 1

                    if step == 1 or step == total_train_steps or step % log_every == 0:
                        elapsed_epoch = time.perf_counter() - epoch_start
                        step_eta = (elapsed_epoch / step) * (total_train_steps - step)
                        logger.info(
                            "epoch %d/%d | train %d/%d (%.1f%%) | loss=%.6f | eta=%s",
                            epoch,
                            int(config.epochs),
                            step,
                            total_train_steps,
                            (100.0 * step) / total_train_steps,
                            float(loss_raw.item()),
                            _format_seconds(step_eta),
                        )
                        remaining_epochs = max(0, int(config.epochs) - epoch)
                        eta_seconds = step_eta + (remaining_epochs * (elapsed_epoch / max(1e-6, step / total_train_steps)))
                        write_report(
                            current_epoch=epoch,
                            epoch_percent=(100.0 * step) / total_train_steps,
                            eta_seconds=max(0.0, eta_seconds),
                            status_value=status,
                        )

                train_loss = train_loss_sum / max(1, train_steps)
                train_acc = train_acc_sum / max(1, train_steps)
                train_iou = train_iou_sum / max(1, train_steps)
                epoch_timings["train"] = time.perf_counter() - epoch_start
                logger.info("VAL_START | epoch=%d train_elapsed=%s", epoch, _format_seconds(epoch_timings["train"]))
                logger.info("VAL_DATALOADER_READY | epoch=%d batches=%d", epoch, len(val_loader))

                val_start = time.perf_counter()
                model.eval()
                val_loss_sum = 0.0
                val_acc_sum = 0.0
                val_iou_sum = 0.0
                val_steps = 0
                total_val_steps = max(1, len(val_loader))
                val_log_every = max(1, min(int(config.val_progress_log_every_batches), int(total_val_steps * 0.05) or 1))
                val_heartbeat = _Heartbeat(float(config.post_epoch_heartbeat_seconds), logger)
                with torch.no_grad():
                    for step, (x, y) in enumerate(val_loader, start=1):
                        x = x.to(device, non_blocking=pin_memory)
                        y = y.to(device, non_blocking=pin_memory)
                        val_samples_processed += int(x.shape[0])
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = model(x)
                            loss = criterion(logits, y)
                        val_loss_sum += float(loss.item())
                        val_acc_sum += _binary_accuracy_from_logits(logits, y)
                        val_iou_sum += _binary_iou_from_logits(logits, y)
                        val_steps += 1
                        if step == 1 or step == total_val_steps or step % val_log_every == 0:
                            elapsed = time.perf_counter() - val_start
                            eta = (elapsed / step) * (total_val_steps - step)
                            logger.info(
                                "VAL_PROGRESS | epoch=%d batch=%d/%d loss=%.6f iou=%.4f elapsed=%s eta=%s",
                                epoch,
                                step,
                                total_val_steps,
                                float(loss.item()),
                                _binary_iou_from_logits(logits, y),
                                _format_seconds(elapsed),
                                _format_seconds(eta),
                            )
                        val_heartbeat.beat(
                            "HEARTBEAT | phase=validation epoch=%d batch=%d/%d elapsed=%s",
                            epoch,
                            step,
                            total_val_steps,
                            _format_seconds(time.perf_counter() - val_start),
                        )
                epoch_timings["validation"] = time.perf_counter() - val_start
                logger.info("VAL_END | epoch=%d elapsed=%s", epoch, _format_seconds(epoch_timings["validation"]))

                logger.info("METRIC_REDUCTION_START | epoch=%d val_steps=%d", epoch, val_steps)
                val_loss = val_loss_sum / max(1, val_steps)
                val_acc = val_acc_sum / max(1, val_steps)
                val_iou = val_iou_sum / max(1, val_steps)
                logger.info("METRIC_REDUCTION_END | epoch=%d val_loss=%.6f val_iou=%.4f", epoch, val_loss, val_iou)
                epoch_runtime = time.perf_counter() - epoch_start

                export_start = time.perf_counter()
                latest_samples, missing_fixed, tracked_count = export_tracking_samples(epoch)
                epoch_timings["export_tracking"] = time.perf_counter() - export_start
                tracking_samples_processed += int(tracked_count)
                if missing_fixed:
                    logger.warning("missing fixed validation sample names for tracking: %s", ", ".join(missing_fixed))

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_iou": train_iou,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_iou": val_iou,
                    "epoch_runtime_seconds": epoch_runtime,
                    "tracked_samples": latest_samples,
                }
                history_start = time.perf_counter()
                logger.info("EPOCH_HISTORY_WRITE_START | epoch=%d path=%s", epoch, history_jsonl_path)
                history.append(record)
                _append_jsonl_logged(history_jsonl_path, record, logger=logger, context="EPOCH_HISTORY_WRITE")
                epoch_timings["history_write"] = time.perf_counter() - history_start
                logger.info("EPOCH_HISTORY_WRITE_END | epoch=%d elapsed=%s", epoch, _format_seconds(epoch_timings["history_write"]))

                checkpoint = {
                    "schema_version": _checkpoint_schema_for_architecture(architecture),
                    "created_utc": _utc_now(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "backend": backend_label,
                    "model_architecture": architecture,
                    "model_initialization": init_mode,
                    "pretrained_init": pretrained_payload,
                    "config": config_payload,
                    "config_sha256": config_sha256,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                ckpt_start = time.perf_counter()
                logger.info("CKPT_SAVE_START | epoch=%d path=%s", epoch, last_path)
                torch.save(checkpoint, last_path)
                logger.info("CKPT_SAVE_END | epoch=%d path=%s size_bytes=%d", epoch, last_path, int(last_path.stat().st_size))
                checkpoint_files.append(_to_rel(last_path, output_dir))
                if int(config.checkpoint_every) > 0 and epoch % int(config.checkpoint_every) == 0:
                    epoch_ckpt = output_dir / f"epoch_{epoch:03d}.pt"
                    logger.info("CKPT_SAVE_START | epoch=%d path=%s", epoch, epoch_ckpt)
                    torch.save(checkpoint, epoch_ckpt)
                    logger.info("CKPT_SAVE_END | epoch=%d path=%s size_bytes=%d", epoch, epoch_ckpt, int(epoch_ckpt.stat().st_size))
                    checkpoint_files.append(_to_rel(epoch_ckpt, output_dir))

                improved = val_loss < (best_val_loss - float(config.early_stopping_min_delta))
                if improved:
                    best_val_loss = val_loss
                    checkpoint["best_val_loss"] = best_val_loss
                    logger.info("CKPT_SAVE_START | epoch=%d path=%s", epoch, best_path)
                    torch.save(checkpoint, best_path)
                    logger.info("CKPT_SAVE_END | epoch=%d path=%s size_bytes=%d", epoch, best_path, int(best_path.stat().st_size))
                    checkpoint_files.append(_to_rel(best_path, output_dir))
                    no_improve = 0
                else:
                    no_improve += 1
                epoch_timings["checkpoint_save"] = time.perf_counter() - ckpt_start

                elapsed_total = time.perf_counter() - run_start
                completed = len(history)
                avg_epoch = elapsed_total / max(1, completed)
                eta_total = avg_epoch * max(0, int(config.epochs) - completed)
                logger.info(
                    "epoch %d complete | train_loss=%.6f val_loss=%.6f val_iou=%.4f | elapsed=%s | eta=%s",
                    epoch,
                    train_loss,
                    val_loss,
                    val_iou,
                    _format_seconds(elapsed_total),
                    _format_seconds(eta_total),
                )
                _append_jsonl_logged(
                    events_path,
                    {
                        "event": "epoch_completed",
                        "ts_utc": _utc_now(),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "train_iou": train_iou,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_iou": val_iou,
                        "epoch_runtime_seconds": epoch_runtime,
                        "eta_seconds": eta_total,
                    },
                    logger=logger,
                    context="EPOCH_COMPLETED_EVENT",
                )
                report_start = time.perf_counter()
                logger.info(
                    "REPORT_UPDATE_START | epoch=%d progress_epoch=%d/%d percent=100.0 path=%s",
                    epoch,
                    epoch,
                    int(config.epochs),
                    html_report_path,
                )
                write_report(
                    current_epoch=epoch,
                    epoch_percent=100.0,
                    eta_seconds=eta_total,
                    status_value=status,
                )
                epoch_timings["report_write"] = time.perf_counter() - report_start
                logger.info("REPORT_UPDATE_END | epoch=%d elapsed=%s", epoch, _format_seconds(epoch_timings["report_write"]))
                logger.info(
                    "EPOCH_TIMINGS | epoch=%d train=%s val=%s export_tracking=%s history_write=%s ckpt_save=%s report_write=%s",
                    epoch,
                    _format_seconds(epoch_timings.get("train", 0.0)),
                    _format_seconds(epoch_timings.get("validation", 0.0)),
                    _format_seconds(epoch_timings.get("export_tracking", 0.0)),
                    _format_seconds(epoch_timings.get("history_write", 0.0)),
                    _format_seconds(epoch_timings.get("checkpoint_save", 0.0)),
                    _format_seconds(epoch_timings.get("report_write", 0.0)),
                )

                if no_improve >= int(config.early_stopping_patience):
                    logger.info(
                        "early stopping triggered at epoch %d (patience=%d)",
                        epoch,
                        int(config.early_stopping_patience),
                    )
                    _append_jsonl_logged(
                        events_path,
                        {
                            "event": "early_stopping",
                            "ts_utc": _utc_now(),
                            "epoch": epoch,
                            "patience": int(config.early_stopping_patience),
                        },
                        logger=logger,
                        context="EARLY_STOPPING_EVENT",
                    )
                    break

            status = "completed"
        except KeyboardInterrupt:
            interrupted = True
            status = "interrupted"
            logger.warning("training interrupted by user; writing partial artifacts")
            _append_jsonl_logged(events_path, {"event": "interrupted", "ts_utc": _utc_now(), "epoch": len(history)}, logger=logger, context="INTERRUPTED_EVENT")
        except Exception as exc:
            status = "failed"
            error_payload = {
                "schema_version": "microseg.training_error.v1",
                "created_utc": _utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json(output_dir / "error_report.json", error_payload)
            write_report(
                current_epoch=max(start_epoch, len(history)),
                epoch_percent=0.0,
                eta_seconds=0.0,
                status_value=status,
                error=error_payload,
            )
            raise

        model_weight_stats = _model_weight_statistics(model)
        peak_memory_mb = _current_gpu_peak_memory_mb(str(device))
        if peak_memory_mb is not None:
            runtime_hardware["gpu_peak_memory_allocated_mb"] = float(peak_memory_mb)
        final_runtime = time.perf_counter() - run_start
        write_report(
            current_epoch=max(start_epoch, len(history)),
            epoch_percent=100.0 if status == "completed" else 0.0,
            eta_seconds=0.0,
            status_value=status,
        )

        model_path = best_path if best_path.exists() else last_path
        model_checkpoint_size_bytes = int(model_path.stat().st_size) if model_path.exists() and model_path.is_file() else None
        estimated_total_flops = None
        if flops_per_sample_forward is not None:
            estimated_total_flops = int(
                int(flops_per_sample_forward)
                * ((3 * int(train_samples_processed)) + int(val_samples_processed) + int(tracking_samples_processed))
            )
        manifest = {
            "schema_version": "microseg.training_manifest.v2",
            "backend": backend_label,
            "created_utc": _utc_now(),
            "status": status,
            "interrupted": interrupted,
            "code_version": _code_version(),
            "model_architecture": architecture,
            "model_initialization": init_mode,
            "pretrained_init": pretrained_payload,
            "device": device,
            "device_reason": resolved.reason,
            "config": config_payload,
            "config_sha256": config_sha256,
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "best_checkpoint": model_path.name if model_path.exists() else "",
            "last_checkpoint": last_path.name if last_path.exists() else "",
            "model_parameter_count": model_parameter_count,
            "model_trainable_parameter_count": model_trainable_parameter_count,
            "model_weight_statistics": model_weight_stats,
            "model_checkpoint_size_bytes": model_checkpoint_size_bytes,
            "model_checkpoint_size_mb": (
                float(model_checkpoint_size_bytes) / (1024.0 * 1024.0)
                if model_checkpoint_size_bytes is not None
                else None
            ),
            "runtime_hardware": runtime_hardware,
            "compute_effort": {
                "train_samples_processed": int(train_samples_processed),
                "val_samples_processed": int(val_samples_processed),
                "tracking_samples_processed": int(tracking_samples_processed),
                "estimated_forward_flops_per_sample": flops_per_sample_forward,
                "estimated_total_flops": estimated_total_flops,
                "estimated_total_tflops": (
                    float(estimated_total_flops) / 1_000_000_000_000.0
                    if estimated_total_flops is not None
                    else None
                ),
            },
            "report_path": report_path.name,
            "html_report_path": html_report_path.name if html_report_path.exists() else "",
            "runtime_seconds": final_runtime,
            "runtime_human": _format_seconds(final_runtime),
            "history": history,
        }
        manifest_path = output_dir / "training_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "backend": backend_label,
            "status": status,
            "interrupted": interrupted,
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "html_report_path": str(html_report_path) if html_report_path.exists() else "",
            "device": device,
            "model_initialization": init_mode,
            "pretrained_init": pretrained_payload,
            "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
            "epochs_completed": len(history),
            "model_parameter_count": model_parameter_count,
            "model_trainable_parameter_count": model_trainable_parameter_count,
            "model_weight_statistics": model_weight_stats,
            "runtime_hardware": runtime_hardware,
            "compute_effort": {
                "train_samples_processed": int(train_samples_processed),
                "val_samples_processed": int(val_samples_processed),
                "tracking_samples_processed": int(tracking_samples_processed),
                "estimated_forward_flops_per_sample": flops_per_sample_forward,
                "estimated_total_flops": estimated_total_flops,
                "estimated_total_tflops": (
                    float(estimated_total_flops) / 1_000_000_000_000.0
                    if estimated_total_flops is not None
                    else None
                ),
            },
        }


def load_unet_binary_model(
    checkpoint_path: str | Path,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> dict[str, Any]:
    """Load UNet checkpoint bundle with resolved runtime device."""

    import torch

    resolved = resolve_torch_device(enable_gpu=enable_gpu, policy=device_policy)
    device = resolved.selected_device

    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    schema = str(ckpt.get("schema_version", "")).strip()
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    architecture = str(
        ckpt.get(
            "model_architecture",
            cfg.get("model_architecture", cfg.get("backend", "unet_binary")),
        )
    ).strip() or "unet_binary"
    if schema == "microseg.torch_unet_binary.v1":
        architecture = "unet_binary"

    model = _build_binary_model(
        architecture=architecture,
        base_channels=int(cfg.get("model_base_channels", 16)),
        transformer_depth=int(cfg.get("transformer_depth", 2)),
        transformer_num_heads=int(cfg.get("transformer_num_heads", 4)),
        transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 2.0)),
        transformer_dropout=float(cfg.get("transformer_dropout", 0.0)),
        segformer_patch_size=int(cfg.get("segformer_patch_size", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "device": device,
        "reason": resolved.reason,
        "schema_version": schema,
        "architecture": architecture,
        "backend": str(ckpt.get("backend", cfg.get("backend", architecture))),
        "model_initialization": str(ckpt.get("model_initialization", "unknown")),
        "pretrained_init": ckpt.get("pretrained_init", {}),
        "config": cfg,
    }


def predict_unet_binary_mask(image: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    """Predict binary mask from RGB image using loaded UNet bundle."""

    import torch

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB with shape (H, W, 3)")

    model = bundle["model"]
    device = bundle["device"]

    x = torch.from_numpy((image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()[0, 0]
    return pred


def infer_image_with_unet_binary_model(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> dict[str, str]:
    """Run UNet inference for one image and export indexed prediction mask."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    bundle = load_unet_binary_model(
        checkpoint_path,
        enable_gpu=enable_gpu,
        device_policy=device_policy,
    )
    pred = predict_unet_binary_mask(image, bundle).astype(np.uint8)

    stem = Path(image_path).stem
    out_img = output_root / f"{stem}_input.png"
    out_mask = output_root / f"{stem}_prediction_indexed.png"
    Image.fromarray(image).save(out_img)
    Image.fromarray(pred).save(out_mask)

    return {
        "input": str(out_img),
        "prediction_indexed": str(out_mask),
    }
