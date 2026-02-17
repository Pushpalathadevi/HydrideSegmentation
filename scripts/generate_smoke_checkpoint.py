"""Generate deterministic tiny smoke-checkpoint files for pipeline debugging.

This utility intentionally creates a very small random-weight model artifact that
is useful only for exercising loading/inference/evaluation code paths.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate tiny smoke checkpoint (.pth)")
    p.add_argument(
        "--output",
        type=str,
        default="frozen_checkpoints/smoke/torch_pixel_smoke_random_v1.pth",
        help="Output checkpoint path",
    )
    p.add_argument("--seed", type=int, default=17, help="Random seed")
    p.add_argument(
        "--class-values",
        type=str,
        default="0,1",
        help="Comma-separated class values (e.g. 0,1 or 0,1,2)",
    )
    p.add_argument(
        "--max-size-kb",
        type=int,
        default=256,
        help="Fail if generated file exceeds this size threshold",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    return p


def _parse_class_values(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(values) < 2:
        raise ValueError("class-values must contain at least two integer classes")
    return values


def generate_torch_pixel_smoke_checkpoint(
    output_path: Path,
    *,
    seed: int,
    class_values: list[int],
    max_size_kb: int,
    force: bool,
) -> Path:
    import torch

    if output_path.exists() and not force:
        raise FileExistsError(f"output exists: {output_path} (use --force to overwrite)")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    model = torch.nn.Linear(3, len(class_values))
    with torch.no_grad():
        model.weight.uniform_(-0.05, 0.05)
        model.bias.zero_()

    checkpoint = {
        "schema_version": "microseg.torch_pixel_classifier.v1",
        "created_utc": _utc_now(),
        "config": {
            "purpose": "smoke_test_pipeline_validation",
            "seed": int(seed),
            "note": (
                "Random weights only for loading/inference/evaluation smoke tests. "
                "Not for scientific use."
            ),
        },
        "class_values": [int(v) for v in class_values],
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output_path)

    size_bytes = output_path.stat().st_size
    if size_bytes > int(max_size_kb) * 1024:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"generated checkpoint is too large ({size_bytes} bytes); "
            f"threshold={int(max_size_kb) * 1024} bytes"
        )

    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "microseg.smoke_checkpoint_meta.v1",
                "created_utc": _utc_now(),
                "path": output_path.as_posix(),
                "size_bytes": size_bytes,
                "size_kb": round(size_bytes / 1024.0, 3),
                "checkpoint_schema": checkpoint["schema_version"],
                "class_values": checkpoint["class_values"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    args = _build_parser().parse_args()
    out = Path(args.output)
    ckpt = generate_torch_pixel_smoke_checkpoint(
        out,
        seed=int(args.seed),
        class_values=_parse_class_values(args.class_values),
        max_size_kb=int(args.max_size_kb),
        force=bool(args.force),
    )
    size = ckpt.stat().st_size
    print(f"smoke checkpoint: {ckpt}")
    print(f"size_bytes: {size}")
    print(f"size_kb: {size / 1024.0:.3f}")
    print(f"metadata: {ckpt.with_suffix(ckpt.suffix + '.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

