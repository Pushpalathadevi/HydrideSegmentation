"""GA-driven HPC experiment bundle generator for training/evaluation sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_csv(value: str) -> tuple[str, ...]:
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    return tuple(items)


def _split_csv_ints(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in _split_csv(value):
        values.append(int(item))
    return tuple(values)


def _float_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("log-uniform ranges require positive bounds")
    a = math.log10(min(lo, hi))
    b = math.log10(max(lo, hi))
    return float(10 ** rng.uniform(a, b))


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _normalize_values(values: list[float], *, neutral: float = 0.5) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [neutral for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class HpcGaPlanConfig:
    """Configuration for GA-based HPC script bundle generation."""

    dataset_dir: str
    output_dir: str
    experiment_name: str = "microseg_hpc_ga_sweep"
    base_train_config: str = "configs/train.default.yml"
    base_eval_config: str = "configs/evaluate.default.yml"
    run_mode: str = "train_eval"
    eval_split: str = "val"

    architectures: tuple[str, ...] = (
        "unet_binary",
        "smp_unet_resnet18",
        "smp_deeplabv3plus_resnet101",
        "smp_unetplusplus_resnet101",
        "smp_pspnet_resnet101",
        "smp_fpn_resnet101",
        "hf_segformer_b0",
        "hf_segformer_b2",
        "hf_segformer_b5",
        "hf_upernet_swin_large",
        "transunet_tiny",
        "segformer_mini",
        "torch_pixel",
    )
    num_candidates: int = 8
    population_size: int = 24
    generations: int = 8
    mutation_rate: float = 0.20
    crossover_rate: float = 0.70
    seed: int = 42

    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2
    batch_size_choices: tuple[int, ...] = (4, 8, 16, 32)
    epochs_min: int = 8
    epochs_max: int = 40
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-3
    max_samples_min: int = 50000
    max_samples_max: int = 250000

    fitness_mode: str = "novelty"
    feedback_sources: tuple[str, ...] = ()
    feedback_min_samples: int = 3
    feedback_k: int = 5
    exploration_weight: float = 0.55
    fitness_weight_mean_iou: float = 0.50
    fitness_weight_macro_f1: float = 0.30
    fitness_weight_pixel_accuracy: float = 0.20
    fitness_weight_runtime: float = 0.05

    enable_gpu: bool = True
    device_policy: str = "auto"

    scheduler: str = "slurm"
    queue: str = ""
    account: str = ""
    qos: str = ""
    gpus_per_job: int = 1
    cpus_per_task: int = 8
    mem_gb: int = 32
    time_limit: str = "08:00:00"
    job_prefix: str = "microseg"

    python_executable: str = "python"
    microseg_cli_path: str = "scripts/microseg_cli.py"

    pretrained_init_mode: str = "scratch"
    pretrained_registry_path: str = "pre_trained_weights/registry.json"
    pretrained_model_map: dict[str, str] = field(default_factory=dict)
    pretrained_verify_sha256: bool = True
    pretrained_ignore_mismatched_sizes: bool = True
    pretrained_strict: bool = False


@dataclass(frozen=True)
class HpcGaCandidate:
    """One planned experiment candidate."""

    candidate_id: str
    backend: str
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    max_samples: int
    seed: int
    novelty_score: float
    predicted_fitness: float | None = None
    selection_score: float | None = None


@dataclass(frozen=True)
class HpcGaHistoricalSample:
    """Feedback sample loaded from previous candidate evaluation outputs."""

    source_path: str
    candidate_id: str
    backend: str
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    max_samples: int
    pixel_accuracy: float
    macro_f1: float
    mean_iou: float
    runtime_seconds: float
    fitness_score: float


@dataclass(frozen=True)
class HpcGaBundleResult:
    """Output paths and summary for generated HPC bundle."""

    bundle_dir: str
    manifest_path: str
    submit_script: str
    candidates: tuple[HpcGaCandidate, ...]


def _validate_config(cfg: HpcGaPlanConfig) -> None:
    if not str(cfg.dataset_dir).strip():
        raise ValueError("dataset_dir is required")
    if not str(cfg.output_dir).strip():
        raise ValueError("output_dir is required")
    if not cfg.architectures:
        raise ValueError("architectures must include at least one backend")
    if cfg.num_candidates <= 0:
        raise ValueError("num_candidates must be > 0")
    if cfg.population_size <= 1:
        raise ValueError("population_size must be > 1")
    if cfg.generations <= 0:
        raise ValueError("generations must be > 0")
    if not (0.0 <= cfg.mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0,1]")
    if not (0.0 <= cfg.crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be in [0,1]")
    if cfg.learning_rate_min <= 0 or cfg.learning_rate_max <= 0:
        raise ValueError("learning_rate bounds must be > 0")
    if cfg.weight_decay_min <= 0 or cfg.weight_decay_max <= 0:
        raise ValueError("weight_decay bounds must be > 0")
    if cfg.epochs_min <= 0 or cfg.epochs_max <= 0:
        raise ValueError("epoch bounds must be > 0")
    if cfg.max_samples_min <= 0 or cfg.max_samples_max <= 0:
        raise ValueError("max_samples bounds must be > 0")
    if not cfg.batch_size_choices:
        raise ValueError("batch_size_choices must contain at least one value")
    if cfg.scheduler not in {"slurm", "pbs", "local"}:
        raise ValueError("scheduler must be one of: slurm, pbs, local")
    if cfg.run_mode not in {"train_only", "train_eval"}:
        raise ValueError("run_mode must be one of: train_only, train_eval")
    if cfg.fitness_mode not in {"novelty", "feedback_hybrid"}:
        raise ValueError("fitness_mode must be one of: novelty, feedback_hybrid")
    if not (0.0 <= cfg.exploration_weight <= 1.0):
        raise ValueError("exploration_weight must be in [0,1]")
    if cfg.feedback_min_samples < 1:
        raise ValueError("feedback_min_samples must be >= 1")
    if cfg.feedback_k < 1:
        raise ValueError("feedback_k must be >= 1")
    if cfg.pretrained_init_mode not in {"scratch", "auto", "local"}:
        raise ValueError("pretrained_init_mode must be one of: scratch, auto, local")
    if cfg.pretrained_init_mode == "local":
        missing = [
            arch
            for arch in cfg.architectures
            if _architecture_supports_pretrained(arch)
            and not str(cfg.pretrained_model_map.get(arch, "")).strip()
        ]
        if missing:
            raise ValueError(
                "pretrained_init_mode=local requires pretrained_model_map entries for architectures: "
                + ", ".join(sorted(missing))
            )
    for backend, model_id in cfg.pretrained_model_map.items():
        if not str(model_id).strip():
            raise ValueError(f"pretrained_model_map has empty model_id for backend: {backend}")


def _architecture_supports_pretrained(architecture: str) -> bool:
    arch = str(architecture).strip().lower()
    if arch in {"unet_binary", "transunet_tiny", "segformer_mini"}:
        return True
    return arch.startswith("hf_segformer_") or arch.startswith("hf_upernet_") or arch.startswith("smp_")


def _candidate_pretrained_overrides(cfg: HpcGaPlanConfig, candidate: HpcGaCandidate) -> tuple[str, ...]:
    mode = str(cfg.pretrained_init_mode).strip().lower()
    backend = str(candidate.backend).strip()

    if mode == "scratch" or not _architecture_supports_pretrained(backend):
        return (
            "pretrained_init_mode=scratch",
            f"pretrained_registry_path={cfg.pretrained_registry_path}",
            "pretrained_model_id=",
        )

    model_id = str(cfg.pretrained_model_map.get(backend, "")).strip()
    if mode == "local" and not model_id:
        raise ValueError(
            f"candidate backend '{backend}' requires pretrained_model_map entry when pretrained_init_mode=local"
        )
    if mode == "auto" and not model_id:
        return (
            "pretrained_init_mode=scratch",
            f"pretrained_registry_path={cfg.pretrained_registry_path}",
            "pretrained_model_id=",
        )

    return (
        "pretrained_init_mode=local",
        f"pretrained_model_id={model_id}",
        f"pretrained_registry_path={cfg.pretrained_registry_path}",
        f"pretrained_verify_sha256={str(bool(cfg.pretrained_verify_sha256)).lower()}",
        f"pretrained_ignore_mismatched_sizes={str(bool(cfg.pretrained_ignore_mismatched_sizes)).lower()}",
        f"pretrained_strict={str(bool(cfg.pretrained_strict)).lower()}",
    )


def _sample_candidate(rng: random.Random, cfg: HpcGaPlanConfig, idx: int, score: float = 0.0) -> HpcGaCandidate:
    backend = cfg.architectures[rng.randrange(len(cfg.architectures))]
    lr = _float_log_uniform(rng, cfg.learning_rate_min, cfg.learning_rate_max)
    wd = _float_log_uniform(rng, cfg.weight_decay_min, cfg.weight_decay_max)
    batch = cfg.batch_size_choices[rng.randrange(len(cfg.batch_size_choices))]
    epochs = rng.randint(min(cfg.epochs_min, cfg.epochs_max), max(cfg.epochs_min, cfg.epochs_max))
    max_samples = rng.randint(min(cfg.max_samples_min, cfg.max_samples_max), max(cfg.max_samples_min, cfg.max_samples_max))
    return HpcGaCandidate(
        candidate_id=f"cand_{idx:03d}",
        backend=backend,
        learning_rate=float(lr),
        batch_size=int(batch),
        epochs=int(epochs),
        weight_decay=float(wd),
        max_samples=int(max_samples),
        seed=int(cfg.seed + idx),
        novelty_score=float(score),
    )


def _candidate_features(c: HpcGaCandidate, cfg: HpcGaPlanConfig) -> tuple[float, ...]:
    arch_idx = cfg.architectures.index(c.backend)
    arch_norm = 0.0 if len(cfg.architectures) == 1 else arch_idx / float(len(cfg.architectures) - 1)
    lr_lo = math.log10(min(cfg.learning_rate_min, cfg.learning_rate_max))
    lr_hi = math.log10(max(cfg.learning_rate_min, cfg.learning_rate_max))
    wd_lo = math.log10(min(cfg.weight_decay_min, cfg.weight_decay_max))
    wd_hi = math.log10(max(cfg.weight_decay_min, cfg.weight_decay_max))
    lr_norm = (math.log10(c.learning_rate) - lr_lo) / max(1e-12, (lr_hi - lr_lo))
    wd_norm = (math.log10(c.weight_decay) - wd_lo) / max(1e-12, (wd_hi - wd_lo))
    e_lo, e_hi = min(cfg.epochs_min, cfg.epochs_max), max(cfg.epochs_min, cfg.epochs_max)
    ms_lo, ms_hi = min(cfg.max_samples_min, cfg.max_samples_max), max(cfg.max_samples_min, cfg.max_samples_max)
    b_sorted = tuple(sorted(set(cfg.batch_size_choices)))
    b_idx = b_sorted.index(c.batch_size)
    b_norm = 0.0 if len(b_sorted) == 1 else b_idx / float(len(b_sorted) - 1)
    e_norm = (c.epochs - e_lo) / max(1, (e_hi - e_lo))
    ms_norm = (c.max_samples - ms_lo) / max(1, (ms_hi - ms_lo))
    return (arch_norm, lr_norm, b_norm, e_norm, wd_norm, ms_norm)


def _feature_tuple_from_values(
    *,
    backend: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    max_samples: int,
    cfg: HpcGaPlanConfig,
) -> tuple[float, ...]:
    proxy = HpcGaCandidate(
        candidate_id="proxy",
        backend=backend,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        max_samples=max_samples,
        seed=cfg.seed,
        novelty_score=0.0,
    )
    return _candidate_features(proxy, cfg)


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b))))


def _novelty_scores(pop: list[HpcGaCandidate], cfg: HpcGaPlanConfig) -> list[float]:
    if len(pop) <= 1:
        return [0.0 for _ in pop]
    feats = [_candidate_features(c, cfg) for c in pop]
    k = min(3, len(pop) - 1)
    scores: list[float] = []
    for i, fi in enumerate(feats):
        dists = sorted(_euclidean(fi, fj) for j, fj in enumerate(feats) if j != i)
        base = float(sum(dists[:k]) / max(1, k))
        arch_bonus = 0.05 if len({c.backend for c in pop}) > 1 else 0.0
        scores.append(base + arch_bonus)
    return scores


def _select_tournament(pop: list[HpcGaCandidate], scores: list[float], rng: random.Random) -> HpcGaCandidate:
    i = rng.randrange(len(pop))
    j = rng.randrange(len(pop))
    return pop[i] if scores[i] >= scores[j] else pop[j]


def _crossover(a: HpcGaCandidate, b: HpcGaCandidate, rng: random.Random, new_id: str) -> HpcGaCandidate:
    return HpcGaCandidate(
        candidate_id=new_id,
        backend=a.backend if rng.random() < 0.5 else b.backend,
        learning_rate=a.learning_rate if rng.random() < 0.5 else b.learning_rate,
        batch_size=a.batch_size if rng.random() < 0.5 else b.batch_size,
        epochs=a.epochs if rng.random() < 0.5 else b.epochs,
        weight_decay=a.weight_decay if rng.random() < 0.5 else b.weight_decay,
        max_samples=a.max_samples if rng.random() < 0.5 else b.max_samples,
        seed=a.seed if rng.random() < 0.5 else b.seed,
        novelty_score=0.0,
    )


def _mutate(c: HpcGaCandidate, cfg: HpcGaPlanConfig, rng: random.Random, new_id: str) -> HpcGaCandidate:
    backend = c.backend
    lr = c.learning_rate
    bs = c.batch_size
    epochs = c.epochs
    wd = c.weight_decay
    ms = c.max_samples
    seed = c.seed

    if rng.random() < cfg.mutation_rate:
        backend = cfg.architectures[rng.randrange(len(cfg.architectures))]
    if rng.random() < cfg.mutation_rate:
        lr = _float_log_uniform(rng, cfg.learning_rate_min, cfg.learning_rate_max)
    if rng.random() < cfg.mutation_rate:
        wd = _float_log_uniform(rng, cfg.weight_decay_min, cfg.weight_decay_max)
    if rng.random() < cfg.mutation_rate:
        bs = cfg.batch_size_choices[rng.randrange(len(cfg.batch_size_choices))]
    if rng.random() < cfg.mutation_rate:
        epochs += rng.randint(-4, 4)
    if rng.random() < cfg.mutation_rate:
        ms += rng.randint(-5000, 5000)
    if rng.random() < cfg.mutation_rate:
        seed += rng.randint(1, 101)

    epochs = int(_clip(float(epochs), float(min(cfg.epochs_min, cfg.epochs_max)), float(max(cfg.epochs_min, cfg.epochs_max))))
    ms = int(_clip(float(ms), float(min(cfg.max_samples_min, cfg.max_samples_max)), float(max(cfg.max_samples_min, cfg.max_samples_max))))
    return HpcGaCandidate(
        candidate_id=new_id,
        backend=backend,
        learning_rate=float(lr),
        batch_size=int(bs),
        epochs=int(epochs),
        weight_decay=float(wd),
        max_samples=int(ms),
        seed=int(seed),
        novelty_score=0.0,
    )


def _candidate_key(c: HpcGaCandidate) -> tuple[Any, ...]:
    return (
        c.backend,
        round(c.learning_rate, 8),
        c.batch_size,
        c.epochs,
        round(c.weight_decay, 9),
        c.max_samples,
    )


def _extract_bundle_root(path: Path) -> Path | None:
    if path.is_dir():
        return path
    if path.is_file() and path.name == "ga_plan_manifest.json":
        return path.parent
    return None


def _fitness_from_metrics(
    *,
    pixel_accuracy: float,
    macro_f1: float,
    mean_iou: float,
    runtime_norm: float,
    cfg: HpcGaPlanConfig,
) -> float:
    score = 0.0
    score += float(cfg.fitness_weight_mean_iou) * float(mean_iou)
    score += float(cfg.fitness_weight_macro_f1) * float(macro_f1)
    score += float(cfg.fitness_weight_pixel_accuracy) * float(pixel_accuracy)
    score -= float(cfg.fitness_weight_runtime) * float(runtime_norm)
    return float(score)


def load_feedback_samples(
    feedback_sources: tuple[str, ...],
    *,
    cfg: HpcGaPlanConfig,
) -> list[HpcGaHistoricalSample]:
    """Load feedback samples from previously completed HPC GA bundles."""

    if not feedback_sources:
        return []

    raw: list[dict[str, Any]] = []
    for src in feedback_sources:
        source_text = str(src).strip()
        if not source_text:
            continue
        path = Path(source_text)
        bundle_root = _extract_bundle_root(path)
        if bundle_root is None:
            continue
        candidates_dir = bundle_root / "candidates"
        runs_dir = bundle_root / "runs"
        if not candidates_dir.exists() or not runs_dir.exists():
            continue
        for candidate_json in sorted(candidates_dir.glob("cand_*.json")):
            try:
                c_payload = json.loads(candidate_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            cid = str(c_payload.get("candidate_id", "")).strip() or candidate_json.stem
            report = runs_dir / cid / "eval_report.json"
            if not report.exists():
                continue
            try:
                r_payload = json.loads(report.read_text(encoding="utf-8"))
            except Exception:
                continue
            metrics = r_payload.get("metrics", {})
            if not isinstance(metrics, dict):
                continue

            pa = _coerce_float(metrics.get("pixel_accuracy"), default=float("nan"))
            f1 = _coerce_float(metrics.get("macro_f1"), default=float("nan"))
            mi = _coerce_float(metrics.get("mean_iou"), default=float("nan"))
            rt = _coerce_float(r_payload.get("runtime_seconds"), default=float("nan"))
            if any(math.isnan(v) for v in (pa, f1, mi, rt)):
                continue

            try:
                backend = str(c_payload["backend"])
                lr = float(c_payload["learning_rate"])
                bs = int(c_payload["batch_size"])
                epochs = int(c_payload["epochs"])
                wd = float(c_payload["weight_decay"])
                ms = int(c_payload["max_samples"])
            except Exception:
                continue

            if backend not in cfg.architectures:
                continue
            raw.append(
                {
                    "source_path": report.as_posix(),
                    "candidate_id": cid,
                    "backend": backend,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "epochs": epochs,
                    "weight_decay": wd,
                    "max_samples": ms,
                    "pixel_accuracy": pa,
                    "macro_f1": f1,
                    "mean_iou": mi,
                    "runtime_seconds": rt,
                }
            )

    if not raw:
        return []
    runtime_vals = [float(item["runtime_seconds"]) for item in raw]
    runtime_norm = _normalize_values(runtime_vals, neutral=0.0)

    out: list[HpcGaHistoricalSample] = []
    for item, rt_norm in zip(raw, runtime_norm):
        fit = _fitness_from_metrics(
            pixel_accuracy=float(item["pixel_accuracy"]),
            macro_f1=float(item["macro_f1"]),
            mean_iou=float(item["mean_iou"]),
            runtime_norm=float(rt_norm),
            cfg=cfg,
        )
        out.append(
            HpcGaHistoricalSample(
                source_path=str(item["source_path"]),
                candidate_id=str(item["candidate_id"]),
                backend=str(item["backend"]),
                learning_rate=float(item["learning_rate"]),
                batch_size=int(item["batch_size"]),
                epochs=int(item["epochs"]),
                weight_decay=float(item["weight_decay"]),
                max_samples=int(item["max_samples"]),
                pixel_accuracy=float(item["pixel_accuracy"]),
                macro_f1=float(item["macro_f1"]),
                mean_iou=float(item["mean_iou"]),
                runtime_seconds=float(item["runtime_seconds"]),
                fitness_score=float(fit),
            )
        )
    return out


def summarize_feedback_sources(
    feedback_sources: tuple[str, ...],
    *,
    cfg: HpcGaPlanConfig,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize prior feedback sources for operator review."""

    samples = load_feedback_samples(feedback_sources, cfg=cfg)
    ranked = sorted(samples, key=lambda s: s.fitness_score, reverse=True)
    top = ranked[: max(0, int(top_k))]
    return {
        "schema_version": "microseg.hpc_ga_feedback_summary.v1",
        "created_utc": _utc_now(),
        "fitness_mode": cfg.fitness_mode,
        "sample_count": len(samples),
        "sources": list(feedback_sources),
        "weights": {
            "mean_iou": cfg.fitness_weight_mean_iou,
            "macro_f1": cfg.fitness_weight_macro_f1,
            "pixel_accuracy": cfg.fitness_weight_pixel_accuracy,
            "runtime": cfg.fitness_weight_runtime,
        },
        "top_candidates": [
            {
                "candidate_id": s.candidate_id,
                "backend": s.backend,
                "fitness_score": s.fitness_score,
                "mean_iou": s.mean_iou,
                "macro_f1": s.macro_f1,
                "pixel_accuracy": s.pixel_accuracy,
                "runtime_seconds": s.runtime_seconds,
                "source_path": s.source_path,
            }
            for s in top
        ],
    }


def _predict_fitness_knn(
    candidate: HpcGaCandidate,
    history: list[HpcGaHistoricalSample],
    *,
    cfg: HpcGaPlanConfig,
) -> float | None:
    if not history:
        return None
    cf = _candidate_features(candidate, cfg)
    d_fit: list[tuple[float, float]] = []
    for s in history:
        hf = _feature_tuple_from_values(
            backend=s.backend,
            learning_rate=s.learning_rate,
            batch_size=s.batch_size,
            epochs=s.epochs,
            weight_decay=s.weight_decay,
            max_samples=s.max_samples,
            cfg=cfg,
        )
        d_fit.append((_euclidean(cf, hf), float(s.fitness_score)))
    d_fit.sort(key=lambda item: item[0])
    k = min(max(1, int(cfg.feedback_k)), len(d_fit))
    top = d_fit[:k]
    weighted_sum = 0.0
    weight_sum = 0.0
    for dist, fit in top:
        w = 1.0 / (dist + 1e-6)
        weighted_sum += w * fit
        weight_sum += w
    if weight_sum <= 0.0:
        return None
    return float(weighted_sum / weight_sum)


def _compute_selection_scores(
    pop: list[HpcGaCandidate],
    *,
    cfg: HpcGaPlanConfig,
    history: list[HpcGaHistoricalSample],
) -> tuple[list[float], list[float | None], list[float]]:
    novelty = _novelty_scores(pop, cfg)
    predicted = [_predict_fitness_knn(c, history, cfg=cfg) for c in pop]

    use_feedback = cfg.fitness_mode == "feedback_hybrid" and len(history) >= int(cfg.feedback_min_samples)
    if not use_feedback:
        return novelty, predicted, novelty

    novelty_norm = _normalize_values(novelty, neutral=0.5)
    pred_vals = [0.0 if p is None else float(p) for p in predicted]
    pred_norm = _normalize_values(pred_vals, neutral=0.5)
    exp = float(cfg.exploration_weight)
    combined = [(exp * n) + ((1.0 - exp) * p) for n, p in zip(novelty_norm, pred_norm)]
    return novelty, predicted, combined


def plan_hpc_ga_candidates(cfg: HpcGaPlanConfig) -> list[HpcGaCandidate]:
    """Plan candidate experiments with novelty or feedback-hybrid GA ranking."""

    _validate_config(cfg)
    history = load_feedback_samples(cfg.feedback_sources, cfg=cfg)
    rng = random.Random(int(cfg.seed))
    pop = [_sample_candidate(rng, cfg, idx=i + 1) for i in range(max(cfg.population_size, cfg.num_candidates))]

    serial = len(pop) + 1
    for _gen in range(int(cfg.generations)):
        novelty, predicted, selection = _compute_selection_scores(pop, cfg=cfg, history=history)
        rank = sorted(range(len(pop)), key=lambda i: selection[i], reverse=True)
        elite_count = max(2, int(len(pop) * 0.2))
        next_pop: list[HpcGaCandidate] = [
            HpcGaCandidate(
                **{
                    **asdict(pop[i]),
                    "candidate_id": f"cand_{k + 1:03d}",
                    "novelty_score": float(novelty[i]),
                    "predicted_fitness": predicted[i],
                    "selection_score": float(selection[i]),
                }
            )
            for k, i in enumerate(rank[:elite_count])
        ]
        while len(next_pop) < len(pop):
            pa = _select_tournament(pop, selection, rng)
            pb = _select_tournament(pop, selection, rng)
            child_id = f"cand_{serial:03d}"
            serial += 1
            if rng.random() < cfg.crossover_rate:
                child = _crossover(pa, pb, rng, child_id)
            else:
                child = HpcGaCandidate(**{**asdict(pa), "candidate_id": child_id, "novelty_score": 0.0})
            child = _mutate(child, cfg, rng, child_id)
            next_pop.append(child)
        pop = next_pop

    novelty, predicted, selection = _compute_selection_scores(pop, cfg=cfg, history=history)
    ranked = sorted(range(len(pop)), key=lambda i: selection[i], reverse=True)
    selected: list[HpcGaCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for idx in ranked:
        cand = pop[idx]
        key = _candidate_key(cand)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            HpcGaCandidate(
                **{
                    **asdict(cand),
                    "novelty_score": float(novelty[idx]),
                    "predicted_fitness": predicted[idx],
                    "selection_score": float(selection[idx]),
                }
            )
        )
        if len(selected) >= cfg.num_candidates:
            break

    selected = [HpcGaCandidate(**{**asdict(c), "candidate_id": f"cand_{i + 1:03d}"}) for i, c in enumerate(selected)]
    return selected


def _script_header(cfg: HpcGaPlanConfig, candidate: HpcGaCandidate) -> list[str]:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if cfg.scheduler == "slurm":
        lines.extend(
            [
                f"#SBATCH --job-name={cfg.job_prefix}_{candidate.candidate_id}",
                f"#SBATCH --gres=gpu:{cfg.gpus_per_job}",
                f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
                f"#SBATCH --mem={cfg.mem_gb}G",
                f"#SBATCH --time={cfg.time_limit}",
            ]
        )
        if cfg.queue:
            lines.append(f"#SBATCH -p {cfg.queue}")
        if cfg.account:
            lines.append(f"#SBATCH -A {cfg.account}")
        if cfg.qos:
            lines.append(f"#SBATCH --qos={cfg.qos}")
        lines.append("")
    elif cfg.scheduler == "pbs":
        lines.extend(
            [
                f"#PBS -N {cfg.job_prefix}_{candidate.candidate_id}",
                f"#PBS -l select=1:ncpus={cfg.cpus_per_task}:ngpus={cfg.gpus_per_job}:mem={cfg.mem_gb}gb",
                f"#PBS -l walltime={cfg.time_limit}",
            ]
        )
        if cfg.queue:
            lines.append(f"#PBS -q {cfg.queue}")
        if cfg.account:
            lines.append(f"#PBS -A {cfg.account}")
        lines.append("")
    lines.extend(
        [
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"',
            'REPO_ROOT="${REPO_ROOT:-$(cd "${BUNDLE_ROOT}/../.." && pwd)}"',
            "",
            'resolve_path() {',
            '  local p="$1"',
            '  if [[ "$p" == /* ]]; then',
            '    printf "%s" "$p"',
            "  else",
            '    printf "%s/%s" "$REPO_ROOT" "$p"',
            "  fi",
            "}",
            "",
            f'PYTHON_EXE="{cfg.python_executable}"',
            f'MICROSEG_CLI="$(resolve_path "{cfg.microseg_cli_path}")"',
            f'TRAIN_CONFIG="$(resolve_path "{cfg.base_train_config}")"',
            f'EVAL_CONFIG="$(resolve_path "{cfg.base_eval_config}")"',
            f'DATASET_DIR="{cfg.dataset_dir}"',
            f'RUN_ROOT="${{BUNDLE_ROOT}}/runs/{candidate.candidate_id}"',
            'mkdir -p "${RUN_ROOT}"',
            "",
        ]
    )
    return lines


def _job_script_lines(cfg: HpcGaPlanConfig, candidate: HpcGaCandidate) -> list[str]:
    lines = _script_header(cfg, candidate)
    pretrained_overrides = _candidate_pretrained_overrides(cfg, candidate)
    lines.extend(
        [
            "TRAIN_CMD=(",
            '  "${PYTHON_EXE}"',
            '  "${MICROSEG_CLI}"',
            "  train",
            '  "--config" "${TRAIN_CONFIG}"',
            '  "--dataset-dir" "${DATASET_DIR}"',
            '  "--output-dir" "${RUN_ROOT}"',
            f'  "--set" "backend={candidate.backend}"',
            f'  "--set" "learning_rate={candidate.learning_rate:.8g}"',
            f'  "--set" "batch_size={candidate.batch_size}"',
            f'  "--set" "epochs={candidate.epochs}"',
            f'  "--set" "weight_decay={candidate.weight_decay:.8g}"',
            f'  "--set" "max_samples={candidate.max_samples}"',
            f'  "--set" "seed={candidate.seed}"',
            f'  "--set" "model_architecture={candidate.backend}"',
            f'  "--set" "enable_gpu={str(bool(cfg.enable_gpu)).lower()}"',
            f'  "--set" "device_policy={cfg.device_policy}"',
            "  --no-auto-prepare-dataset",
            ")",
            'echo "[HPC-GA] Running training for ' + candidate.candidate_id + '"',
            '"${TRAIN_CMD[@]}"',
            "",
        ]
    )
    for override in pretrained_overrides:
        lines.insert(
            lines.index("  --no-auto-prepare-dataset"),
            f'  "--set" "{override}"',
        )
    if cfg.run_mode == "train_eval":
        lines.extend(
            [
                'MODEL_PATH="${RUN_ROOT}/best_checkpoint.pt"',
                'if [[ ! -f "${MODEL_PATH}" ]]; then',
                '  MODEL_PATH="${RUN_ROOT}/last_checkpoint.pt"',
                "fi",
                'if [[ ! -f "${MODEL_PATH}" ]]; then',
                '  MODEL_PATH="${RUN_ROOT}/torch_pixel_classifier.pt"',
                "fi",
                "",
                "EVAL_CMD=(",
                '  "${PYTHON_EXE}"',
                '  "${MICROSEG_CLI}"',
                "  evaluate",
                '  "--config" "${EVAL_CONFIG}"',
                '  "--dataset-dir" "${DATASET_DIR}"',
                '  "--model-path" "${MODEL_PATH}"',
                f'  "--split" "{cfg.eval_split}"',
                f'  "--set" "enable_gpu={str(bool(cfg.enable_gpu)).lower()}"',
                f'  "--set" "device_policy={cfg.device_policy}"',
                '  "--output-path" "${RUN_ROOT}/eval_report.json"',
                "  --no-auto-prepare-dataset",
                ")",
                'echo "[HPC-GA] Running evaluation for ' + candidate.candidate_id + '"',
                '"${EVAL_CMD[@]}"',
                "",
            ]
        )
    lines.append('echo "[HPC-GA] Completed ' + candidate.candidate_id + '"')
    return lines


def _submit_all_lines(cfg: HpcGaPlanConfig) -> list[str]:
    cmd = "bash"
    if cfg.scheduler == "slurm":
        cmd = "sbatch"
    elif cfg.scheduler == "pbs":
        cmd = "qsub"
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'JOB_DIR="${SCRIPT_DIR}/jobs"',
        'echo "[HPC-GA] Submitting jobs from ${JOB_DIR}"',
        'for job in "${JOB_DIR}"/*.sh; do',
        '  echo "  -> ' + cmd + ' ${job}"',
        "  " + cmd + ' "${job}"',
        "done",
        'echo "[HPC-GA] Submission complete"',
    ]


def _candidate_json_payload(candidate: HpcGaCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["schema_version"] = "microseg.hpc_ga_candidate.v1"
    payload["created_utc"] = _utc_now()
    return payload


def _candidate_yaml_text(candidate: HpcGaCandidate) -> str:
    lines = [
        "schema_version: microseg.hpc_ga_candidate.v1",
        f"candidate_id: {candidate.candidate_id}",
        f"backend: {candidate.backend}",
        f"learning_rate: {candidate.learning_rate:.8g}",
        f"batch_size: {candidate.batch_size}",
        f"epochs: {candidate.epochs}",
        f"weight_decay: {candidate.weight_decay:.8g}",
        f"max_samples: {candidate.max_samples}",
        f"seed: {candidate.seed}",
        f"novelty_score: {candidate.novelty_score:.8g}",
    ]
    if candidate.predicted_fitness is not None:
        lines.append(f"predicted_fitness: {candidate.predicted_fitness:.8g}")
    if candidate.selection_score is not None:
        lines.append(f"selection_score: {candidate.selection_score:.8g}")
    return "\n".join(lines) + "\n"


def generate_hpc_ga_bundle(cfg: HpcGaPlanConfig) -> HpcGaBundleResult:
    """Generate GA candidates and scheduler scripts for HPC execution."""

    _validate_config(cfg)
    feedback_summary = summarize_feedback_sources(cfg.feedback_sources, cfg=cfg, top_k=5) if cfg.feedback_sources else None
    selected = plan_hpc_ga_candidates(cfg)
    out_root = Path(cfg.output_dir)
    jobs_dir = out_root / "jobs"
    runs_dir = out_root / "runs"
    candidates_dir = out_root / "candidates"
    for path in (jobs_dir, runs_dir, candidates_dir):
        path.mkdir(parents=True, exist_ok=True)

    scripts_written: list[str] = []
    for cand in selected:
        job = jobs_dir / f"{cand.candidate_id}.sh"
        job.write_text("\n".join(_job_script_lines(cfg, cand)) + "\n", encoding="utf-8")
        job.chmod(0o755)
        scripts_written.append(job.as_posix())

        c_json = candidates_dir / f"{cand.candidate_id}.json"
        c_json.write_text(json.dumps(_candidate_json_payload(cand), indent=2), encoding="utf-8")
        c_yaml = candidates_dir / f"{cand.candidate_id}.yml"
        c_yaml.write_text(_candidate_yaml_text(cand), encoding="utf-8")

    submit = out_root / "submit_all.sh"
    submit.write_text("\n".join(_submit_all_lines(cfg)) + "\n", encoding="utf-8")
    submit.chmod(0o755)

    quick_lines = [
        "MicroSeg HPC GA Bundle",
        "",
        "1) Upload this bundle directory to your HPC workspace.",
        "2) Ensure your environment has Python + project dependencies.",
        "3) Set REPO_ROOT when launching if repository is not adjacent to bundle:",
        "   REPO_ROOT=/path/to/HydrideSegmentation ./submit_all.sh",
        "4) For scheduler='local', submit_all.sh runs job scripts sequentially.",
        "",
        f"Scheduler: {cfg.scheduler}",
        f"Candidates: {len(selected)}",
        f"Run mode: {cfg.run_mode}",
        f"Fitness mode: {cfg.fitness_mode}",
        f"Pretrained mode: {cfg.pretrained_init_mode}",
    ]
    if feedback_summary is not None:
        quick_lines.append(f"Feedback samples available: {int(feedback_summary.get('sample_count', 0))}")
    quickstart = out_root / "README.txt"
    quickstart.write_text("\n".join(quick_lines) + "\n", encoding="utf-8")

    manifest = out_root / "ga_plan_manifest.json"
    payload = {
        "schema_version": "microseg.hpc_ga_bundle.v1",
        "created_utc": _utc_now(),
        "experiment_name": cfg.experiment_name,
        "config": asdict(cfg),
        "feedback_summary": feedback_summary,
        "candidates": [asdict(c) for c in selected],
        "paths": {
            "bundle_dir": out_root.as_posix(),
            "submit_script": submit.as_posix(),
            "jobs_dir": jobs_dir.as_posix(),
            "runs_dir": runs_dir.as_posix(),
            "candidates_dir": candidates_dir.as_posix(),
            "quickstart": quickstart.as_posix(),
            "scripts_written": scripts_written,
        },
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return HpcGaBundleResult(
        bundle_dir=out_root.as_posix(),
        manifest_path=manifest.as_posix(),
        submit_script=submit.as_posix(),
        candidates=tuple(selected),
    )


def parse_architectures(value: object) -> tuple[str, ...]:
    """Parse architecture list from config/CLI input."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return _split_csv(str(value))


def parse_batch_sizes(value: object) -> tuple[int, ...]:
    """Parse batch-size choices from config/CLI input."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return _split_csv_ints(str(value))


def parse_feedback_sources(value: object) -> tuple[str, ...]:
    """Parse feedback source list from config/CLI input."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return _split_csv(str(value))


def parse_pretrained_model_map(value: object) -> dict[str, str]:
    """Parse backend->pretrained-model-id map from config/CLI input."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(k).strip(): str(v).strip()
            for k, v in value.items()
            if str(k).strip() and str(v).strip()
        }
    text = str(value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return {
                str(k).strip(): str(v).strip()
                for k, v in payload.items()
                if str(k).strip() and str(v).strip()
            }
    except Exception:
        payload = None
    pairs: dict[str, str] = {}
    for item in _split_csv(text):
        if "=" not in item:
            raise ValueError(
                "pretrained_model_map entries must use backend=model_id format; "
                f"invalid item: {item!r}"
            )
        backend, model_id = item.split("=", 1)
        b = str(backend).strip()
        m = str(model_id).strip()
        if not b or not m:
            raise ValueError(
                "pretrained_model_map entries must include non-empty backend and model_id; "
                f"invalid item: {item!r}"
            )
        pairs[b] = m
    return pairs
