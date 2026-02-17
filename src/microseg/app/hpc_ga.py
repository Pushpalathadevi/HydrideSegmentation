"""GA-driven HPC experiment bundle generator for training/evaluation sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class HpcGaPlanConfig:
    """Configuration for GA-based HPC script bundle generation.

    Parameters
    ----------
    dataset_dir:
        Dataset directory for training jobs.
    output_dir:
        Bundle output directory.
    architectures:
        Candidate model backends to compare.
    """

    dataset_dir: str
    output_dir: str
    experiment_name: str = "microseg_hpc_ga_sweep"
    base_train_config: str = "configs/train.default.yml"
    base_eval_config: str = "configs/evaluate.default.yml"
    run_mode: str = "train_eval"
    eval_split: str = "val"

    architectures: tuple[str, ...] = ("unet_binary", "torch_pixel")
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


def plan_hpc_ga_candidates(cfg: HpcGaPlanConfig) -> list[HpcGaCandidate]:
    """Plan candidate experiments with a novelty-oriented GA.

    Parameters
    ----------
    cfg:
        Planner configuration.

    Returns
    -------
    list[HpcGaCandidate]
        Planned candidate configurations.
    """

    _validate_config(cfg)
    rng = random.Random(int(cfg.seed))
    pop = [_sample_candidate(rng, cfg, idx=i + 1) for i in range(max(cfg.population_size, cfg.num_candidates))]

    serial = len(pop) + 1
    for _gen in range(int(cfg.generations)):
        scores = _novelty_scores(pop, cfg)
        rank = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        elite_count = max(2, int(len(pop) * 0.2))
        next_pop: list[HpcGaCandidate] = [
            HpcGaCandidate(**{**asdict(pop[i]), "candidate_id": f"cand_{k + 1:03d}", "novelty_score": scores[i]})
            for k, i in enumerate(rank[:elite_count])
        ]
        while len(next_pop) < len(pop):
            pa = _select_tournament(pop, scores, rng)
            pb = _select_tournament(pop, scores, rng)
            child_id = f"cand_{serial:03d}"
            serial += 1
            if rng.random() < cfg.crossover_rate:
                child = _crossover(pa, pb, rng, child_id)
            else:
                child = HpcGaCandidate(**{**asdict(pa), "candidate_id": child_id, "novelty_score": 0.0})
            child = _mutate(child, cfg, rng, child_id)
            next_pop.append(child)
        pop = next_pop

    final_scores = _novelty_scores(pop, cfg)
    ranked = sorted(range(len(pop)), key=lambda i: final_scores[i], reverse=True)
    selected: list[HpcGaCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for idx in ranked:
        cand = pop[idx]
        key = _candidate_key(cand)
        if key in seen:
            continue
        seen.add(key)
        selected.append(HpcGaCandidate(**{**asdict(cand), "novelty_score": float(final_scores[idx])}))
        if len(selected) >= cfg.num_candidates:
            break

    selected = [
        HpcGaCandidate(**{**asdict(c), "candidate_id": f"cand_{i + 1:03d}"})
        for i, c in enumerate(selected)
    ]
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
            f'  "--set" "enable_gpu={str(bool(cfg.enable_gpu)).lower()}"',
            f'  "--set" "device_policy={cfg.device_policy}"',
            "  --no-auto-prepare-dataset",
            ")",
            'echo "[HPC-GA] Running training for ' + candidate.candidate_id + '"',
            '"${TRAIN_CMD[@]}"',
            "",
        ]
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
    return "\n".join(lines) + "\n"


def generate_hpc_ga_bundle(cfg: HpcGaPlanConfig) -> HpcGaBundleResult:
    """Generate GA candidates and scheduler scripts for HPC execution.

    Parameters
    ----------
    cfg:
        Planner configuration.

    Returns
    -------
    HpcGaBundleResult
        Output paths and selected candidates.
    """

    _validate_config(cfg)
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

    quickstart = out_root / "README.txt"
    quickstart.write_text(
        "\n".join(
            [
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = out_root / "ga_plan_manifest.json"
    payload = {
        "schema_version": "microseg.hpc_ga_bundle.v1",
        "created_utc": _utc_now(),
        "experiment_name": cfg.experiment_name,
        "config": asdict(cfg),
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
