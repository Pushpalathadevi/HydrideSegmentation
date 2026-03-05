#!/usr/bin/env bash
# Robust HPC runner for HydrideSegmentation benchmark workflow.
#
# Design goals:
# - Works even when Slurm spools/copies this script to another directory.
# - Enforces repo-root .venv Python for reproducible dependency resolution.
# - Uses absolute paths and hard-fails early with clear diagnostics.

set -Eeuo pipefail

# -----------------------
# Defaults
# -----------------------
CUDA_ID=""
DRYRUN="false"
STRICT="false"
PROFILE="smoke"             # smoke|dev|full
DATASET_CHOICE="tiny"       # tiny|full|custom
DATASET_DIR=""              # required for custom
SUITE_TEMPLATE="configs/hydride/benchmark_suite.top5.yml"
MODELS="unet_binary,hf_segformer_b0,hf_segformer_b2,transunet_tiny,segformer_mini"
SEEDS=""                    # if empty -> profile default
OUTPUT_ROOT=""              # default auto under ./outputs/benchmarks
LOG_ROOT="./logs/hydride_benchmarks"
KILL_PYTHON="false"         # default false on shared nodes
GPU_WATCH_SEC="60"
SKIP_DATASET_QA="false"
SKIP_REGISTRY_VALIDATION="false"
SKIP_TRAIN="false"
SKIP_EVAL="false"
SINGLE_SEED="false"
HF_OFFLINE="true"
HF_CACHE_ROOT=""
PYTHON_OVERRIDE=""

REPO_ROOT=""
SCRIPT_PATH=""
SCRIPT_DIR=""
VENV_DIR=""
PYTHON_EXE=""
STARTUP_LOG=""
SUITE_OUTER_LOG=""
MANIFEST_JSON=""
gpu_watch_pid=""

usage() {
  cat <<'USAGE'
Usage: bash ./run_training_jobs.sh [options]

Core options:
  --dataset tiny|full|custom            Which dataset root to use (default: tiny)
  --dataset_dir <path>                  Required if --dataset custom
  --profile smoke|dev|full              Controls seeds + train overrides (default: smoke)
  --models "a,b,c"                      Subset of models from the top5 suite (default: all top5)
  --seeds "42,43,44"                    Override seeds list (default depends on profile)
  --suite_template <path.yml>           Suite template YAML (default: configs/hydride/benchmark_suite.top5.yml)
  --output_root <path>                  Where suite outputs go (default: auto)
  --log_root <path>                     Outer log root (default: ./logs/hydride_benchmarks)

Runtime controls:
  --cuda_id "0,1"                       Sets CUDA_VISIBLE_DEVICES
  --gpu_watch_sec <sec>                 nvidia-smi polling interval seconds (default: 60)
  --kill_python true|false              Kill all python3 before run (default: false; risky on shared nodes)
  --python <path>                       Optional override; must match repo .venv python

Workflow toggles:
  --dryrun true|false                   Dry-run: sanity checks + plan commands only (default: false)
  --drey_run true|false                 Alias for --dryrun (typo-tolerant)
  --strict true|false                   Strict: fail if any run fails (default: false)
  --skip_dataset_qa true|false          Skip dataset QA checks (default: false)
  --skip_registry_validation true|false Skip registry validation (default: false)
  --skip_train true|false               Skip training stage (default: false)
  --skip_eval true|false                Skip evaluation stage (default: false)
  --single_seed true|false              Override to first configured seed only (default: false)

HF/Transformers offline policy:
  --hf_offline true|false               Set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE (default: true)
  --hf_cache_root <path>                Cache root (default: per-run log dir)

Environment hints for Slurm-spooled runs:
  HYDRIDE_REPO_ROOT  Preferred repo root (exported by submitJob_1GPU.sh)
  SLURM_SUBMIT_DIR   Slurm submit directory fallback

Examples:
  bash ./run_training_jobs.sh --dataset tiny --profile smoke --dryrun true
  bash ./run_training_jobs.sh --dataset full --profile full --cuda_id "0"
  bash ./run_training_jobs.sh --dataset custom --dataset_dir /data/HydrideData6.0/mado_style --profile dev
USAGE
}

# Allow quick help display without requiring repo root or .venv bootstrapping.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

log_ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

emit_log() {
  local level="$1"
  shift
  local msg="[$(log_ts)] [$level] $*"
  if [[ -n "$SUITE_OUTER_LOG" ]]; then
    echo "$msg" | tee -a "$SUITE_OUTER_LOG"
  elif [[ -n "$STARTUP_LOG" ]]; then
    echo "$msg" | tee -a "$STARTUP_LOG"
  else
    echo "$msg"
  fi
}

die() {
  emit_log "ERROR" "$*"
  exit 1
}

on_error() {
  local rc=$?
  local line_no="$1"
  local cmd="$2"
  emit_log "ERROR" "Command failed (rc=$rc) at line $line_no: $cmd"
  exit "$rc"
}

trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

on_exit() {
  if [[ -n "$gpu_watch_pid" ]]; then
    kill "$gpu_watch_pid" >/dev/null 2>&1 || true
    wait "$gpu_watch_pid" >/dev/null 2>&1 || true
    gpu_watch_pid=""
  fi
}

trap on_exit EXIT

have_cmd() { command -v "$1" >/dev/null 2>&1; }

parse_bool() {
  local value
  value="${1,,}"
  case "$value" in
    true|false) echo "$value" ;;
    *) die "Invalid boolean: $1 (expected true or false)" ;;
  esac
}

require_arg_value() {
  local flag="$1"
  local value="${2-}"
  [[ -n "$value" ]] || die "Missing value for $flag"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Required directory not found: $path"
}

resolve_script_path() {
  local src="$1"
  if have_cmd readlink; then
    local resolved
    resolved="$(readlink -f "$src" 2>/dev/null || true)"
    if [[ -n "$resolved" ]]; then
      printf '%s\n' "$resolved"
      return 0
    fi
  fi
  (
    cd "$(dirname "$src")"
    printf '%s/%s\n' "$(pwd -P)" "$(basename "$src")"
  )
}

canonicalize_dir() {
  local raw="$1"
  [[ -n "$raw" ]] || return 1
  [[ -d "$raw" ]] || return 1
  (
    cd "$raw"
    pwd -P
  )
}

is_repo_root() {
  local path="$1"
  [[ -f "$path/pyproject.toml" ]] && [[ -f "$path/scripts/hydride_benchmark_suite.py" ]] && [[ -f "$path/scripts/microseg_cli.py" ]]
}

resolve_repo_root() {
  local candidate=""

  for raw in "${HYDRIDE_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR" "$(pwd -P)"; do
    [[ -z "$raw" ]] && continue
    if candidate="$(canonicalize_dir "$raw" 2>/dev/null)" && is_repo_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    candidate="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
    if candidate="$(canonicalize_dir "$candidate" 2>/dev/null)" && is_repo_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  return 1
}

to_abs_path() {
  local raw="$1"
  if [[ "$raw" == /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s\n' "$REPO_ROOT/$raw"
  fi
}

normalize_csv() {
  local s="$1"
  s="${s// /}"
  s="${s//,,/,}"
  s="${s#,}"
  s="${s%,}"
  echo "$s"
}

as_python_list_of_ints() {
  local csv
  csv="$(normalize_csv "$1")"
  if [[ -z "$csv" ]]; then
    echo "[]"
    return
  fi
  local out="["
  local first="true"
  IFS=',' read -r -a arr <<<"$csv"
  for x in "${arr[@]}"; do
    [[ -z "$x" ]] && continue
    [[ "$x" =~ ^[0-9]+$ ]] || die "Invalid seed (not int): $x"
    if [[ "$first" == "true" ]]; then
      out+="$x"
      first="false"
    else
      out+=", $x"
    fi
  done
  out+="]"
  echo "$out"
}

dataset_guess_path() {
  local choice="$1"
  local p1="$REPO_ROOT/${choice}_data_HydrideData6.0/mado_style"
  local p2="$REPO_ROOT/test_data/${choice}_data_HydrideData6.0/mado_style"
  local p3="$REPO_ROOT/HydrideData6.0/mado_style"
  local p4="$REPO_ROOT/tiny_data_HydrideData6.0/mado_style"

  if [[ "$choice" == "tiny" ]]; then
    [[ -d "$p4" ]] && { echo "$p4"; return; }
    [[ -d "$p2" ]] && { echo "$p2"; return; }
  fi

  if [[ "$choice" == "full" ]]; then
    [[ -d "$p3" ]] && { echo "$p3"; return; }
    [[ -d "$p1" ]] && { echo "$p1"; return; }
  fi

  echo ""
}

validate_mado_style_layout() {
  local root="$1"
  local missing=0
  for split in train val test; do
    for sub in images masks; do
      if [[ ! -d "$root/$split/$sub" ]]; then
        emit_log "ERROR" "Missing dataset path: $root/$split/$sub"
        missing=1
      fi
    done
  done
  [[ "$missing" -eq 0 ]] || die "Dataset layout validation failed for: $root"
}

write_startup_log() {
  local log="$1"
  {
    echo "Run started (UTC): $(log_ts)"
    echo "Host: $(hostname)"
    echo "User: $(whoami)"
    echo "Original PWD: ${ORIGINAL_PWD}"
    echo "Effective PWD: $(pwd -P)"
    echo "Script path: $SCRIPT_PATH"
    echo "Script dir: $SCRIPT_DIR"
    echo "HYDRIDE_REPO_ROOT: ${HYDRIDE_REPO_ROOT:-<unset>}"
    echo "HYDRIDE_SUBMIT_DIR: ${HYDRIDE_SUBMIT_DIR:-<unset>}"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
    echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-<unset>}"
    echo "REPO_ROOT: $REPO_ROOT"
    echo "VENV_DIR: $VENV_DIR"
    echo "Python executable: $PYTHON_EXE"
    "$PYTHON_EXE" --version 2>&1 || true
    echo "HF_OFFLINE: $HF_OFFLINE"
    echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-<unset>}"
    echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-<unset>}"
    echo "HF_HOME: ${HF_HOME:-<unset>}"
    echo

    if have_cmd nvidia-smi; then
      echo "nvidia-smi:"
      nvidia-smi
    else
      echo "nvidia-smi: <not available>"
    fi
    echo

    if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
      echo "Git branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
      echo "Git commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
      echo "Git status (porcelain):"
      git -C "$REPO_ROOT" status --porcelain || true
    else
      echo "Git repo: <not detected>"
    fi
  } | tee -a "$log"
}

start_gpu_watch() {
  local log_file="$1"
  if have_cmd nvidia-smi; then
    (
      while true; do
        echo "---- GPU snapshot @ $(log_ts) ----"
        nvidia-smi
        sleep "$GPU_WATCH_SEC"
      done
    ) >>"$log_file" 2>&1 &
    gpu_watch_pid=$!
  fi
}

# -----------------------
# Bootstrap repo root + .venv
# -----------------------
ORIGINAL_PWD="$(pwd -P)"
SCRIPT_PATH="$(resolve_script_path "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"

if ! REPO_ROOT="$(resolve_repo_root)"; then
  echo "ERROR: unable to resolve repository root." >&2
  echo "DEBUG SCRIPT_PATH: $SCRIPT_PATH" >&2
  echo "DEBUG SCRIPT_DIR: $SCRIPT_DIR" >&2
  echo "DEBUG ORIGINAL_PWD: $ORIGINAL_PWD" >&2
  echo "DEBUG HYDRIDE_REPO_ROOT: ${HYDRIDE_REPO_ROOT:-<unset>}" >&2
  echo "DEBUG SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}" >&2
  exit 1
fi

cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"

require_file "$ACTIVATE_SCRIPT"
[[ -x "$PYTHON_EXE" ]] || die "Repo venv python is missing or not executable: $PYTHON_EXE"

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"
export PATH="$VENV_DIR/bin:$PATH"

MICROSEG_CMD=("$PYTHON_EXE" "$REPO_ROOT/scripts/microseg_cli.py")

# -----------------------
# Arg parsing
# -----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda_id) require_arg_value "$1" "${2-}"; CUDA_ID="$2"; shift 2;;
    --dryrun) require_arg_value "$1" "${2-}"; DRYRUN="$(parse_bool "$2")"; shift 2;;
    --drey_run) require_arg_value "$1" "${2-}"; DRYRUN="$(parse_bool "$2")"; shift 2;;
    --strict) require_arg_value "$1" "${2-}"; STRICT="$(parse_bool "$2")"; shift 2;;
    --profile) require_arg_value "$1" "${2-}"; PROFILE="$2"; shift 2;;
    --dataset) require_arg_value "$1" "${2-}"; DATASET_CHOICE="$2"; shift 2;;
    --dataset_dir) require_arg_value "$1" "${2-}"; DATASET_DIR="$2"; shift 2;;
    --suite_template) require_arg_value "$1" "${2-}"; SUITE_TEMPLATE="$2"; shift 2;;
    --models) require_arg_value "$1" "${2-}"; MODELS="$2"; shift 2;;
    --seeds) require_arg_value "$1" "${2-}"; SEEDS="$2"; shift 2;;
    --output_root) require_arg_value "$1" "${2-}"; OUTPUT_ROOT="$2"; shift 2;;
    --log_root) require_arg_value "$1" "${2-}"; LOG_ROOT="$2"; shift 2;;
    --python) require_arg_value "$1" "${2-}"; PYTHON_OVERRIDE="$2"; shift 2;;
    --kill_python) require_arg_value "$1" "${2-}"; KILL_PYTHON="$(parse_bool "$2")"; shift 2;;
    --gpu_watch_sec) require_arg_value "$1" "${2-}"; GPU_WATCH_SEC="$2"; shift 2;;
    --skip_dataset_qa) require_arg_value "$1" "${2-}"; SKIP_DATASET_QA="$(parse_bool "$2")"; shift 2;;
    --skip_registry_validation) require_arg_value "$1" "${2-}"; SKIP_REGISTRY_VALIDATION="$(parse_bool "$2")"; shift 2;;
    --skip_train) require_arg_value "$1" "${2-}"; SKIP_TRAIN="$(parse_bool "$2")"; shift 2;;
    --skip_eval) require_arg_value "$1" "${2-}"; SKIP_EVAL="$(parse_bool "$2")"; shift 2;;
    --single_seed|--single-seed) require_arg_value "$1" "${2-}"; SINGLE_SEED="$(parse_bool "$2")"; shift 2;;
    --hf_offline) require_arg_value "$1" "${2-}"; HF_OFFLINE="$(parse_bool "$2")"; shift 2;;
    --hf_cache_root) require_arg_value "$1" "${2-}"; HF_CACHE_ROOT="$2"; shift 2;;
    --help|-h) usage; exit 0;;
    *) die "Unknown argument: $1" ;;
  esac
done

if [[ -n "$PYTHON_OVERRIDE" ]]; then
  OVERRIDE_ABS="$(to_abs_path "$PYTHON_OVERRIDE")"
  if [[ "$OVERRIDE_ABS" != "$PYTHON_EXE" ]]; then
    die "--python override is not allowed unless it matches repo venv python: $PYTHON_EXE"
  fi
fi

if [[ -n "$CUDA_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_ID"
fi

[[ "$GPU_WATCH_SEC" =~ ^[0-9]+$ ]] || die "--gpu_watch_sec must be an integer (got: $GPU_WATCH_SEC)"

case "$PROFILE" in
  smoke|dev|full) ;;
  *) die "Invalid --profile: $PROFILE (expected smoke|dev|full)" ;;
esac

case "$DATASET_CHOICE" in
  tiny|full|custom) ;;
  *) die "Invalid --dataset: $DATASET_CHOICE (expected tiny|full|custom)" ;;
esac

SUITE_TEMPLATE_PATH="$(to_abs_path "$SUITE_TEMPLATE")"
require_file "$SUITE_TEMPLATE_PATH"
require_file "$REPO_ROOT/scripts/hydride_benchmark_suite.py"

# -----------------------
# Resolve dataset dir
# -----------------------
if [[ "$DATASET_CHOICE" == "custom" ]]; then
  [[ -n "$DATASET_DIR" ]] || die "--dataset_dir is required when --dataset custom"
  DATASET_DIR="$(to_abs_path "$DATASET_DIR")"
else
  guess="$(dataset_guess_path "$DATASET_CHOICE")"
  if [[ -z "$guess" ]]; then
    die "Could not auto-find dataset for '$DATASET_CHOICE'. Use --dataset custom --dataset_dir /path/to/HydrideData6.0/mado_style"
  fi
  DATASET_DIR="$guess"
fi

require_dir "$DATASET_DIR"
validate_mado_style_layout "$DATASET_DIR"

# -----------------------
# HF offline policy
# -----------------------
if [[ "$HF_OFFLINE" == "true" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
SLURM_TAG="${SLURM_JOB_ID:-noslurm}"
DATASET_TAG="${DATASET_CHOICE}_${PROFILE}"

LOG_ROOT="$(to_abs_path "$LOG_ROOT")"
mkdir -p "$LOG_ROOT"
JOB_DIR="$LOG_ROOT/${RUN_TS}_${DATASET_TAG}_job${SLURM_TAG}"
mkdir -p "$JOB_DIR"

if [[ -z "$HF_CACHE_ROOT" ]]; then
  HF_CACHE_ROOT="$JOB_DIR/hf_cache"
else
  HF_CACHE_ROOT="$(to_abs_path "$HF_CACHE_ROOT")"
fi
mkdir -p "$HF_CACHE_ROOT"
export HF_HOME="$HF_CACHE_ROOT"

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$REPO_ROOT/outputs/benchmarks/${RUN_TS}_${DATASET_TAG}_job${SLURM_TAG}"
else
  OUTPUT_ROOT="$(to_abs_path "$OUTPUT_ROOT")"
fi
mkdir -p "$OUTPUT_ROOT"

STARTUP_LOG="$JOB_DIR/startup.log"
SUITE_OUTER_LOG="$JOB_DIR/suite_outer.log"
MANIFEST_JSON="$JOB_DIR/run_manifest.json"
PY_ENV_LOG="$JOB_DIR/python_env_sanity.log"

write_startup_log "$STARTUP_LOG"

# Ensure we are truly in the repo venv and have required dependency versions.
set +e
"$PYTHON_EXE" - <<'PY' >"$PY_ENV_LOG" 2>&1
import json
import sys
import pydantic

payload = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "pydantic_version": pydantic.__version__,
}
print(json.dumps(payload, indent=2))

major = int(str(pydantic.__version__).split(".")[0])
if major < 2:
    raise SystemExit(f"pydantic>=2 is required, found {pydantic.__version__}")
PY
py_env_rc=$?
set -e
cat "$PY_ENV_LOG" | tee -a "$STARTUP_LOG" >/dev/null
if [[ "$py_env_rc" -ne 0 ]]; then
  die "Python environment sanity check failed. See $PY_ENV_LOG"
fi

if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$REPO_ROOT" rev-parse HEAD >"$JOB_DIR/git_commit.txt" 2>/dev/null || true
  git -C "$REPO_ROOT" status --porcelain >"$JOB_DIR/git_status_porcelain.txt" 2>/dev/null || true
  git -C "$REPO_ROOT" diff >"$JOB_DIR/git_diff.patch" 2>/dev/null || true
fi

"$PYTHON_EXE" -m pip freeze >"$JOB_DIR/pip_freeze.txt" 2>&1 || true

# -----------------------
# Preflight: registry validation + dataset QA (optional)
# -----------------------
if [[ "$SKIP_REGISTRY_VALIDATION" == "false" ]]; then
  emit_log "INFO" "Running registry validation..."
  set +e
  "${MICROSEG_CMD[@]}" validate-registry --config "$REPO_ROOT/configs/registry_validation.default.yml" --strict 2>&1 | tee -a "$SUITE_OUTER_LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    emit_log "ERROR" "Registry validation failed (rc=$rc)."
    [[ "$STRICT" == "true" ]] && exit "$rc"
  fi
fi

if [[ "$SKIP_DATASET_QA" == "false" ]]; then
  emit_log "INFO" "Running dataset QA (strict)..."
  set +e
  "${MICROSEG_CMD[@]}" dataset-qa --config "$REPO_ROOT/configs/dataset_qa.default.yml" --dataset-dir "$DATASET_DIR" --output-path "$JOB_DIR/dataset_qa.json" --strict 2>&1 | tee -a "$SUITE_OUTER_LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    emit_log "ERROR" "Dataset QA failed (rc=$rc). See: $JOB_DIR/dataset_qa.json"
    [[ "$STRICT" == "true" ]] && exit "$rc"
  fi
fi

# -----------------------
# Profile -> seed list + train overrides
# -----------------------
if [[ -z "$SEEDS" ]]; then
  case "$PROFILE" in
    smoke) SEEDS="42" ;;
    dev)   SEEDS="42" ;;
    full)  SEEDS="42,43,44" ;;
  esac
fi

SEEDS_PY="$(as_python_list_of_ints "$SEEDS")"
MODELS="$(normalize_csv "$MODELS")"

RESOLVED_SUITE_YML="$JOB_DIR/benchmark_suite.resolved.yml"

export RUNNER_REPO_ROOT="$REPO_ROOT"
export RUNNER_SUITE_TEMPLATE_PATH="$SUITE_TEMPLATE_PATH"
export RUNNER_DATASET_DIR="$DATASET_DIR"
export RUNNER_OUTPUT_ROOT="$OUTPUT_ROOT"
export RUNNER_MODELS="$MODELS"
export RUNNER_PROFILE="$PROFILE"
export RUNNER_SEEDS_PY="$SEEDS_PY"
export RUNNER_PYTHON_EXE="$PYTHON_EXE"
export RUNNER_RESOLVED_SUITE_YML="$RESOLVED_SUITE_YML"

"$PYTHON_EXE" - <<'PY'
import ast
import os
from pathlib import Path

import yaml

repo = Path(os.environ["RUNNER_REPO_ROOT"])
template_path = Path(os.environ["RUNNER_SUITE_TEMPLATE_PATH"])
cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit(f"Suite template is not a mapping: {template_path}")

dataset_dir = os.environ["RUNNER_DATASET_DIR"]
output_root = os.environ["RUNNER_OUTPUT_ROOT"]
models_csv = os.environ["RUNNER_MODELS"]
models = [m.strip() for m in models_csv.split(",") if m.strip()]
profile = os.environ["RUNNER_PROFILE"]
seeds = ast.literal_eval(os.environ["RUNNER_SEEDS_PY"])
if not isinstance(seeds, list):
    raise SystemExit("Parsed seeds are invalid")

if profile == "smoke":
    overrides = {
        "unet_binary": ["epochs=2", "batch_size=2"],
        "hf_segformer_b0": ["epochs=2", "batch_size=1"],
        "hf_segformer_b2": ["epochs=2", "batch_size=1"],
        "hf_segformer_b5": ["epochs=2", "batch_size=1"],
        "transunet_tiny": ["epochs=2", "batch_size=1"],
        "transunet_tiny_deep": ["epochs=2", "batch_size=1"],
        "segformer_mini": ["epochs=2", "batch_size=2"],
        "segformer_mini_wide": ["epochs=2", "batch_size=2"],
    }
elif profile == "dev":
    overrides = {
        "unet_binary": ["epochs=10", "batch_size=4"],
        "hf_segformer_b0": ["epochs=10", "batch_size=1"],
        "hf_segformer_b2": ["epochs=10", "batch_size=1"],
        "hf_segformer_b5": ["epochs=10", "batch_size=1"],
        "transunet_tiny": ["epochs=10", "batch_size=1"],
        "transunet_tiny_deep": ["epochs=10", "batch_size=1"],
        "segformer_mini": ["epochs=10", "batch_size=2"],
        "segformer_mini_wide": ["epochs=10", "batch_size=2"],
    }
else:
    overrides = {}

cfg["dataset_dir"] = dataset_dir
cfg["output_root"] = output_root
cfg["python_executable"] = str(Path(os.environ["RUNNER_PYTHON_EXE"]))
cfg["seeds"] = [int(s) for s in seeds]

exps = cfg.get("experiments", [])
if not isinstance(exps, list):
    raise SystemExit("suite config experiments must be a list")

filtered = []
for exp in exps:
    if not isinstance(exp, dict):
        continue
    name = str(exp.get("name", "")).strip()
    if models and name not in models:
        continue
    if name in overrides and overrides[name]:
        exp = dict(exp)
        exp["train_overrides"] = list(overrides[name])
    filtered.append(exp)

if not filtered:
    raise SystemExit(f"No experiments left after filtering with models={models!r}")

cfg["experiments"] = filtered

out_path = Path(os.environ["RUNNER_RESOLVED_SUITE_YML"])
out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"Wrote resolved suite config: {out_path}")
PY

unset RUNNER_REPO_ROOT RUNNER_SUITE_TEMPLATE_PATH RUNNER_DATASET_DIR RUNNER_OUTPUT_ROOT
unset RUNNER_MODELS RUNNER_PROFILE RUNNER_SEEDS_PY RUNNER_PYTHON_EXE RUNNER_RESOLVED_SUITE_YML

export RUNNER_MANIFEST_PATH="$MANIFEST_JSON"
export RUNNER_MANIFEST_REPO_ROOT="$REPO_ROOT"
export RUNNER_MANIFEST_DATASET_DIR="$DATASET_DIR"
export RUNNER_MANIFEST_SUITE_TEMPLATE="$SUITE_TEMPLATE_PATH"
export RUNNER_MANIFEST_SUITE_RESOLVED="$RESOLVED_SUITE_YML"
export RUNNER_MANIFEST_OUTPUT_ROOT="$OUTPUT_ROOT"
export RUNNER_MANIFEST_LOG_ROOT="$LOG_ROOT"
export RUNNER_MANIFEST_JOB_DIR="$JOB_DIR"
export RUNNER_MANIFEST_PROFILE="$PROFILE"
export RUNNER_MANIFEST_MODELS="$MODELS"
export RUNNER_MANIFEST_SEEDS="$SEEDS"
export RUNNER_MANIFEST_DRYRUN="$DRYRUN"
export RUNNER_MANIFEST_STRICT="$STRICT"
export RUNNER_MANIFEST_SKIP_DATASET_QA="$SKIP_DATASET_QA"
export RUNNER_MANIFEST_SKIP_REGISTRY_VALIDATION="$SKIP_REGISTRY_VALIDATION"
export RUNNER_MANIFEST_SKIP_TRAIN="$SKIP_TRAIN"
export RUNNER_MANIFEST_SKIP_EVAL="$SKIP_EVAL"
export RUNNER_MANIFEST_SINGLE_SEED="$SINGLE_SEED"
export RUNNER_MANIFEST_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export RUNNER_MANIFEST_HF_OFFLINE="$HF_OFFLINE"
export RUNNER_MANIFEST_HF_HOME="${HF_HOME:-}"

"$PYTHON_EXE" - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "schema_version": "microseg.hpc_runner_manifest.v2",
    "repo_root": os.environ["RUNNER_MANIFEST_REPO_ROOT"],
    "dataset_dir": os.environ["RUNNER_MANIFEST_DATASET_DIR"],
    "suite_template": os.environ["RUNNER_MANIFEST_SUITE_TEMPLATE"],
    "suite_resolved": os.environ["RUNNER_MANIFEST_SUITE_RESOLVED"],
    "output_root": os.environ["RUNNER_MANIFEST_OUTPUT_ROOT"],
    "log_root": os.environ["RUNNER_MANIFEST_LOG_ROOT"],
    "job_dir": os.environ["RUNNER_MANIFEST_JOB_DIR"],
    "profile": os.environ["RUNNER_MANIFEST_PROFILE"],
    "models": os.environ["RUNNER_MANIFEST_MODELS"],
    "seeds": os.environ["RUNNER_MANIFEST_SEEDS"],
    "dry_run": os.environ["RUNNER_MANIFEST_DRYRUN"],
    "strict": os.environ["RUNNER_MANIFEST_STRICT"],
    "skip_dataset_qa": os.environ["RUNNER_MANIFEST_SKIP_DATASET_QA"],
    "skip_registry_validation": os.environ["RUNNER_MANIFEST_SKIP_REGISTRY_VALIDATION"],
    "skip_train": os.environ["RUNNER_MANIFEST_SKIP_TRAIN"],
    "skip_eval": os.environ["RUNNER_MANIFEST_SKIP_EVAL"],
    "single_seed": os.environ["RUNNER_MANIFEST_SINGLE_SEED"],
    "cuda_visible_devices": os.environ["RUNNER_MANIFEST_CUDA_VISIBLE_DEVICES"],
    "hf_offline": os.environ["RUNNER_MANIFEST_HF_OFFLINE"],
    "hf_home": os.environ["RUNNER_MANIFEST_HF_HOME"],
    "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR", ""),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "hydride_repo_root_env": os.environ.get("HYDRIDE_REPO_ROOT", ""),
    "hydride_submit_dir_env": os.environ.get("HYDRIDE_SUBMIT_DIR", ""),
}

out = Path(os.environ["RUNNER_MANIFEST_PATH"])
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote manifest: {out}")
PY

unset RUNNER_MANIFEST_PATH RUNNER_MANIFEST_REPO_ROOT RUNNER_MANIFEST_DATASET_DIR
unset RUNNER_MANIFEST_SUITE_TEMPLATE RUNNER_MANIFEST_SUITE_RESOLVED RUNNER_MANIFEST_OUTPUT_ROOT
unset RUNNER_MANIFEST_LOG_ROOT RUNNER_MANIFEST_JOB_DIR RUNNER_MANIFEST_PROFILE RUNNER_MANIFEST_MODELS
unset RUNNER_MANIFEST_SEEDS RUNNER_MANIFEST_DRYRUN RUNNER_MANIFEST_STRICT
unset RUNNER_MANIFEST_SKIP_DATASET_QA RUNNER_MANIFEST_SKIP_REGISTRY_VALIDATION
unset RUNNER_MANIFEST_SKIP_TRAIN RUNNER_MANIFEST_SKIP_EVAL RUNNER_MANIFEST_SINGLE_SEED RUNNER_MANIFEST_CUDA_VISIBLE_DEVICES
unset RUNNER_MANIFEST_HF_OFFLINE RUNNER_MANIFEST_HF_HOME

emit_log "INFO" "Resolved suite YAML: $RESOLVED_SUITE_YML"
emit_log "INFO" "Suite outputs root: $OUTPUT_ROOT"

CMD=("$PYTHON_EXE" "$REPO_ROOT/scripts/hydride_benchmark_suite.py" "--config" "$RESOLVED_SUITE_YML")
[[ "$DRYRUN" == "true" ]] && CMD+=("--dry-run")
[[ "$STRICT" == "true" ]] && CMD+=("--strict")
[[ "$SKIP_TRAIN" == "true" ]] && CMD+=("--skip-train")
[[ "$SKIP_EVAL" == "true" ]] && CMD+=("--skip-eval")
[[ "$SINGLE_SEED" == "true" ]] && CMD+=("--single-seed")

emit_log "INFO" "Command: ${CMD[*]}"

if [[ "$DRYRUN" == "true" ]]; then
  emit_log "INFO" "DRYRUN=true -> planning only. No training/eval will execute."
fi

if [[ "$KILL_PYTHON" == "true" && "$DRYRUN" == "false" ]]; then
  emit_log "INFO" "KILL_PYTHON=true -> killing all python3 processes (risky on shared nodes)."
  killall -9 python3 >/dev/null 2>&1 || true
fi

start_gpu_watch "$SUITE_OUTER_LOG"

set +e
"${CMD[@]}" 2>&1 | tee -a "$SUITE_OUTER_LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ -n "$gpu_watch_pid" ]]; then
  kill "$gpu_watch_pid" >/dev/null 2>&1 || true
  wait "$gpu_watch_pid" >/dev/null 2>&1 || true
  gpu_watch_pid=""
fi

emit_log "INFO" "Suite runner exit code: $rc"
emit_log "INFO" "Dashboard (if training/eval ran): $OUTPUT_ROOT/benchmark_dashboard.html"
emit_log "INFO" "Summary JSON: $OUTPUT_ROOT/benchmark_summary.json"
emit_log "INFO" "Aggregate CSV: $OUTPUT_ROOT/benchmark_aggregate.csv"

if [[ "$STRICT" == "true" && "$rc" -ne 0 ]]; then
  exit "$rc"
fi

exit 0
