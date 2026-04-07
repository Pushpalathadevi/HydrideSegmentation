#!/usr/bin/env bash
# Submit a single-GPU Slurm job while preserving the original submission repo root.

set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: ./submitJob_1GPU.sh <shell_file_to_run> [additional_arguments to shell_file_to_run]

Example:
  ./submitJob_1GPU.sh ./run_training_jobs.sh --dataset tiny --profile smoke
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

resolve_existing_file() {
  local path_in="$1"
  if [[ "$path_in" == /* ]]; then
    [[ -f "$path_in" ]] || return 1
    printf '%s\n' "$path_in"
    return 0
  fi
  [[ -f "$path_in" ]] || return 1
  (
    cd "$(dirname "$path_in")"
    printf '%s/%s\n' "$(pwd -P)" "$(basename "$path_in")"
  )
}

shell_file_input="$1"
shift

if ! shell_file="$(resolve_existing_file "$shell_file_input")"; then
  echo "ERROR: shell file not found: $shell_file_input" >&2
  exit 1
fi

submit_dir="$(pwd -P)"
export_args="ALL,HYDRIDE_REPO_ROOT=${submit_dir},HYDRIDE_SUBMIT_DIR=${submit_dir}"

cmd=(
  sbatch
  --time=14400
  --nodes=1
  --gpus=1
  --cpus-per-task=12
  --export="$export_args"
  "$shell_file"
  "$@"
)

printf 'Submitting command: '
printf '%q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
