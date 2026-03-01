"""Command builders and local orchestration helpers for pipeline jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


def _repo_root_from(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "scripts" / "microseg_cli.py").exists():
            return parent
    raise FileNotFoundError("could not locate repository root containing scripts/microseg_cli.py")


@dataclass(frozen=True)
class OrchestrationCommandBuilder:
    """Build subprocess command lines for infer/train/evaluate/dataops jobs."""

    repo_root: Path
    python_executable: str = sys.executable

    @classmethod
    def discover(cls, *, start: Path | None = None) -> OrchestrationCommandBuilder:
        base = start or Path(__file__)
        return cls(repo_root=_repo_root_from(base))

    @property
    def cli_script(self) -> Path:
        return self.repo_root / "scripts" / "microseg_cli.py"

    def _base(self) -> list[str]:
        return [self.python_executable, str(self.cli_script)]

    @staticmethod
    def _append_set(args: list[str], overrides: list[str] | None) -> None:
        for item in overrides or []:
            args.extend(["--set", item])

    def infer(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        image: str | None = None,
        model_name: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["infer"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if image:
            args.extend(["--image", image])
        if model_name:
            args.extend(["--model-name", model_name])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def train(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        dataset_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["train"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if dataset_dir:
            args.extend(["--dataset-dir", dataset_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def evaluate(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        dataset_dir: str | None = None,
        model_path: str | None = None,
        split: str | None = None,
        output_path: str | None = None,
    ) -> list[str]:
        args = self._base() + ["evaluate"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if dataset_dir:
            args.extend(["--dataset-dir", dataset_dir])
        if model_path:
            args.extend(["--model-path", model_path])
        if split:
            args.extend(["--split", split])
        if output_path:
            args.extend(["--output-path", output_path])
        return args

    def package(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        input_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["package"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if input_dir:
            args.extend(["--input-dir", input_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def dataset_prepare(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        dataset_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["dataset-prepare"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if dataset_dir:
            args.extend(["--dataset-dir", dataset_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def dataset_qa(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        dataset_dir: str | None = None,
        output_path: str | None = None,
        strict: bool = False,
    ) -> list[str]:
        args = self._base() + ["dataset-qa"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if dataset_dir:
            args.extend(["--dataset-dir", dataset_dir])
        if output_path:
            args.extend(["--output-path", output_path])
        if strict:
            args.append("--strict")
        return args

    def hpc_ga_generate(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        dataset_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["hpc-ga-generate"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if dataset_dir:
            args.extend(["--dataset-dir", dataset_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def hpc_ga_feedback_report(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        feedback_sources: str | None = None,
        output_path: str | None = None,
    ) -> list[str]:
        args = self._base() + ["hpc-ga-feedback-report"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if feedback_sources:
            args.extend(["--feedback-sources", feedback_sources])
        if output_path:
            args.extend(["--output-path", output_path])
        return args

    def preflight(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        mode: str | None = None,
        dataset_dir: str | None = None,
        model_path: str | None = None,
        benchmark_config: str | None = None,
    ) -> list[str]:
        args = self._base() + ["preflight"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if mode:
            args.extend(["--mode", mode])
        if dataset_dir:
            args.extend(["--dataset-dir", dataset_dir])
        if model_path:
            args.extend(["--model-path", model_path])
        if benchmark_config:
            args.extend(["--benchmark-config", benchmark_config])
        return args

    def deploy_package(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        model_path: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["deploy-package"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if model_path:
            args.extend(["--model-path", model_path])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def deploy_validate(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        package_dir: str | None = None,
        strict: bool = False,
    ) -> list[str]:
        args = self._base() + ["deploy-validate"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if package_dir:
            args.extend(["--package-dir", package_dir])
        if strict:
            args.append("--strict")
        return args

    def deploy_smoke(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        package_dir: str | None = None,
        image_path: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["deploy-smoke"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if package_dir:
            args.extend(["--package-dir", package_dir])
        if image_path:
            args.extend(["--image-path", image_path])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def deploy_health(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        package_dir: str | None = None,
        image_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["deploy-health"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if package_dir:
            args.extend(["--package-dir", package_dir])
        if image_dir:
            args.extend(["--image-dir", image_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def deploy_worker_run(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        package_dir: str | None = None,
        image_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["deploy-worker-run"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if package_dir:
            args.extend(["--package-dir", package_dir])
        if image_dir:
            args.extend(["--image-dir", image_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def deploy_canary_shadow(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        baseline_package_dir: str | None = None,
        candidate_package_dir: str | None = None,
        image_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["deploy-canary-shadow"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if baseline_package_dir:
            args.extend(["--baseline-package-dir", baseline_package_dir])
        if candidate_package_dir:
            args.extend(["--candidate-package-dir", candidate_package_dir])
        if image_dir:
            args.extend(["--image-dir", image_dir])
        return args

    def deploy_perf(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        package_dir: str | None = None,
        image_dir: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["deploy-perf"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if package_dir:
            args.extend(["--package-dir", package_dir])
        if image_dir:
            args.extend(["--image-dir", image_dir])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def promote_model(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        summary_json: str | None = None,
        model_name: str | None = None,
    ) -> list[str]:
        args = self._base() + ["promote-model"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if summary_json:
            args.extend(["--summary-json", summary_json])
        if model_name:
            args.extend(["--model-name", model_name])
        return args

    def support_bundle(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        run_root: str | None = None,
        output_dir: str | None = None,
    ) -> list[str]:
        args = self._base() + ["support-bundle"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if run_root:
            args.extend(["--run-root", run_root])
        if output_dir:
            args.extend(["--output-dir", output_dir])
        return args

    def compatibility_matrix(
        self,
        *,
        config: str | None = None,
        overrides: list[str] | None = None,
        output_path: str | None = None,
    ) -> list[str]:
        args = self._base() + ["compatibility-matrix"]
        if config:
            args.extend(["--config", config])
        self._append_set(args, overrides)
        if output_path:
            args.extend(["--output-path", output_path])
        return args
