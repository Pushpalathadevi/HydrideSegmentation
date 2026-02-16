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
    """Build subprocess command lines for infer/train/evaluate/package jobs."""

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
