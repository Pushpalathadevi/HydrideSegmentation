"""Data contracts for per-inference feedback capture and active-learning loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FeedbackRating = Literal["unrated", "thumbs_up", "thumbs_down"]

FEEDBACK_RECORD_SCHEMA = "microseg.feedback_record.v1"
FEEDBACK_BUNDLE_SCHEMA = "microseg.feedback_bundle.v1"
FEEDBACK_BUNDLE_RESULT_SCHEMA = "microseg.feedback_bundle_result.v1"
FEEDBACK_INGEST_REPORT_SCHEMA = "microseg.feedback_ingest_report.v1"
FEEDBACK_DATASET_MANIFEST_SCHEMA = "microseg.feedback_dataset_manifest.v1"
FEEDBACK_TRIGGER_REPORT_SCHEMA = "microseg.feedback_train_trigger_report.v1"


@dataclass(frozen=True)
class FeedbackCaptureConfig:
    """Configuration for writing per-inference feedback records."""

    feedback_root: str = "outputs/feedback_records"
    deployment_id: str = "local_desktop"
    operator_id: str = "unknown_operator"
    source: str = "desktop_gui"


@dataclass(frozen=True)
class FeedbackRecord:
    """Canonical per-inference feedback payload."""

    schema_version: str
    record_id: str
    run_id: str
    created_utc: str
    updated_utc: str
    deployment_id: str
    operator_id: str
    source: str
    source_image_path: str
    source_image_sha256: str
    model_id: str
    model_name: str
    model_artifact_hint: str
    started_utc: str
    finished_utc: str
    runtime: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    resolved_config_sha256: str = ""
    inference_manifest: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    correction: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_manifest_sha256: str = ""


@dataclass(frozen=True)
class FeedbackCaptureResult:
    """Filesystem paths returned when a feedback record is created."""

    record_id: str
    record_dir: str
    record_path: str
    events_path: str
    artifacts_manifest_path: str


@dataclass(frozen=True)
class FeedbackBundleConfig:
    """Configuration for bundling locally captured feedback records."""

    feedback_root: str
    output_dir: str
    deployment_id: str
    max_records: int = 200
    state_path: str = ""
    cadence_days: int = 7
    cadence_record_count: int = 200


@dataclass(frozen=True)
class FeedbackBundleManifest:
    """Bundle manifest contract for transfer between deployments and central ingest."""

    schema_version: str
    bundle_id: str
    created_utc: str
    deployment_id: str
    source_feedback_root: str
    record_count: int
    records: list[dict[str, Any]] = field(default_factory=list)
    cadence_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackBundleResult:
    """Result summary for local feedback bundle export."""

    schema_version: str
    created_utc: str
    deployment_id: str
    bundle_id: str
    bundle_dir: str
    bundle_zip_path: str
    manifest_path: str
    selected_records: int
    pending_records: int
    state_path: str


@dataclass(frozen=True)
class FeedbackIngestConfig:
    """Configuration for central ingest of deployment feedback bundles."""

    bundle_paths: tuple[str, ...]
    ingest_root: str
    output_path: str
    dedup_index_path: str
    review_queue_path: str


@dataclass(frozen=True)
class FeedbackIngestReport:
    """Central ingest report for bundle processing."""

    schema_version: str
    created_utc: str
    ingest_root: str
    bundles_processed: int
    accepted_records: int
    duplicate_records: int
    rejected_records: int
    review_queue_records: int
    accepted: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FeedbackDatasetBuildConfig:
    """Configuration for building training datasets from ingested feedback records."""

    feedback_root: str
    output_dir: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    thumbs_up_weight: float = 0.2
    corrected_weight: float = 1.0
    leakage_group: Literal["source_stem", "record_id"] = "source_stem"


@dataclass(frozen=True)
class FeedbackDatasetBuildResult:
    """Result summary for feedback-derived dataset generation."""

    schema_version: str
    created_utc: str
    output_dir: str
    manifest_path: str
    total_records_scanned: int
    included_samples: int
    corrected_samples: int
    pseudo_labeled_samples: int
    excluded_unrated: int
    excluded_downvote_without_correction: int
    split_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackTrainTriggerConfig:
    """Configuration for threshold-based retrain trigger evaluation."""

    feedback_root: str
    output_path: str
    state_path: str
    corrected_threshold: int = 500
    max_days_since_last_trigger: int = 14
    train_config: str = "configs/train.default.yml"
    evaluate_config: str = "configs/evaluate.default.yml"
    dataset_output_dir: str = "outputs/feedback_training_dataset"
    train_output_dir: str = "outputs/training_feedback_cycle"
    evaluate_output_path: str = "outputs/evaluation/feedback_cycle_eval.json"
    execute: bool = False
    train_overrides: tuple[str, ...] = ()
    evaluate_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeedbackTrainTriggerReport:
    """Report produced by threshold-based retrain trigger evaluation."""

    schema_version: str
    created_utc: str
    should_trigger: bool
    trigger_reason: str
    corrected_records_since_last_trigger: int
    days_since_last_trigger: float
    state_path: str
    report_path: str
    dataset_manifest_path: str = ""
    train_exit_code: int | None = None
    evaluate_exit_code: int | None = None
    commands: list[list[str]] = field(default_factory=list)
