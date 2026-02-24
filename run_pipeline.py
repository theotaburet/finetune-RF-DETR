#!/usr/bin/env python3
"""Unified RF-DETR Pipeline for Audio Event Detection.

Complete pipeline with toggleable steps:
  1. Preprocess: Audio files → Spectrogram chunks (COCO format)
  2. Split: Split dataset into train/val/test
  3. Train: Fine-tune RF-DETR on spectrograms
  4. Evaluate: Run inference on test set
  5. Optimize merging: Grid-search merge parameters on test audio
  6. Infer: Run inference on new audio files

Usage:
    # Run full pipeline
    python run_pipeline.py --config config/pipeline.yaml --all

    # Preprocess and train only
    python run_pipeline.py --config config/pipeline.yaml --preprocess --train

    # Evaluate existing model
    python run_pipeline.py --config config/pipeline.yaml --evaluate

    # Optimize merge parameters after training
    python run_pipeline.py --config config/pipeline.yaml --optimize-merging

    # Generate config template
    python run_pipeline.py --generate-config config/my_pipeline.yaml

"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PreprocessConfig:
    """Preprocessing step configuration."""

    enabled: bool = True
    audio_dir: str = "data/audio"
    metadata_dir: str = "data/metadata"
    output_dir: str = "data/processed"
    chunking_config: str = "config/chunking.yaml"
    extensions: list[str] = field(default_factory=lambda: [".flac", ".wav", ".mp3"])
    recursive: bool = True  # Search subdirectories for audio files
    debug_visualize: bool = False
    max_files: int | None = None


@dataclass
class SplitConfig:
    """Dataset splitting configuration."""

    enabled: bool = True
    input_dir: str = "data/processed"
    output_dir: str = "data/split_dataset"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42
    stratify: bool = True

    def __post_init__(self) -> None:
        """Validate split ratios."""
        for name, val in [
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


@dataclass
class TrainConfig:
    """Training step configuration."""

    enabled: bool = True
    dataset_dir: str = "data/split_dataset"
    output_dir: str = "output"
    model_size: str = "base"
    pretrained_weights: str | None = None
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    resume_from: str | None = None
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 5


@dataclass
class EvaluateConfig:
    """Evaluation step configuration."""

    enabled: bool = False
    weights: str = "output/checkpoint_best.pth"
    test_dir: str = "data/split_dataset/test"
    output_dir: str = "output/eval"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    save_visualizations: bool = True


@dataclass
class OptimizeMergingConfig:
    """Merge parameter optimization step configuration."""

    enabled: bool = False
    weights: str = "output/checkpoint_best.pth"
    metadata: str = "data/downloaded/metadata/download_metadata.json"
    audio_dir: str | None = None
    chunking_config: str = "config/chunking.yaml"
    output: str = "config/merging.yaml"
    results_json: str | None = None
    confidence: float = 0.3
    iou_threshold: float = 0.3
    per_class: bool = False
    max_files: int | None = None
    delta_time_values: list[float] = field(default_factory=lambda: [200, 500, 1000, 2000, 4000, 8000])
    delta_freq_values: list[float] = field(default_factory=lambda: [50, 100, 200, 500])
    score_threshold_values: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    min_duration_values: list[float] = field(default_factory=lambda: [0, 100, 200])


@dataclass
class InferConfig:
    """Inference step configuration."""

    enabled: bool = False
    weights: str = "output/checkpoint_best.pth"
    audio_dir: str = "data/inference"
    output_dir: str = "output/predictions"
    chunking_config: str = "config/chunking.yaml"
    extensions: list[str] = field(default_factory=lambda: [".flac", ".wav", ".mp3"])
    recursive: bool = True  # Search subdirectories for audio files
    confidence_threshold: float = 0.5
    merge_gap_ms: float = 100.0
    output_format: str = "json"  # json, csv, raven


@dataclass
class DownloadStepConfig:
    """Download step configuration for fetching data from EKB API."""

    enabled: bool = False
    config_file: str | None = None  # Path to download.yaml config
    api_url: str = ""
    api_token: str = ""
    output_dir: str = "data/downloaded"
    # Test mode: limit downloads to this many labels (None = no limit)
    max_labels: int | None = None
    # Force re-download even if data appears unchanged
    force: bool = False
    # Filters
    sources: list[str] | None = None
    label_hierarchy: str | None = None


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    name: str = "rf_detr_pipeline"
    description: str = "RF-DETR Audio Event Detection Pipeline"

    # Class configuration
    class_names: list[str] = field(default_factory=list)
    classes_file: str | None = None

    # Step configurations
    download: DownloadStepConfig = field(default_factory=DownloadStepConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    optimize_merging: OptimizeMergingConfig = field(default_factory=OptimizeMergingConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        # Top-level fields
        config.name = data.get("name", config.name)
        config.description = data.get("description", config.description)
        config.class_names = data.get("class_names", config.class_names)
        config.classes_file = data.get("classes_file", config.classes_file)

        # Step configs
        if "download" in data:
            config.download = DownloadStepConfig(**data["download"])
        if "preprocess" in data:
            config.preprocess = PreprocessConfig(**data["preprocess"])
        if "split" in data:
            config.split = SplitConfig(**data["split"])
        if "train" in data:
            config.train = TrainConfig(**data["train"])
        if "evaluate" in data:
            config.evaluate = EvaluateConfig(**data["evaluate"])
        if "optimize_merging" in data:
            config.optimize_merging = OptimizeMergingConfig(**data["optimize_merging"])
        if "infer" in data:
            config.infer = InferConfig(**data["infer"])

        return config

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "name": self.name,
            "description": self.description,
            "class_names": self.class_names,
            "classes_file": self.classes_file,
            "download": {
                "enabled": self.download.enabled,
                "config_file": self.download.config_file,
                "api_url": self.download.api_url,
                "api_token": self.download.api_token,
                "output_dir": self.download.output_dir,
                "max_labels": self.download.max_labels,
                "force": self.download.force,
                "sources": self.download.sources,
                "label_hierarchy": self.download.label_hierarchy,
            },
            "preprocess": {
                "enabled": self.preprocess.enabled,
                "audio_dir": self.preprocess.audio_dir,
                "metadata_dir": self.preprocess.metadata_dir,
                "output_dir": self.preprocess.output_dir,
                "chunking_config": self.preprocess.chunking_config,
                "extensions": self.preprocess.extensions,
                "recursive": self.preprocess.recursive,
                "debug_visualize": self.preprocess.debug_visualize,
                "max_files": self.preprocess.max_files,
            },
            "split": {
                "enabled": self.split.enabled,
                "input_dir": self.split.input_dir,
                "output_dir": self.split.output_dir,
                "train_ratio": self.split.train_ratio,
                "val_ratio": self.split.val_ratio,
                "test_ratio": self.split.test_ratio,
                "seed": self.split.seed,
                "stratify": self.split.stratify,
            },
            "train": {
                "enabled": self.train.enabled,
                "dataset_dir": self.train.dataset_dir,
                "output_dir": self.train.output_dir,
                "model_size": self.train.model_size,
                "pretrained_weights": self.train.pretrained_weights,
                "epochs": self.train.epochs,
                "batch_size": self.train.batch_size,
                "learning_rate": self.train.learning_rate,
                "num_workers": self.train.num_workers,
                "device": self.train.device,
                "resume_from": self.train.resume_from,
                "early_stopping_patience": self.train.early_stopping_patience,
                "save_every_n_epochs": self.train.save_every_n_epochs,
            },
            "evaluate": {
                "enabled": self.evaluate.enabled,
                "weights": self.evaluate.weights,
                "test_dir": self.evaluate.test_dir,
                "output_dir": self.evaluate.output_dir,
                "confidence_threshold": self.evaluate.confidence_threshold,
                "iou_threshold": self.evaluate.iou_threshold,
                "save_visualizations": self.evaluate.save_visualizations,
            },
            "optimize_merging": {
                "enabled": self.optimize_merging.enabled,
                "weights": self.optimize_merging.weights,
                "metadata": self.optimize_merging.metadata,
                "audio_dir": self.optimize_merging.audio_dir,
                "chunking_config": self.optimize_merging.chunking_config,
                "output": self.optimize_merging.output,
                "results_json": self.optimize_merging.results_json,
                "confidence": self.optimize_merging.confidence,
                "iou_threshold": self.optimize_merging.iou_threshold,
                "per_class": self.optimize_merging.per_class,
                "max_files": self.optimize_merging.max_files,
                "delta_time_values": self.optimize_merging.delta_time_values,
                "delta_freq_values": self.optimize_merging.delta_freq_values,
                "score_threshold_values": self.optimize_merging.score_threshold_values,
                "min_duration_values": self.optimize_merging.min_duration_values,
            },
            "infer": {
                "enabled": self.infer.enabled,
                "weights": self.infer.weights,
                "audio_dir": self.infer.audio_dir,
                "output_dir": self.infer.output_dir,
                "chunking_config": self.infer.chunking_config,
                "extensions": self.infer.extensions,
                "recursive": self.infer.recursive,
                "confidence_threshold": self.infer.confidence_threshold,
                "merge_gap_ms": self.infer.merge_gap_ms,
                "output_format": self.infer.output_format,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Progress Bar Utilities
# =============================================================================


def create_step_progress() -> Progress:
    """Create a progress bar for step-level tasks with time tracking."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_file_progress() -> Progress:
    """Create a progress bar for file-level processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("→"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


class PipelineProgressManager:
    """Manages overall pipeline progress display."""

    def __init__(self, steps: list[tuple[str, type]]) -> None:
        """Initialize the progress manager.

        Args:
            steps: List of (step_name, step_class) tuples.

        """
        self.steps = steps
        self.step_status: dict[str, str] = {name: "pending" for name, _ in steps}
        self.current_step: str | None = None
        self.step_start_times: dict[str, datetime] = {}
        self.step_end_times: dict[str, datetime] = {}

    def start_step(self, step_name: str) -> None:
        """Mark a step as started."""
        self.current_step = step_name
        self.step_status[step_name] = "running"
        self.step_start_times[step_name] = datetime.now()

    def complete_step(self, step_name: str, success: bool = True) -> None:
        """Mark a step as completed."""
        self.step_status[step_name] = "done" if success else "failed"
        self.step_end_times[step_name] = datetime.now()
        if self.current_step == step_name:
            self.current_step = None

    def get_step_duration(self, step_name: str) -> str:
        """Get formatted duration for a step."""
        if step_name not in self.step_start_times:
            return "-"
        start = self.step_start_times[step_name]
        end = self.step_end_times.get(step_name, datetime.now())
        delta = end - start
        total_seconds = max(0, int(delta.total_seconds()))  # Ensure non-negative
        if total_seconds < 60:
            return f"{total_seconds}s" if total_seconds > 0 else "<1s"
        minutes, seconds = divmod(total_seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"

    def render_status_table(self) -> Table:
        """Render the pipeline status as a table."""
        table = Table(
            title="Pipeline Progress",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Step", style="cyan", width=12)
        table.add_column("Status", width=10)
        table.add_column("Duration", width=12, justify="right")

        status_icons = {
            "pending": "[dim]○ Pending[/dim]",
            "running": "[bold yellow]● Running[/bold yellow]",
            "done": "[bold green]✓ Done[/bold green]",
            "failed": "[bold red]✗ Failed[/bold red]",
            "skipped": "[dim]⊘ Skipped[/dim]",
        }

        for step_name, _ in self.steps:
            status = self.step_status.get(step_name, "pending")
            duration = self.get_step_duration(step_name) if status in ("done", "running", "failed") else "-"
            table.add_row(
                step_name,
                status_icons.get(status, status),
                duration,
            )

        return table


# =============================================================================
# Pipeline Steps
# =============================================================================


class PipelineStep:
    """Base class for pipeline steps."""

    name: str = "step"

    def __init__(self, config: PipelineConfig, dry_run: bool = False, force: bool = False) -> None:
        """Initialize the pipeline step.

        Args:
            config: Pipeline configuration.
            dry_run: If True, validate and report without executing.
            force: If True, skip cache and re-run even if output exists.

        """
        self.config = config
        self.dry_run = dry_run
        self.force = force
        self.results: dict[str, Any] = {}

    def run(self) -> bool:
        """Execute the step.

        Returns True on success.

        """
        raise NotImplementedError

    def validate(self) -> list[str]:
        """Validate prerequisites.

        Returns list of error messages.

        """
        return []

    def dry_run_report(self) -> dict[str, Any]:
        """Report what this step would do without executing.

        Returns:
            Dictionary describing planned actions.

        """
        return {"step": self.name, "action": "would execute"}


class DownloadStep(PipelineStep):
    """Download step: Fetch labeled data from EKB API."""

    name = "download"

    def validate(self) -> list[str]:
        """Validate download configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.download

        # Check that we have API credentials (either direct or via config file)
        if cfg.config_file:
            if not Path(cfg.config_file).exists():
                errors.append(f"Download config file not found: {cfg.config_file}")
        else:
            if not cfg.api_url:
                errors.append("API URL is required (download.api_url or download.config_file)")
            if not cfg.api_token:
                errors.append("API token is required (download.api_token or download.config_file)")

        return errors

    def _load_download_config(self) -> Any:
        """Load DownloadConfig from file or pipeline config.

        Returns:
            DownloadConfig instance.

        """
        # Import here to avoid circular imports at module level
        from run_download_data import DownloadConfig

        cfg = self.config.download

        if cfg.config_file and Path(cfg.config_file).exists():
            # Load from YAML file
            download_config = DownloadConfig.from_yaml(Path(cfg.config_file))
        else:
            # Build from pipeline config
            download_config = DownloadConfig()
            download_config.api.url = cfg.api_url
            download_config.api.token = cfg.api_token
            download_config.output.dir = cfg.output_dir

        # Apply overrides from pipeline config
        if cfg.max_labels is not None:
            download_config.limits.max_labels = cfg.max_labels
        if cfg.sources is not None:
            download_config.filters.sources = cfg.sources
        if cfg.label_hierarchy is not None:
            download_config.filters.label_hierarchy = cfg.label_hierarchy

        return download_config

    def _check_for_changes(self, download_config: Any) -> dict[str, Any]:
        """Check if remote data has changed since last download.

        Compares the current API label count with the cached metadata.

        Args:
            download_config: Download configuration.

        Returns:
            Dictionary with change detection results:
            - needs_download: bool indicating if download is needed
            - reason: string explaining why
            - cached_count: number of labels in cache (or None)
            - remote_count: number of labels from API (or None)

        """
        from run_download_data import EKBAPIClient

        metadata_path = (
            Path(download_config.output.dir) / download_config.output.metadata_dir / "download_metadata.json"
        )

        # Check if metadata exists
        if not metadata_path.exists():
            return {
                "needs_download": True,
                "reason": "No previous download metadata found",
                "cached_count": None,
                "remote_count": None,
            }

        # Load cached metadata
        try:
            with open(metadata_path) as f:
                cached_metadata = json.load(f)
            cached_count = cached_metadata.get("download_info", {}).get("total_sounds", 0)
        except (OSError, json.JSONDecodeError) as e:
            return {
                "needs_download": True,
                "reason": f"Could not read cached metadata: {e}",
                "cached_count": None,
                "remote_count": None,
            }

        # Query API for current label count
        try:
            api_client = EKBAPIClient(
                base_url=download_config.api.url,
                token=download_config.api.token,
            )

            # Build filters
            filters = {}
            if download_config.filters.sources:
                filters["sources"] = download_config.filters.sources
            if download_config.filters.label_hierarchy:
                filters["label_hierarchy"] = download_config.filters.label_hierarchy

            # Just get count (limit=1 to minimize data transfer)
            response = api_client.list_labels(limit=1, **filters)
            remote_count = response.get("total", 0)

        except Exception as e:
            return {
                "needs_download": True,
                "reason": f"Could not query API for changes: {e}",
                "cached_count": cached_count,
                "remote_count": None,
            }

        # Compare counts
        if remote_count != cached_count:
            return {
                "needs_download": True,
                "reason": f"Label count changed: {cached_count} cached vs {remote_count} remote",
                "cached_count": cached_count,
                "remote_count": remote_count,
            }

        return {
            "needs_download": False,
            "reason": f"Data unchanged ({cached_count} labels)",
            "cached_count": cached_count,
            "remote_count": remote_count,
        }

    def run(self) -> bool:
        """Execute the download step.

        Returns:
            True if successful, False otherwise.

        """
        from run_download_data import DataDownloader, EKBAPIClient

        cfg = self.config.download
        console.print("\n[bold cyan]Step: Download[/bold cyan]")
        console.print(f"  Output dir: {cfg.output_dir}")

        if cfg.max_labels:
            console.print(f"  Max labels (test mode): {cfg.max_labels}")

        # Load download config
        download_config = self._load_download_config()

        # Check for changes (unless force is set)
        if not cfg.force:
            change_info = self._check_for_changes(download_config)
            console.print(f"  Change detection: {change_info['reason']}")

            if not change_info["needs_download"]:
                console.print("  [green]✓[/green] Data is up to date, skipping download")
                self.results = {
                    "status": "skipped",
                    "skipped": True,
                    "reason": change_info["reason"],
                    "cached_count": change_info["cached_count"],
                }
                # Still discover class names from cached metadata
                self._auto_discover_class_names(download_config)
                return True
        else:
            console.print("  [yellow]Force mode enabled, skipping change detection[/yellow]")

        if self.dry_run:
            report = self.dry_run_report()
            self.results = report
            console.print("  [yellow]DRY RUN[/yellow] - would download:")
            for k, v in report.items():
                console.print(f"    {k}: {v}")
            return True

        try:
            # Initialize API client
            api_client = EKBAPIClient(
                base_url=download_config.api.url,
                token=download_config.api.token,
            )

            # Initialize downloader
            downloader = DataDownloader(
                api_client=api_client,
                config=download_config,
                dry_run=False,
            )

            # Run download
            results = downloader.run()

            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)

            self.results = {
                "total_sounds": len(results),
                "successful": successful,
                "failed": failed,
                "output_dir": str(download_config.output.dir),
            }

            if failed > 0:
                console.print(f"  [yellow]![/yellow] {failed} downloads failed")

            console.print(f"  [green]✓[/green] Downloaded {successful} sounds")

            # Auto-discover class names if not already set
            self._auto_discover_class_names(download_config)

            return failed == 0

        except Exception as e:
            logger.error(f"Download failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _auto_discover_class_names(self, download_config: Any) -> None:
        """Auto-discover class names from the remote EKB API.

        Queries the API for all distinct label hierarchies and populates
        ``self.config.class_names`` with the sorted leaf names. Falls back
        to reading local download metadata if the API is unreachable.

        Args:
            download_config: Download configuration with API credentials
                and output paths.

        """
        if self.config.class_names:
            return  # Already populated, don't override

        from run_download_data import EKBAPIClient

        # Primary: discover from remote API
        try:
            api_client = EKBAPIClient(
                base_url=download_config.api.url,
                token=download_config.api.token,
            )
            discovered = api_client.discover_class_names()
            if discovered:
                self.config.class_names = discovered
                console.print(
                    f"  [green]✓[/green] Discovered {len(discovered)} classes from API: {', '.join(discovered)}"
                )
                self.results["class_names"] = discovered
                return
        except Exception as e:
            logger.warning(f"Could not discover class names from API: {e}")

        # Fallback: discover from local metadata file
        from run_download_data import discover_class_names_from_metadata

        metadata_path = (
            Path(download_config.output.dir) / download_config.output.metadata_dir / "download_metadata.json"
        )

        if not metadata_path.exists():
            return

        try:
            discovered = discover_class_names_from_metadata(metadata_path)
            if discovered:
                self.config.class_names = discovered
                console.print(
                    f"  [green]✓[/green] Discovered {len(discovered)} classes from metadata: {', '.join(discovered)}"
                )
                self.results["class_names"] = discovered
        except Exception as e:
            logger.warning(f"Could not auto-discover class names: {e}")

    def dry_run_report(self) -> dict[str, Any]:
        """Report download plan."""
        cfg = self.config.download
        download_config = self._load_download_config()

        # Try to get label count from API
        label_count = "unknown"
        try:
            from run_download_data import EKBAPIClient

            api_client = EKBAPIClient(
                base_url=download_config.api.url,
                token=download_config.api.token,
            )

            filters = {}
            if download_config.filters.sources:
                filters["sources"] = download_config.filters.sources
            if download_config.filters.label_hierarchy:
                filters["label_hierarchy"] = download_config.filters.label_hierarchy

            response = api_client.list_labels(limit=1, **filters)
            label_count = response.get("total", 0)

            # Apply max_labels limit
            if cfg.max_labels and label_count > cfg.max_labels:
                label_count = f"{cfg.max_labels} (limited from {response.get('total', 0)})"

        except Exception as e:
            label_count = f"error: {e}"

        return {
            "api_url": download_config.api.url,
            "output_dir": download_config.output.dir,
            "label_count": label_count,
            "max_labels": cfg.max_labels,
            "filters": download_config.filters.to_dict() if hasattr(download_config.filters, "to_dict") else {},
        }


class PreprocessStep(PipelineStep):
    """Audio preprocessing step: Audio → Spectrogram chunks."""

    name = "preprocess"

    def validate(self) -> list[str]:
        """Validate preprocessing configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.preprocess

        if not Path(cfg.audio_dir).exists():
            errors.append(f"Audio directory not found: {cfg.audio_dir}")
        if not Path(cfg.metadata_dir).exists():
            errors.append(f"Metadata directory not found: {cfg.metadata_dir}")
        if not Path(cfg.chunking_config).exists():
            errors.append(f"Chunking config not found: {cfg.chunking_config}")

        return errors

    def run(self) -> bool:
        """Execute the preprocessing step.

        Returns:
            True if successful, False otherwise.

        """
        cfg = self.config.preprocess
        console.print("\n[bold cyan]Step: Preprocessing[/bold cyan]")
        console.print(f"  Audio dir: {cfg.audio_dir}")
        console.print(f"  Metadata dir: {cfg.metadata_dir}")
        console.print(f"  Output dir: {cfg.output_dir}")

        if self.dry_run:
            report = self.dry_run_report()
            self.results = report
            console.print("  [yellow]DRY RUN[/yellow] - would process files")
            for k, v in report.items():
                console.print(f"    {k}: {v}")
            return True

        try:
            import numpy as np
            from PIL import Image

            from rf_detr_finetuning.dataprocessor import (
                AudioChunker,
                grayscale_to_rgb,
                load_chunking_config_from_yaml,
                normalize_to_range,
            )

            # Load chunker config
            fft_config, chunk_config, preproc_config = load_chunking_config_from_yaml(Path(cfg.chunking_config))

            chunker = AudioChunker(
                fft_config=fft_config,
                chunk_config=chunk_config,
                preprocessing_config=preproc_config,
            )

            # Find audio-metadata pairs
            audio_dir = Path(cfg.audio_dir)
            metadata_dir = Path(cfg.metadata_dir)
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "images").mkdir(exist_ok=True)

            # Find audio files (recursive or not)
            audio_files = []
            glob_pattern = "**/*" if cfg.recursive else "*"
            for ext in cfg.extensions:
                audio_files.extend(audio_dir.glob(f"{glob_pattern}{ext}"))
                audio_files.extend(audio_dir.glob(f"{glob_pattern}{ext.upper()}"))

            audio_files = sorted(set(audio_files))  # Remove duplicates and sort
            console.print(f"  Found {len(audio_files)} audio files")

            if cfg.max_files:
                audio_files = audio_files[: cfg.max_files]

            # Check if preprocessing output already exists and is complete
            coco_path = output_dir / "_annotations.coco.json"
            if coco_path.exists() and not self.force:
                with open(coco_path) as f:
                    existing_coco = json.load(f)
                existing_images = len(existing_coco.get("images", []))
                existing_annotations = len(existing_coco.get("annotations", []))
                if existing_images > 0 and existing_annotations > 0:
                    console.print(
                        f"  [yellow]Skipping[/yellow] - output already exists with "
                        f"{existing_images} images, {existing_annotations} annotations"
                    )
                    self.results = {
                        "num_images": existing_images,
                        "num_annotations": existing_annotations,
                        "num_classes": len(existing_coco.get("categories", [])),
                        "skipped_already_processed": len(audio_files),
                        "output_dir": str(output_dir),
                        "skipped": True,
                    }
                    return True
            elif coco_path.exists() and self.force:
                console.print("  [cyan]Force re-run[/cyan] - upstream step changed, reprocessing...")

            # Build class mapping
            class_to_id = {}
            all_annotations = []
            all_images = []
            image_id = 0
            annotation_id = 0
            skipped_no_metadata = 0
            processed_files = 0

            with create_file_progress() as progress:
                task = progress.add_task(
                    f"[cyan]Processing {len(audio_files)} audio files...",
                    total=len(audio_files),
                )

                for audio_path in audio_files:
                    # Update description with current file
                    progress.update(
                        task,
                        description=f"[cyan]{audio_path.name[:40]}{'...' if len(audio_path.name) > 40 else ''}",
                    )

                    # Find metadata - check same directory as audio file first
                    metadata_path = audio_path.with_suffix(".json")
                    if not metadata_path.exists():
                        # Fall back to metadata_dir with same filename
                        metadata_path = metadata_dir / f"{audio_path.stem}.json"

                    if not metadata_path.exists():
                        skipped_no_metadata += 1
                        progress.advance(task)
                        continue

                    # Process file - chunk_audio_file handles metadata parsing natively
                    # It extracts labels from label_hierarchy and creates file-level events
                    try:
                        chunks = chunker.chunk_audio_file(
                            audio_path=audio_path,
                            metadata_path=metadata_path,
                        )
                        processed_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to process {audio_path.name}: {e}")
                        progress.advance(task)
                        continue

                    for chunk in chunks:
                        # Save image
                        normalized = normalize_to_range(chunk.spectrogram, 0, 255)
                        rgb = grayscale_to_rgb(normalized.astype(np.uint8))
                        img_name = f"{audio_path.stem}_chunk{chunk.chunk_index:04d}.png"
                        img_path = output_dir / "images" / img_name
                        Image.fromarray(rgb).save(img_path)

                        # Add image entry
                        h, w = chunk.spectrogram.shape
                        all_images.append(
                            {
                                "id": image_id,
                                "file_name": f"images/{img_name}",
                                "width": w,
                                "height": h,
                            }
                        )

                        # Add annotations (1-based IDs per COCO convention)
                        for bbox in chunk.bboxes:
                            label = bbox.category or "unknown"
                            if label not in class_to_id:
                                class_to_id[label] = len(class_to_id) + 1

                            all_annotations.append(
                                {
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": class_to_id[label],
                                    "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                                    "area": bbox.width * bbox.height,
                                    "iscrowd": 0,
                                }
                            )
                            annotation_id += 1

                        image_id += 1

                    progress.advance(task)

                # Final update
                progress.update(task, description=f"[green]Completed {processed_files} files")

            # Build categories (with supercategory - must NOT be "none" for RF-DETR)
            categories = [
                {"id": cid, "name": name, "supercategory": "object"}
                for name, cid in sorted(class_to_id.items(), key=lambda x: x[1])
            ]

            # Save COCO annotations
            coco_data = {
                "images": all_images,
                "annotations": all_annotations,
                "categories": categories,
            }

            with open(output_dir / "_annotations.coco.json", "w") as f:
                json.dump(coco_data, f, indent=2)

            self.results = {
                "num_images": len(all_images),
                "num_annotations": len(all_annotations),
                "num_classes": len(categories),
                "skipped_no_metadata": skipped_no_metadata,
                "output_dir": str(output_dir),
            }

            console.print(
                f"  [green]✓[/green] Generated {len(all_images)} images, "
                f"{len(all_annotations)} annotations, {len(categories)} classes"
            )
            if skipped_no_metadata > 0:
                console.print(f"  [yellow]![/yellow] Skipped {skipped_no_metadata} files (no metadata found)")
            return True

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def dry_run_report(self) -> dict[str, Any]:
        """Report preprocessing plan."""
        cfg = self.config.preprocess
        audio_dir = Path(cfg.audio_dir)
        metadata_dir = Path(cfg.metadata_dir)

        audio_exts = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
        audio_count = 0
        metadata_count = 0
        if audio_dir.exists():
            glob = "**/*" if cfg.recursive else "*"
            for ext in audio_exts:
                audio_count += len(list(audio_dir.glob(f"{glob}{ext}")))
        if metadata_dir.exists():
            metadata_count = len(list(metadata_dir.glob("**/*.json")))

        return {
            "audio_files": audio_count,
            "metadata_files": metadata_count,
            "output_dir": cfg.output_dir,
            "chunking_config": cfg.chunking_config,
            "max_files": cfg.max_files,
        }


class SplitStep(PipelineStep):
    """Dataset splitting step."""

    name = "split"

    def validate(self) -> list[str]:
        """Validate split configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.split

        input_dir = Path(cfg.input_dir)
        if not input_dir.exists():
            errors.append(f"Input directory not found: {cfg.input_dir}")
        elif not (input_dir / "_annotations.coco.json").exists():
            errors.append(f"COCO annotations not found in: {cfg.input_dir}")

        return errors

    def run(self) -> bool:
        """Execute the dataset splitting step.

        Respects pre-existing split structure from the download step.
        If audio_dir contains train/val/test subdirectories, images are
        assigned to splits based on which subdirectory their source audio
        came from. Otherwise, falls back to random splitting.

        Returns:
            True if successful, False otherwise.

        """
        cfg = self.config.split
        console.print("\n[bold cyan]Step: Dataset Splitting[/bold cyan]")
        console.print(f"  Input dir: {cfg.input_dir}")
        console.print(f"  Output dir: {cfg.output_dir}")

        if self.dry_run:
            n_images = 0
            ann_path = Path(cfg.input_dir) / "_annotations.coco.json"
            if ann_path.exists():
                with open(ann_path) as f:
                    n_images = len(json.load(f).get("images", []))
            n_train = int(n_images * cfg.train_ratio)
            n_val = int(n_images * cfg.val_ratio)
            n_test = n_images - n_train - n_val
            self.results = {
                "total_images": n_images,
                "train": n_train,
                "val": n_val,
                "test": n_test,
            }
            console.print("  [yellow]DRY RUN[/yellow] - would split:")
            console.print(f"    train={n_train}, val={n_val}, test={n_test}")
            return True

        try:
            import re

            import numpy as np

            input_dir = Path(cfg.input_dir)
            output_dir = Path(cfg.output_dir)

            # Check if split output already exists and is complete
            train_ann = output_dir / "train" / "_annotations.coco.json"
            valid_ann = output_dir / "valid" / "_annotations.coco.json"
            test_ann = output_dir / "test" / "_annotations.coco.json"
            if train_ann.exists() and valid_ann.exists() and test_ann.exists() and not self.force:
                with open(train_ann) as f:
                    n_train = len(json.load(f).get("images", []))
                with open(valid_ann) as f:
                    n_val = len(json.load(f).get("images", []))
                with open(test_ann) as f:
                    n_test = len(json.load(f).get("images", []))
                if n_train > 0:
                    console.print(
                        f"  [yellow]Skipping[/yellow] - split output already exists "
                        f"(train={n_train}, val={n_val}, test={n_test})"
                    )
                    self.results = {
                        "train_images": n_train,
                        "val_images": n_val,
                        "test_images": n_test,
                        "output_dir": str(output_dir),
                        "skipped": True,
                    }
                    return True

            if self.force:
                console.print("  [cyan]Force re-run[/cyan] - upstream step changed, re-splitting...")

            # Load COCO data
            console.print("  Loading COCO annotations...")
            with open(input_dir / "_annotations.coco.json") as f:
                coco_data = json.load(f)
            console.print(f"  Loaded {len(coco_data['images'])} images")

            # Check if audio_dir has pre-existing split subdirectories
            audio_dir = Path(self.config.preprocess.audio_dir)
            split_dirs = {
                "train": audio_dir / "train",
                "valid": audio_dir / "valid",
                "test": audio_dir / "test",
            }
            has_presplit = any(d.exists() for d in split_dirs.values())

            if has_presplit:
                console.print("  Detected pre-existing splits from download step")
                # Build mapping: audio_stem -> split_name
                stem_to_split = {}
                audio_exts = {".wav", ".flac", ".mp3"}
                for split_name, split_path in split_dirs.items():
                    if not split_path.exists():
                        continue
                    for f in split_path.iterdir():
                        if f.suffix.lower() in audio_exts:
                            stem_to_split[f.stem] = split_name

                # Map chunk images to splits using audio stem
                # Image names follow pattern: {audio_stem}_chunk{NNNN}.png
                chunk_pattern = re.compile(r"^(.+)_chunk\d+\.png$")

                # Build image_id -> split mapping
                image_split_map = {}
                unmatched = 0
                for i, img in enumerate(coco_data["images"]):
                    img_name = Path(img["file_name"]).name
                    match = chunk_pattern.match(img_name)
                    if match:
                        audio_stem = match.group(1)
                        split = stem_to_split.get(audio_stem)
                        if split:
                            image_split_map[i] = split
                        else:
                            unmatched += 1
                            image_split_map[i] = "train"  # fallback
                    else:
                        unmatched += 1
                        image_split_map[i] = "train"

                if unmatched > 0:
                    console.print(
                        f"  [yellow]![/yellow] {unmatched} images could not be mapped to a split, defaulting to train"
                    )

                # Assign images and annotations using the mapping
                id_to_idx = {img["id"]: i for i, img in enumerate(coco_data["images"])}

                splits = {
                    "train": {"images": [], "annotations": [], "categories": coco_data["categories"]},
                    "valid": {"images": [], "annotations": [], "categories": coco_data["categories"]},
                    "test": {"images": [], "annotations": [], "categories": coco_data["categories"]},
                }

                # Map split directory names to output directory names
                split_dir_name: dict[str, str] = {"train": "train", "valid": "valid", "test": "test"}

                for i, img in enumerate(coco_data["images"]):
                    split = image_split_map.get(i, "train")
                    dir_name = split_dir_name.get(split, "train")
                    splits[dir_name]["images"].append(img)

                for ann in coco_data["annotations"]:
                    img_idx = id_to_idx.get(ann["image_id"])
                    if img_idx is not None:
                        split = image_split_map.get(img_idx, "train")
                        dir_name = split_dir_name.get(split, "train")
                        splits[dir_name]["annotations"].append(ann)

            else:
                console.print(
                    f"  No pre-existing splits found, using random split "
                    f"(train={cfg.train_ratio}, val={cfg.val_ratio}, test={cfg.test_ratio})"
                )

                # Random shuffle split (original behavior)
                n = len(coco_data["images"])
                indices = np.arange(n)
                rng = np.random.default_rng(cfg.seed)
                rng.shuffle(indices)

                n_train = int(n * cfg.train_ratio)
                n_val = int(n * cfg.val_ratio)

                train_indices = set(indices[:n_train].tolist())
                val_indices = set(indices[n_train : n_train + n_val].tolist())

                id_to_idx = {img["id"]: i for i, img in enumerate(coco_data["images"])}

                splits = {
                    "train": {"images": [], "annotations": [], "categories": coco_data["categories"]},
                    "valid": {"images": [], "annotations": [], "categories": coco_data["categories"]},
                    "test": {"images": [], "annotations": [], "categories": coco_data["categories"]},
                }

                for i, img in enumerate(coco_data["images"]):
                    if i in train_indices:
                        splits["train"]["images"].append(img)
                    elif i in val_indices:
                        splits["valid"]["images"].append(img)
                    else:
                        splits["test"]["images"].append(img)

                for ann in coco_data["annotations"]:
                    img_idx = id_to_idx.get(ann["image_id"])
                    if img_idx in train_indices:
                        splits["train"]["annotations"].append(ann)
                    elif img_idx in val_indices:
                        splits["valid"]["annotations"].append(ann)
                    else:
                        splits["test"]["annotations"].append(ann)

            # Count total images to copy
            total_images = sum(len(s.get("images", [])) for s in splits.values())

            # Save splits with progress
            with create_file_progress() as progress:
                task = progress.add_task(
                    "[cyan]Copying images to splits...",
                    total=total_images,
                )

                for split_name, split_data in splits.items():
                    split_dir = output_dir / split_name
                    split_dir.mkdir(parents=True, exist_ok=True)

                    # Copy images
                    skipped = 0
                    for img_info in split_data["images"]:
                        progress.update(
                            task, description=f"[cyan]{split_name}: {Path(img_info['file_name']).name[:30]}"
                        )
                        src = input_dir / img_info["file_name"]
                        dst = split_dir / Path(img_info["file_name"]).name
                        if src.exists():
                            shutil.copy2(src, dst)
                            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                                logger.warning(f"Copy verification failed: {src} → {dst}")
                        else:
                            skipped += 1
                            logger.warning(f"Source image not found, skipping: {src}")
                        # Update file_name to be relative
                        img_info["file_name"] = Path(img_info["file_name"]).name
                        progress.advance(task)

                    if skipped:
                        console.print(f"  [yellow]![/yellow] {split_name}: {skipped} images not found")

                    # Save annotations
                    with open(split_dir / "_annotations.coco.json", "w") as f:
                        json.dump(split_data, f, indent=2)

            self.results = {
                "train_images": len(splits.get("train", {}).get("images", [])),
                "val_images": len(splits.get("valid", {}).get("images", [])),
                "test_images": len(splits.get("test", {}).get("images", [])),
                "output_dir": str(output_dir),
            }

            console.print(
                f"  [green]✓[/green] Split: train={self.results['train_images']}, "
                f"val={self.results['val_images']}, test={self.results['test_images']}"
            )
            return True

        except Exception as e:
            logger.error(f"Splitting failed: {e}")
            import traceback

            traceback.print_exc()
            return False


class TrainStep(PipelineStep):
    """Training step."""

    name = "train"

    def validate(self) -> list[str]:
        """Validate training configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.train

        dataset_dir = Path(cfg.dataset_dir)
        if not dataset_dir.exists():
            errors.append(f"Dataset directory not found: {cfg.dataset_dir}")
        else:
            if not (dataset_dir / "train").exists():
                errors.append(f"Train split not found: {cfg.dataset_dir}/train")

        return errors

    def run(self) -> bool:
        """Execute the training step.

        Returns:
            True if successful, False otherwise.

        """
        cfg = self.config.train
        console.print("\n[bold cyan]Step: Training[/bold cyan]")
        console.print(f"  Dataset: {cfg.dataset_dir}")
        console.print(f"  Model: RF-DETR {cfg.model_size}")
        console.print(f"  Epochs: {cfg.epochs}")
        console.print(f"  Batch size: {cfg.batch_size}")

        if self.dry_run:
            self.results = {
                "model_size": cfg.model_size,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "dataset_dir": cfg.dataset_dir,
            }
            console.print("  [yellow]DRY RUN[/yellow] - would train:")
            console.print(f"    {cfg.model_size} for {cfg.epochs} epochs")
            return True

        try:
            from rf_detr_finetuning.trainer import (
                CheckpointConfig,
                OptimizerConfig,
                RFDETRConfig,
                RFDETRTrainer,
                TrainerConfig,
            )

            # Count classes from dataset
            dataset_path = Path(cfg.dataset_dir)
            train_annotations = dataset_path / "train" / "_annotations.coco.json"
            num_classes = 1  # default

            if train_annotations.exists():
                with open(train_annotations) as f:
                    coco_data = json.load(f)
                    num_classes = len(coco_data.get("categories", []))
                    console.print(f"  Detected {num_classes} classes from dataset")

            # Create trainer configuration
            trainer_config = TrainerConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                device=cfg.device,
                optimizer=OptimizerConfig(lr=cfg.learning_rate),
                checkpoint=CheckpointConfig(
                    save_dir=cfg.output_dir,
                    save_every_n_epochs=cfg.save_every_n_epochs,
                    resume_from=cfg.resume_from,
                ),
            )

            # Create model configuration
            model_config = RFDETRConfig(
                model_size=cfg.model_size,
                num_classes=num_classes,
                pretrained_weights=cfg.pretrained_weights,
            )

            # Initialize trainer
            trainer = RFDETRTrainer(
                model_config=model_config,
                trainer_config=trainer_config,
            )

            # Train (RF-DETR expects the parent directory containing train/valid/test)
            results = trainer.train(
                dataset_path=cfg.dataset_dir,
                output_dir=cfg.output_dir,
            )

            self.results = {
                "training_results": results,
                "output_dir": cfg.output_dir,
            }

            console.print("  [green]✓[/green] Training complete")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback

            traceback.print_exc()
            return False


class EvaluateStep(PipelineStep):
    """Evaluation step."""

    name = "evaluate"

    def validate(self) -> list[str]:
        """Validate evaluation configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.evaluate

        if not Path(cfg.weights).exists():
            errors.append(f"Model weights not found: {cfg.weights}")
        if not Path(cfg.test_dir).exists():
            errors.append(f"Test directory not found: {cfg.test_dir}")

        return errors

    def run(self) -> bool:
        """Execute the evaluation step.

        Returns:
            True if successful, False otherwise.

        """
        cfg = self.config.evaluate
        console.print("\n[bold cyan]Step: Evaluation[/bold cyan]")
        console.print(f"  Weights: {cfg.weights}")
        console.print(f"  Test dir: {cfg.test_dir}")

        if self.dry_run:
            n_images = 0
            ann_path = Path(cfg.test_dir) / "_annotations.coco.json"
            if ann_path.exists():
                with open(ann_path) as f:
                    n_images = len(json.load(f).get("images", []))
            self.results = {
                "test_images": n_images,
                "weights": cfg.weights,
                "confidence_threshold": cfg.confidence_threshold,
            }
            console.print("  [yellow]DRY RUN[/yellow] - would evaluate:")
            console.print(f"    {n_images} test images")
            return True

        try:
            import numpy as np
            from PIL import Image

            from rf_detr_finetuning.predictor import RFDETRPredictor
            from rf_detr_finetuning.trainer import compute_coco_metrics

            # Load class names
            test_dir = Path(cfg.test_dir)
            coco_path = test_dir / "_annotations.coco.json"

            with open(coco_path) as f:
                coco_data = json.load(f)

            sorted_categories = sorted(coco_data["categories"], key=lambda x: x["id"])
            class_names = [c["name"] for c in sorted_categories]
            idx_to_category_id = {idx: c["id"] for idx, c in enumerate(sorted_categories)}
            valid_category_ids = {c["id"] for c in sorted_categories}

            # Initialize predictor
            predictor = RFDETRPredictor(
                model_size=self.config.train.model_size,
                weights_path=Path(cfg.weights),
                class_names=class_names,
            )

            # Run predictions
            predictions = []
            ground_truths = []

            # Build image_id to annotations map
            img_to_anns = {}
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)

            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"  Running inference on {len(coco_data['images'])} test images...")

            with create_file_progress() as progress:
                task = progress.add_task(
                    "[cyan]Evaluating...",
                    total=len(coco_data["images"]),
                )

                for img_info in coco_data["images"]:
                    progress.update(task, description=f"[cyan]{img_info['file_name'][:35]}")
                    img_path = test_dir / img_info["file_name"]
                    if not img_path.exists():
                        progress.advance(task)
                        continue

                    # Load image
                    img = np.array(Image.open(img_path).convert("RGB"))

                    # Predict
                    result = predictor.predict(img, cfg.confidence_threshold)

                    # Store predictions
                    for det in result.detections:
                        category_id = idx_to_category_id.get(det.class_id)
                        if category_id is None and det.class_id in valid_category_ids:
                            category_id = det.class_id
                        if category_id is None:
                            continue
                        predictions.append(
                            {
                                "image_id": img_info["id"],
                                "category_id": category_id,
                                "bbox": [det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1],
                                "score": det.score,
                            }
                        )

                    # Store ground truth
                    for ann in img_to_anns.get(img_info["id"], []):
                        ground_truths.append(ann)

                    progress.advance(task)

            # Compute metrics
            metrics = compute_coco_metrics(predictions, ground_truths, coco_data["categories"])

            self.results = {
                "mAP": metrics.get("mAP", 0),
                "mAP50": metrics.get("mAP50", 0),
                "mAP75": metrics.get("mAP75", 0),
                "num_predictions": len(predictions),
                "num_ground_truths": len(ground_truths),
            }

            # Save results
            with open(output_dir / "eval_results.json", "w") as f:
                json.dump(self.results, f, indent=2)

            console.print(f"  [green]✓[/green] mAP={metrics.get('mAP', 0):.3f}, mAP50={metrics.get('mAP50', 0):.3f}")
            return True

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return False


class OptimizeMergingStep(PipelineStep):
    """Optimize merge parameters via grid search on test audio files."""

    name = "optimize_merging"

    def validate(self) -> list[str]:
        """Validate optimization configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.optimize_merging

        if not Path(cfg.weights).exists():
            errors.append(f"Model weights not found: {cfg.weights}")
        if not Path(cfg.metadata).exists():
            errors.append(f"Download metadata not found: {cfg.metadata}")
        if not Path(cfg.chunking_config).exists():
            errors.append(f"Chunking config not found: {cfg.chunking_config}")

        return errors

    def run(self) -> bool:
        """Execute the merge parameter optimization step.

        Returns:
            True if successful, False otherwise.

        """
        cfg = self.config.optimize_merging
        console.print("\n[bold cyan]Step: Optimize Merging[/bold cyan]")
        console.print(f"  Metadata: {cfg.metadata}")
        console.print(f"  Weights: {cfg.weights}")
        console.print(f"  Output: {cfg.output}")
        console.print(f"  Mode: {'per-class' if cfg.per_class else 'default-only'}")

        if self.dry_run:
            n_test = 0
            metadata_path = Path(cfg.metadata)
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                n_test = sum(
                    1 for s in metadata.get("sounds", []) if s.get("split") == "test" and s.get("success", False)
                )
            n_combos = (
                len(cfg.delta_time_values)
                * len(cfg.delta_freq_values)
                * len(cfg.score_threshold_values)
                * len(cfg.min_duration_values)
            )
            self.results = {
                "test_files": n_test,
                "search_combinations": n_combos,
                "mode": "per-class" if cfg.per_class else "default-only",
            }
            console.print("  [yellow]DRY RUN[/yellow] - would optimize:")
            console.print(f"    {n_test} test files, {n_combos} parameter combinations")
            return True

        try:
            from rf_detr_finetuning.eventprocessor.optimizer import (
                GlobalSearchSpace,
                SearchSpace,
                optimize_default_only,
                optimize_per_class,
                save_config_yaml,
            )
            from rf_detr_finetuning.utils import load_class_names
            from run_optimize_merging import (
                build_class_mapping,
                display_config_table,
                display_eval_table,
                events_to_audio_events,
                load_test_data,
                run_inference_on_test_files,
            )

            # Load test data
            metadata_path = Path(cfg.metadata)
            audio_dir = Path(cfg.audio_dir) if cfg.audio_dir else None
            test_sounds = load_test_data(metadata_path, audio_dir)

            if not test_sounds:
                console.print("  [red]No test audio files found in metadata[/red]")
                return False

            if cfg.max_files:
                test_sounds = test_sounds[: cfg.max_files]

            console.print(f"  Found {len(test_sounds)} test audio files")

            # Resolve class names
            class_names = load_class_names(
                self.config.class_names or None,
                self.config.classes_file,
            )

            # Try to load from checkpoint
            try:
                import torch

                checkpoint = torch.load(Path(cfg.weights), map_location="cpu", weights_only=False)
                if hasattr(checkpoint.get("args", object()), "class_names"):
                    class_names = checkpoint["args"].class_names
            except Exception:
                pass

            label_to_class = build_class_mapping(test_sounds, class_names)
            class_names_dict = {v: k for k, v in label_to_class.items()}
            console.print(f"  Classes: {list(label_to_class.keys())}")

            # Build ground truth
            gt_per_file = [events_to_audio_events(sound["events"], label_to_class) for sound in test_sounds]
            total_gt = sum(len(gt) for gt in gt_per_file)
            console.print(f"  Ground truth events: {total_gt}")

            # Diagnostic: show GT class distribution to detect mapping issues
            from collections import Counter

            gt_class_ids = Counter()
            gt_unmapped = Counter()
            for sound in test_sounds:
                for event in sound["events"]:
                    hierarchy = event.get("label_hierarchy", "")
                    label = hierarchy.split(" + ")[-1] if hierarchy else "unknown"
                    if label in label_to_class:
                        gt_class_ids[label] += 1
                    else:
                        gt_unmapped[label] += 1

            if gt_unmapped:
                console.print(
                    f"  [yellow]WARNING: {sum(gt_unmapped.values())} GT events with unmapped labels "
                    f"(defaulting to class_id=0): {dict(gt_unmapped)}[/yellow]"
                )
            console.print(f"  GT class distribution: {dict(gt_class_ids)}")

            # Run inference to get raw detections
            console.print("  Running inference on test files...")
            raw_events_per_file = run_inference_on_test_files(
                test_sounds,
                weights_path=Path(cfg.weights),
                chunking_config_path=Path(cfg.chunking_config),
                model_size=self.config.train.model_size,
                device=self.config.train.device,
                class_names=class_names,
                confidence=cfg.confidence,
            )

            total_raw = sum(len(e) for e in raw_events_per_file)
            console.print(f"  Raw detections: {total_raw}")

            # Diagnostic: show prediction class distribution
            pred_class_ids = Counter()
            for event_list in raw_events_per_file:
                for event in event_list:
                    pred_class_ids[event.class_id] += 1
            console.print(f"  Prediction class distribution: {dict(pred_class_ids)}")

            # Check for class overlap between GT and predictions
            gt_ids_set = set(e.class_id for gt in gt_per_file for e in gt)
            pred_ids_set = set(e.class_id for el in raw_events_per_file for e in el)
            overlap = gt_ids_set & pred_ids_set
            if not overlap:
                console.print(
                    f"  [red]WARNING: No overlap between GT class IDs {gt_ids_set} "
                    f"and prediction class IDs {pred_ids_set}. "
                    f"Evaluation will produce F1=0![/red]"
                )
            elif overlap != gt_ids_set:
                missing = gt_ids_set - pred_ids_set
                console.print(f"  [yellow]WARNING: GT has classes {missing} with no predictions[/yellow]")

            # Set up search space
            default_search = SearchSpace(
                delta_time_ms=cfg.delta_time_values,
                delta_freq_hz=cfg.delta_freq_values,
                score_strategy=["max"],
            )
            global_search = GlobalSearchSpace(
                score_threshold=cfg.score_threshold_values,
                min_duration_ms=cfg.min_duration_values,
            )

            total_combos = default_search.num_combinations * global_search.num_combinations
            console.print(f"  Searching {total_combos} parameter combinations...")

            # Run optimization
            with create_file_progress() as progress:
                if cfg.per_class:
                    class_ids = sorted(label_to_class.values())
                    per_class_combos = default_search.num_combinations * len(class_ids)
                    total_with_per_class = total_combos + per_class_combos
                    task = progress.add_task("Optimizing...", total=total_with_per_class)

                    def progress_cb(phase: str, current: int, total: int) -> None:
                        if phase == "defaults":
                            progress.update(
                                task,
                                completed=current,
                                description=f"Phase 1: defaults ({current}/{total})",
                            )
                        else:
                            base = total_combos
                            progress.update(
                                task,
                                completed=base + current,
                                description=f"Phase 2: {phase} ({current}/{total})",
                            )

                    result = optimize_per_class(
                        raw_events_per_file,
                        gt_per_file,
                        class_ids=class_ids,
                        default_search=default_search,
                        global_search=global_search,
                        iou_threshold=cfg.iou_threshold,
                        class_names=class_names_dict,
                        progress_callback=progress_cb,
                    )
                else:
                    task = progress.add_task("Optimizing...", total=total_combos)

                    def progress_cb_default(current: int, total: int) -> None:
                        progress.update(
                            task,
                            completed=current,
                            description=f"Searching ({current}/{total})",
                        )

                    result = optimize_default_only(
                        raw_events_per_file,
                        gt_per_file,
                        default_search=default_search,
                        global_search=global_search,
                        iou_threshold=cfg.iou_threshold,
                        class_names=class_names_dict,
                        progress_callback=progress_cb_default,
                    )

            if not result.best_config or not result.best_eval:
                console.print("  [red]Optimization failed: no valid configuration found[/red]")
                return False

            # Display results
            console.print(f"  [green]Best F1: {result.best_f1:.4f}[/green] ({result.trials_evaluated} trials)")
            display_config_table(result.best_config, class_names_dict)
            display_eval_table(result.best_eval, "Optimization Results")

            # Save config
            save_config_yaml(result.best_config, cfg.output, class_names_dict)
            console.print(f"  [green]Optimized config saved to:[/green] {cfg.output}")

            # Save detailed results JSON
            if cfg.results_json:
                results_path = Path(cfg.results_json)
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

            self.results = {
                "best_f1": round(result.best_f1, 4),
                "trials": result.trials_evaluated,
                "test_files": len(test_sounds),
                "raw_detections": total_raw,
                "gt_events": total_gt,
            }

            return True

        except Exception as e:
            logger.error(f"Merge optimization failed: {e}")
            import traceback

            traceback.print_exc()
            return False


class InferStep(PipelineStep):
    """Inference step on new audio files."""

    name = "infer"

    def validate(self) -> list[str]:
        """Validate inference configuration.

        Returns:
            List of validation error messages.

        """
        errors = []
        cfg = self.config.infer

        if not Path(cfg.weights).exists():
            errors.append(f"Model weights not found: {cfg.weights}")
        if not Path(cfg.audio_dir).exists():
            errors.append(f"Audio directory not found: {cfg.audio_dir}")
        if not Path(cfg.chunking_config).exists():
            errors.append(f"Chunking config not found: {cfg.chunking_config}")

        return errors

    def run(self) -> bool:
        """Execute the inference step.

        Returns:
            True if successful, False otherwise.

        """
        cfg = self.config.infer
        console.print("\n[bold cyan]Step: Inference[/bold cyan]")
        console.print(f"  Audio dir: {cfg.audio_dir}")
        console.print(f"  Output dir: {cfg.output_dir}")

        if self.dry_run:
            audio_dir = Path(cfg.audio_dir)
            audio_count = 0
            if audio_dir.exists():
                glob = "**/*" if cfg.recursive else "*"
                for ext in cfg.extensions:
                    audio_count += len(list(audio_dir.glob(f"{glob}{ext}")))
            self.results = {
                "audio_files": audio_count,
                "output_dir": cfg.output_dir,
                "confidence_threshold": cfg.confidence_threshold,
            }
            console.print("  [yellow]DRY RUN[/yellow] - would infer:")
            console.print(f"    {audio_count} audio files")
            return True

        try:
            # Import the audio inference pipeline
            from run_inference_audio import AudioInferencePipeline

            # Get class names
            class_names = self.config.class_names
            if self.config.classes_file and Path(self.config.classes_file).exists():
                with open(self.config.classes_file) as f:
                    data = json.load(f)
                if "categories" in data:
                    class_names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]

            pipeline = AudioInferencePipeline(
                weights_path=Path(cfg.weights),
                config_path=Path(cfg.chunking_config),
                model_size=self.config.train.model_size,
                class_names=class_names,
                confidence_threshold=cfg.confidence_threshold,
                merge_gap_ms=cfg.merge_gap_ms,
            )

            # Find audio files (recursive or not)
            audio_dir = Path(cfg.audio_dir)
            audio_files = []
            glob_pattern = "**/*" if cfg.recursive else "*"
            for ext in cfg.extensions:
                audio_files.extend(audio_dir.glob(f"{glob_pattern}{ext}"))
                audio_files.extend(audio_dir.glob(f"{glob_pattern}{ext.upper()}"))

            audio_files = sorted(set(audio_files))  # Remove duplicates and sort
            console.print(f"  Found {len(audio_files)} audio files")

            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            all_results = []

            with create_file_progress() as progress:
                task = progress.add_task(
                    "[cyan]Running inference...",
                    total=len(audio_files),
                )

                for audio_path in audio_files:
                    progress.update(task, description=f"[cyan]{audio_path.name[:40]}")
                    result = pipeline.process_audio(audio_path)
                    all_results.append(result)

                    # Save individual result
                    output_path = output_dir / f"{audio_path.stem}_events.{cfg.output_format}"
                    if cfg.output_format == "json":
                        with open(output_path, "w") as f:
                            json.dump(result.to_dict(), f, indent=2)
                    elif cfg.output_format == "csv":
                        with open(output_path, "w") as f:
                            f.write(result.to_csv())
                    elif cfg.output_format == "raven":
                        output_path = output_path.with_suffix(".txt")
                        with open(output_path, "w") as f:
                            f.write(result.to_raven())

                    progress.advance(task)

            total_events = sum(len(r.events) for r in all_results)

            self.results = {
                "num_files": len(all_results),
                "total_events": total_events,
                "output_dir": str(output_dir),
            }

            console.print(f"  [green]✓[/green] Processed {len(all_results)} files, detected {total_events} events")
            return True

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback

            traceback.print_exc()
            return False


# =============================================================================
# Pipeline Runner
# =============================================================================


class Pipeline:
    """Main pipeline orchestrator."""

    STEPS = [
        ("download", DownloadStep),
        ("preprocess", PreprocessStep),
        ("split", SplitStep),
        ("train", TrainStep),
        ("evaluate", EvaluateStep),
        ("optimize_merging", OptimizeMergingStep),
        ("infer", InferStep),
    ]

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.

        """
        self.config = config
        self.step_results: dict[str, dict] = {}

    def run(
        self,
        download: bool | None = None,
        preprocess: bool | None = None,
        split: bool | None = None,
        train: bool | None = None,
        evaluate: bool | None = None,
        optimize_merging: bool | None = None,
        infer: bool | None = None,
        run_all: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Run the pipeline with selected steps.

        Args:
            download: Override download step enabled.
            preprocess: Override preprocess step enabled.
            split: Override split step enabled.
            train: Override train step enabled.
            evaluate: Override evaluate step enabled.
            optimize_merging: Override merge optimization step enabled.
            infer: Override infer step enabled.
            run_all: Run all steps (overrides config).
            dry_run: If True, validate and report without executing.

        Returns:
            True if all enabled steps succeeded.

        """
        # Determine which steps to run
        steps_to_run = []

        overrides = {
            "download": download,
            "preprocess": preprocess,
            "split": split,
            "train": train,
            "evaluate": evaluate,
            "optimize_merging": optimize_merging,
            "infer": infer,
        }

        for step_name, step_cls in self.STEPS:
            step_config = getattr(self.config, step_name)

            # Determine if step should run
            if run_all:
                should_run = True
            elif overrides[step_name] is not None:
                should_run = overrides[step_name]
            else:
                should_run = step_config.enabled

            if should_run:
                steps_to_run.append((step_name, step_cls))

        if not steps_to_run:
            console.print("[yellow]No steps enabled to run[/yellow]")
            return True

        # Display plan
        mode_label = " [yellow](DRY RUN)[/yellow]" if dry_run else ""
        console.print(
            Panel(
                f"[bold]{self.config.name}[/bold]{mode_label}\n{self.config.description}",
                title="Pipeline",
            )
        )

        step_list = ", ".join(s[0] for s in steps_to_run)
        console.print(f"\n[cyan]Steps to run:[/cyan] {step_list}\n")

        # Build set of steps that will run (for dependency checking)
        steps_running = {s[0] for s in steps_to_run}

        # Validate steps - but skip validation for steps whose inputs
        # will be created by earlier steps in this run
        # Dependencies: preprocess->download, split->preprocess, train->split, etc.
        step_dependencies = {
            "preprocess": ["download"],  # preprocess needs downloaded data
            "split": ["preprocess"],  # split needs preprocess output
            "train": ["split"],  # train needs split output
            "evaluate": ["train"],  # evaluate needs trained model
            "optimize_merging": ["train"],  # optimize needs trained model + test data
            "infer": ["train"],  # infer needs trained model
        }

        all_errors = []
        for step_name, step_cls in steps_to_run:
            # Check if this step's dependencies are being run earlier
            deps = step_dependencies.get(step_name, [])
            deps_will_run = all(d in steps_running for d in deps)

            # Only validate if dependencies won't create the required inputs
            if not deps_will_run:
                step = step_cls(self.config, dry_run=dry_run)
                errors = step.validate()
                if errors:
                    all_errors.extend([f"[{step_name}] {e}" for e in errors])
            else:
                console.print(f"  [dim]Skipping validation for '{step_name}' (deps will run first)[/dim]")

        if all_errors:
            console.print("[red]Validation errors:[/red]")
            for error in all_errors:
                console.print(f"  • {error}")
            return False

        # Initialize progress manager
        progress_manager = PipelineProgressManager(steps_to_run)

        # Mark steps not in our run as skipped (for display)
        all_step_names = {name for name, _ in self.STEPS}
        running_step_names = {name for name, _ in steps_to_run}
        for name in all_step_names - running_step_names:
            progress_manager.step_status[name] = "skipped"

        # Run steps with overall progress tracking
        start_time = datetime.now()
        success = True
        # Track which steps actually executed (vs skipped from cache)
        steps_executed: set[str] = set()

        # Display initial status
        console.print("\n")
        console.print(progress_manager.render_status_table())
        console.print("\n")

        for idx, (step_name, step_cls) in enumerate(steps_to_run):
            # Force re-run if any upstream dependency actually executed
            deps = step_dependencies.get(step_name, [])
            force = any(d in steps_executed for d in deps)
            step = step_cls(self.config, dry_run=dry_run, force=force)
            progress_manager.start_step(step_name)

            try:
                step_success = step.run()
                self.step_results[step_name] = step.results

                # Track if this step actually ran (not skipped from cache)
                if not step.results.get("skipped", False):
                    steps_executed.add(step_name)

                progress_manager.complete_step(step_name, success=step_success)

                if not step_success:
                    console.print(f"\n[red]✗ Step '{step_name}' failed[/red]")
                    success = False
                    break

            except Exception as e:
                progress_manager.complete_step(step_name, success=False)
                console.print(f"\n[red]✗ Step '{step_name}' raised exception: {e}[/red]")
                import traceback

                traceback.print_exc()
                success = False
                break

            # Show updated status after each step
            console.print("\n")
            console.print(progress_manager.render_status_table())

        # Summary
        elapsed = datetime.now() - start_time
        console.print(f"\n{'=' * 60}")

        if success:
            console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
        else:
            console.print("[bold red]✗ Pipeline failed![/bold red]")

        # Format elapsed time nicely
        total_seconds = int(elapsed.total_seconds())
        if total_seconds < 60:
            time_str = f"{total_seconds} seconds"
        elif total_seconds < 3600:
            minutes, seconds = divmod(total_seconds, 60)
            time_str = f"{minutes}m {seconds}s"
        else:
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours}h {minutes}m {seconds}s"

        console.print(f"[bold]Total time:[/bold] {time_str}")

        # Final status table with durations
        console.print("\n")
        console.print(progress_manager.render_status_table())

        # Results table
        if self.step_results:
            console.print("\n")
            table = Table(title="Step Results", show_header=True, header_style="bold blue")
            table.add_column("Step", style="cyan", width=12)
            table.add_column("Key Metrics", style="green")

            for step_name, results in self.step_results.items():
                metrics = ", ".join(f"{k}={v}" for k, v in results.items() if not k.endswith("_dir"))
                table.add_row(step_name, metrics or "completed")

            console.print(table)

        return success


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RF-DETR Pipeline for Audio Event Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=Path,
        help="Pipeline configuration YAML file",
    )
    parser.add_argument(
        "--generate-config",
        type=Path,
        metavar="PATH",
        help="Generate a template configuration file",
    )

    # Step toggles
    step_group = parser.add_argument_group("Pipeline Steps")
    step_group.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline steps",
    )
    step_group.add_argument(
        "--download",
        action="store_true",
        help="Run download step (fetch data from EKB API)",
    )
    step_group.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing step",
    )
    step_group.add_argument(
        "--split",
        action="store_true",
        help="Run dataset splitting step",
    )
    step_group.add_argument(
        "--train",
        action="store_true",
        help="Run training step",
    )
    step_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation step",
    )
    step_group.add_argument(
        "--optimize-merging",
        action="store_true",
        help="Run merge parameter optimization step",
    )
    step_group.add_argument(
        "--infer",
        action="store_true",
        help="Run inference step",
    )

    # Skip toggles
    skip_group = parser.add_argument_group("Skip Steps")
    skip_group.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download step",
    )
    skip_group.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing step",
    )
    skip_group.add_argument(
        "--no-split",
        action="store_true",
        help="Skip splitting step",
    )
    skip_group.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training step",
    )
    skip_group.add_argument(
        "--no-optimize-merging",
        action="store_true",
        help="Skip merge optimization step",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and report what each step would do without executing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entrypoint."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Pipeline[/bold blue]\n")

    # Generate config template
    if args.generate_config:
        config = PipelineConfig()
        config.to_yaml(args.generate_config)
        console.print(f"[green]Generated config template:[/green] {args.generate_config}")
        return 0

    # Load config
    if not args.config:
        console.print("[red]Error:[/red] --config is required (or use --generate-config)")
        return 1

    if not args.config.exists():
        console.print(f"[red]Error:[/red] Config not found: {args.config}")
        return 1

    config = PipelineConfig.from_yaml(args.config)

    # Determine step overrides
    # If specific steps are requested, use those
    # If --no-X is specified, disable that step
    any_step_requested = (
        args.download
        or args.preprocess
        or args.split
        or args.train
        or args.evaluate
        or args.optimize_merging
        or args.infer
    )

    download = None
    preprocess = None
    split = None
    train = None
    evaluate = None
    optimize_merging = None
    infer = None

    if any_step_requested:
        download = args.download
        preprocess = args.preprocess
        split = args.split
        train = args.train
        evaluate = args.evaluate
        optimize_merging = args.optimize_merging
        infer = args.infer
    else:
        # Use config defaults, but apply --no-X overrides
        if args.no_download:
            download = False
        if args.no_preprocess:
            preprocess = False
        if args.no_split:
            split = False
        if args.no_train:
            train = False
        if args.no_optimize_merging:
            optimize_merging = False

    # Run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run(
        download=download,
        preprocess=preprocess,
        split=split,
        train=train,
        evaluate=evaluate,
        optimize_merging=optimize_merging,
        infer=infer,
        run_all=args.all,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
