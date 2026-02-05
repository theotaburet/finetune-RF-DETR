"""Training logging and metrics tracking.

Provides:
- Rich console logging
- Metrics accumulation and reporting
- Training progress display

"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logger = logging.getLogger(__name__)


@dataclass
class MetricsTracker:
    """Tracks and aggregates training metrics.

    Accumulates metrics over batches/epochs and computes statistics.

    Attributes:
        metrics: Dictionary mapping metric names to lists of values.
        epoch_metrics: Aggregated metrics per epoch.

    """

    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    epoch_metrics: list[dict[str, float]] = field(default_factory=list)

    def update(self, **kwargs: float) -> None:
        """Add metric values.

        Args:
            **kwargs: Metric name-value pairs.

        """
        for name, value in kwargs.items():
            self.metrics[name].append(value)

    def get_mean(self, name: str) -> float:
        """Get mean of accumulated metric.

        Args:
            name: Metric name.

        Returns:
            Mean value, or 0 if no values.

        """
        values = self.metrics.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_last(self, name: str) -> float:
        """Get last value of metric.

        Args:
            name: Metric name.

        Returns:
            Last value, or 0 if no values.

        """
        values = self.metrics.get(name, [])
        return values[-1] if values else 0.0

    def end_epoch(self) -> dict[str, float]:
        """Finalize epoch metrics.

        Computes mean of all accumulated metrics and stores for the epoch.
        Resets per-batch accumulators.

        Returns:
            Dictionary of mean metric values for the epoch.

        """
        epoch_summary = {name: self.get_mean(name) for name in self.metrics}
        self.epoch_metrics.append(epoch_summary)

        # Reset for next epoch
        self.metrics = defaultdict(list)

        return epoch_summary

    def get_history(self, name: str) -> list[float]:
        """Get metric history across all epochs.

        Args:
            name: Metric name.

        Returns:
            List of epoch values.

        """
        return [epoch.get(name, 0.0) for epoch in self.epoch_metrics]

    def to_dict(self) -> dict[str, Any]:
        """Convert tracker to dictionary for serialization."""
        return {
            "epoch_metrics": self.epoch_metrics,
        }

    def save(self, path: str | Path) -> None:
        """Save metrics to JSON file.

        Args:
            path: Output file path.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> MetricsTracker:
        """Load metrics from JSON file.

        Args:
            path: Input file path.

        Returns:
            MetricsTracker instance.

        """
        with open(path) as f:
            data = json.load(f)

        tracker = cls()
        tracker.epoch_metrics = data.get("epoch_metrics", [])
        return tracker


class TrainingLogger:
    """Rich-based training logger with progress bars.

    Provides formatted console output for training progress.

    Args:
        total_epochs: Total number of training epochs.
        total_batches: Batches per epoch.
        log_dir: Directory for log files.

    """

    def __init__(
        self,
        total_epochs: int,
        total_batches: int,
        log_dir: str | Path | None = None,
    ) -> None:
        """Initialize training logger.

        Args:
            total_epochs: Total number of training epochs.
            total_batches: Total number of batches per epoch.
            log_dir: Optional directory for log files.

        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.log_dir = Path(log_dir) if log_dir else None
        self.console = Console()
        self.metrics_tracker = MetricsTracker()

        self._epoch_start_time: float = 0
        self._train_start_time: float = 0
        self._progress: Progress | None = None
        self._epoch_task = None
        self._batch_task = None

    def on_train_start(self) -> None:
        """Called at start of training."""
        self._train_start_time = time.time()
        self.console.print(f"[bold blue]Starting training for {self.total_epochs} epochs[/bold blue]")
        self.console.print(f"Batches per epoch: {self.total_batches}")

    def on_train_end(self) -> None:
        """Called at end of training."""
        elapsed = time.time() - self._train_start_time
        self.console.print(f"\n[bold green]Training complete![/bold green] Total time: {elapsed / 60:.1f} minutes")

        # Save metrics
        if self.log_dir:
            self.metrics_tracker.save(self.log_dir / "metrics.json")

    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        """
        self._epoch_start_time = time.time()
        self.console.print(f"\n[bold]Epoch {epoch + 1}/{self.total_epochs}[/bold]")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at end of each epoch.

        Args:
            epoch: Current epoch number.
            metrics: Epoch metrics dictionary.

        """
        elapsed = time.time() - self._epoch_start_time

        # Update tracker
        for name, value in metrics.items():
            self.metrics_tracker.update(**{name: value})
        self.metrics_tracker.end_epoch()

        # Display metrics table
        table = Table(title=f"Epoch {epoch + 1} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for name, value in metrics.items():
            table.add_row(name, f"{value:.4f}")

        table.add_row("Time", f"{elapsed:.1f}s")

        self.console.print(table)

    def on_batch_end(self, batch_idx: int, loss: float, **extra: float) -> None:
        """Called after each batch.

        Args:
            batch_idx: Current batch index.
            loss: Batch loss value.
            **extra: Additional metrics.

        """
        self.metrics_tracker.update(loss=loss, **extra)

    def create_epoch_progress(self) -> Progress:
        """Create Rich progress bar for epoch iteration.

        Returns:
            Progress context manager.

        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def log_info(self, message: str) -> None:
        """Log info message.

        Args:
            message: Message to log.

        """
        self.console.print(f"[blue]ℹ[/blue] {message}")
        logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Message to log.

        """
        self.console.print(f"[yellow]⚠[/yellow] {message}")
        logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message.

        Args:
            message: Message to log.

        """
        self.console.print(f"[red]✗[/red] {message}")
        logger.error(message)

    def log_success(self, message: str) -> None:
        """Log success message.

        Args:
            message: Message to log.

        """
        self.console.print(f"[green]✓[/green] {message}")
        logger.info(message)

    def display_final_summary(self) -> None:
        """Display final training summary."""
        if not self.metrics_tracker.epoch_metrics:
            return

        # Find best epoch
        val_losses = self.metrics_tracker.get_history("val_loss")
        if val_losses:
            best_idx = val_losses.index(min(val_losses))
            best_metrics = self.metrics_tracker.epoch_metrics[best_idx]

            self.console.print(f"\n[bold]Best Model (Epoch {best_idx + 1})[/bold]")
            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for name, value in best_metrics.items():
                table.add_row(name, f"{value:.4f}")

            self.console.print(table)
