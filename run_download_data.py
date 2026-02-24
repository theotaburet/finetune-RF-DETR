#!/usr/bin/env python3
"""Download human-annotated audio events from EKB API.

This script fetches labeled events from the EKB API across all available sources
(or specified sources) and groups overlapping events into minimal enclosing sounds.
This helps the network learn overlapping events without penalizing it wrongly.

Workflow:
    1. Scan all available sources (or use specified sources)
    2. Fetch all labeled events from those sources
    3. Group events by source file
    4. Within each file, group temporally overlapping events
    5. Download each grouped sound segment
    6. Optionally split into train/val/test sets

Example scenario:
    deepship_0.wav (30s) contains:
    - ship at 0-10s and 15-30s
    - sonar beeps at 4.5-5.5s and 20.5-21.5s
    - humpback whale at 12.5-13s

    Resulting sounds (minimal enclosing overlapping events):
    - Sound 1: ship (0-10s) + sonar beep (4.5-5.5s)
    - Sound 2: humpback whale (12.5-13s)
    - Sound 3: ship (15-30s) + sonar beep (20.5-21.5s)

Usage:
    # Use YAML configuration file
    python run_download_data.py --config config/download.yaml

    # Override specific config values via CLI
    python run_download_data.py --config config/download.yaml --token YOUR_TOKEN

    # Download from all available sources
    python run_download_data.py --api-url https://api.example.com/ekb/api \
        --token YOUR_TOKEN --output-dir data/downloaded

    # Download from specific sources
    python run_download_data.py --api-url https://api.example.com/ekb/api \
        --token YOUR_TOKEN --source deepship,whale,siren --output-dir data/downloaded

    # Create train/val/test splits with test set of 10 files
    python run_download_data.py --config config/download.yaml \
        --split --test-files 10 --train-ratio 0.8 --val-ratio 0.2

    # Create train/val/test splits with specific test files
    python run_download_data.py --config config/download.yaml \
        --split --test-source-files "file1.wav,file2.wav,file3.wav" --train-ratio 0.8 --val-ratio 0.2

    # Dry run to preview what would be downloaded
    python run_download_data.py --config config/download.yaml --dry-run

    # Filter by label hierarchy and date range
    python run_download_data.py --api-url https://api.example.com/ekb/api \
        --token YOUR_TOKEN --label-hierarchy "marine/ship" \
        --from-date 2024-01-01 --to-date 2024-12-31 \
        --output-dir data/downloaded

"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import requests
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
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
# Configuration Classes
# =============================================================================


@dataclass
class APIConfig:
    """API configuration."""

    url: str = ""
    token: str | None = None


@dataclass
class OutputConfig:
    """Output directory configuration."""

    dir: str = "data/downloaded"
    train_dir: str = "train"
    val_dir: str = "valid"
    test_dir: str = "test"
    metadata_dir: str = "metadata"


@dataclass
class SplitConfig:
    """Dataset split configuration."""

    enabled: bool = False
    test_source_files: list[str] = field(default_factory=list)
    test_files: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    seed: int = 42
    stratify_val: bool = True

    def __post_init__(self) -> None:
        """Validate split ratios."""
        if self.enabled:
            total = self.train_ratio + self.val_ratio
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"train_ratio + val_ratio must equal 1.0, got {total:.4f}")
            if self.test_files < 0:
                raise ValueError(f"test_files must be >= 0, got {self.test_files}")


@dataclass
class FilterConfig:
    """Filter configuration."""

    sources: list[str] | None = None
    label_hierarchy: str | None = None
    labeler: str | None = None
    from_date: str | None = None
    to_date: str | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None
    min_duration: float | None = None
    max_frequency: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v
            for k, v in {
                "sources": self.sources,
                "label_hierarchy": self.label_hierarchy,
                "labeler": self.labeler,
                "from_date": self.from_date,
                "to_date": self.to_date,
                "confidence_min": self.confidence_min,
                "confidence_max": self.confidence_max,
                "min_duration": self.min_duration,
                "max_frequency": self.max_frequency,
            }.items()
            if v is not None
        }


@dataclass
class LimitConfig:
    """Download limits configuration."""

    max_labels: int | None = None


@dataclass
class AdvancedConfig:
    """Advanced options configuration."""

    seed: int = 42
    batch_size: int = 100
    max_retries: int = 3
    retry_backoff_factor: float = 1.0


@dataclass
class DownloadConfig:
    """Complete download configuration."""

    api: APIConfig = field(default_factory=APIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    limits: LimitConfig = field(default_factory=LimitConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> DownloadConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        if "api" in data:
            config.api = APIConfig(**data["api"])
        if "output" in data:
            config.output = OutputConfig(**data["output"])
        if "split" in data:
            config.split = SplitConfig(**data["split"])
        if "filters" in data:
            filter_data = data["filters"]
            # Convert sources string to list if needed
            if filter_data.get("sources") and isinstance(filter_data["sources"], str):
                filter_data["sources"] = [s.strip() for s in filter_data["sources"].split(",")]
            config.filters = FilterConfig(**filter_data)
        if "limits" in data:
            config.limits = LimitConfig(**data["limits"])
        if "advanced" in data:
            config.advanced = AdvancedConfig(**data["advanced"])

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api": {
                "url": self.api.url,
                "token": self.api.token,
            },
            "output": {
                "dir": self.output.dir,
                "train_dir": self.output.train_dir,
                "val_dir": self.output.val_dir,
                "test_dir": self.output.test_dir,
                "metadata_dir": self.output.metadata_dir,
            },
            "split": {
                "enabled": self.split.enabled,
                "test_source_files": self.split.test_source_files,
                "test_files": self.split.test_files,
                "train_ratio": self.split.train_ratio,
                "val_ratio": self.split.val_ratio,
                "seed": self.split.seed,
                "stratify_val": self.split.stratify_val,
            },
            "filters": {
                "sources": self.filters.sources,
                "label_hierarchy": self.filters.label_hierarchy,
                "labeler": self.filters.labeler,
                "from_date": self.filters.from_date,
                "to_date": self.filters.to_date,
                "confidence_min": self.filters.confidence_min,
                "confidence_max": self.filters.confidence_max,
                "min_duration": self.filters.min_duration,
                "max_frequency": self.filters.max_frequency,
            },
            "limits": {
                "max_labels": self.limits.max_labels,
            },
            "advanced": {
                "seed": self.advanced.seed,
                "batch_size": self.advanced.batch_size,
                "max_retries": self.advanced.max_retries,
                "retry_backoff_factor": self.advanced.retry_backoff_factor,
            },
        }


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class LabeledEvent:
    """A labeled event from the EKB API.

    Attributes:
        label_id: UUID of the label.
        source_file: Path to the source audio file.
        source_start: Start time in seconds within the source file.
        source_end: End time in seconds within the source file.
        label_hierarchy: Hierarchical label path (e.g., "marine/ship/engine").
        hz_min: Minimum frequency in Hz (optional).
        hz_max: Maximum frequency in Hz (optional).
        confidence: Confidence score of the label.
        labeler: Who created the label.
        absolute_time: Absolute timestamp of the event (optional).

    """

    label_id: str
    source_file: str
    source_start: float
    source_end: float
    label_hierarchy: str
    hz_min: float | None = None
    hz_max: float | None = None
    confidence: float = 1.0
    labeler: str = ""
    absolute_time: str | None = None

    @property
    def duration_s(self) -> float:
        """Event duration in seconds."""
        return self.source_end - self.source_start

    @property
    def duration_ms(self) -> float:
        """Event duration in milliseconds."""
        return self.duration_s * 1000

    def overlaps(self, other: LabeledEvent) -> bool:
        """Check if this event temporally overlaps with another."""
        return self.source_start < other.source_end and self.source_end > other.source_start


@dataclass
class SoundSegment:
    """A sound segment containing one or more overlapping labeled events.

    Attributes:
        events: List of labeled events in this sound segment.
        start_s: Start time in seconds (encompassing all events).
        end_s: End time in seconds (encompassing all events).
        source_file: Path to the source audio file.
        sound_id: Unique identifier for this sound.

    """

    events: list[LabeledEvent] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    source_file: str = ""
    sound_id: str = ""

    @property
    def duration_s(self) -> float:
        """Sound duration in seconds."""
        return self.end_s - self.start_s

    @property
    def duration_ms(self) -> float:
        """Sound duration in milliseconds."""
        return self.duration_s * 1000

    @property
    def label_hierarchies(self) -> list[str]:
        """List of unique label hierarchies in this sound."""
        return list(set(e.label_hierarchy for e in self.events))

    @property
    def primary_label(self) -> str:
        """Primary label for stratification (first label hierarchy)."""
        if self.events:
            return self.events[0].label_hierarchy
        return "unknown"


@dataclass
class DownloadResult:
    """Result of a download operation.

    Attributes:
        sound_id: Unique identifier for the sound.
        source_file: Original source file path.
        output_path: Path where the audio was saved.
        start_s: Start time in source file.
        end_s: End time in source file.
        events: Events contained in this sound.
        success: Whether the download was successful.
        skipped: Whether the download was skipped (file already exists).
        error: Error message if failed.
        split: Which split this sound belongs to (train/val/test).

    """

    sound_id: str
    source_file: str
    output_path: Path | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    events: list[dict] = field(default_factory=list)
    success: bool = False
    skipped: bool = False
    error: str | None = None
    split: str = "unsplit"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sound_id": self.sound_id,
            "source_file": self.source_file,
            "output_path": str(self.output_path) if self.output_path else None,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.end_s - self.start_s,
            "events": self.events,
            "success": self.success,
            "skipped": self.skipped,
            "error": self.error,
            "split": self.split,
        }


# =============================================================================
# API Client
# =============================================================================


class EKBAPIClient:
    """Client for the EKB API with built-in retry logic.

    Implements exponential backoff with jitter for handling transient errors
    during API calls and file downloads.

    Example usage:
        client = EKBAPIClient(
            base_url="https://api.example.com/ekb/api",
            token="your_token",
            max_retries=5,
            retry_backoff_factor=2.0,
        )

        # Retries are automatic
        labels = client.list_labels()
        audio_data = client.get_sound_file(label_id)

    """

    # HTTP status codes that should trigger a retry
    DEFAULT_RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.0,
        retry_on_status_codes: list[int] | None = None,
    ) -> None:
        """Initialize the API client.

        Args:
            base_url: Base URL of the EKB API (e.g., "https://api.example.com/ekb/api").
            token: Bearer token for authentication (optional).
            max_retries: Maximum number of retry attempts (default: 3).
            retry_backoff_factor: Backoff factor for exponential retry delay (default: 1.0).
            retry_on_status_codes: HTTP status codes that should trigger a retry.
                Defaults to [408, 429, 500, 502, 503, 504].

        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

        # Retry configuration
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_on_status_codes = retry_on_status_codes or self.DEFAULT_RETRY_STATUS_CODES

        # Retry statistics
        self.retry_stats = {
            "total_requests": 0,
            "total_retries": 0,
            "successful_retries": 0,
            "failed_after_retries": 0,
        }

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if a request should be retried based on the exception.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number (0-based).

        Returns:
            True if the request should be retried.

        """
        if attempt >= self.max_retries:
            return False

        # Retry on connection errors (timeouts, network issues)
        if isinstance(
            exception,
            requests.exceptions.ConnectionError
            | requests.exceptions.Timeout
            | requests.exceptions.ChunkedEncodingError,
        ):
            return True

        # Retry on specific HTTP status codes
        if isinstance(exception, requests.exceptions.HTTPError):
            if exception.response is not None:
                status_code = exception.response.status_code
                if status_code in self.retry_on_status_codes:
                    return True

        return False

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate the delay before the next retry attempt.

        Uses exponential backoff with jitter to avoid thundering herd.

        Args:
            attempt: Current attempt number (0-based).

        Returns:
            Delay in seconds.

        """
        # Exponential backoff: base_delay * (2 ^ attempt) * backoff_factor
        base_delay = 1.0
        delay = base_delay * (2**attempt) * self.retry_backoff_factor

        # Add jitter (±25% randomization)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        final_delay = max(0, delay + jitter)

        # Cap maximum delay at 60 seconds
        return min(final_delay, 60.0)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a GET request to the API with retry logic."""
        self.retry_stats["total_requests"] += 1
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        last_exception = None
        first_attempt = True

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                result = response.json()

                # If succeeded after retries, update stats
                if not first_attempt:
                    self.retry_stats["successful_retries"] += 1
                    logger.info(f"Request to {endpoint} succeeded after {attempt + 1} attempts")

                return result

            except Exception as e:
                last_exception = e
                first_attempt = False

                if not self._should_retry(e, attempt):
                    if attempt > 0:
                        self.retry_stats["failed_after_retries"] += 1
                    raise

                if attempt < self.max_retries:
                    self.retry_stats["total_retries"] += 1
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Request to {endpoint} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.retry_stats["failed_after_retries"] += 1
                    logger.error(f"All {self.max_retries + 1} attempts failed for {endpoint}. Last error: {e}")
                    raise

        # Should never reach here
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in retry logic")

    def list_sources(self) -> list[str]:
        """List all available sources from the API.

        Returns:
            List of source names.

        """
        response = self._get("/sdk/sources")
        sources = response.get("sources", [])
        return [s.get("name", "") for s in sources if s.get("name")]

    def _get_binary(self, endpoint: str) -> bytes:
        """Make a GET request and return binary content with retry logic."""
        self.retry_stats["total_requests"] += 1
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        last_exception = None
        first_attempt = True

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                content = response.content

                # If succeeded after retries, update stats
                if not first_attempt:
                    self.retry_stats["successful_retries"] += 1
                    logger.info(f"Binary request to {endpoint} succeeded after {attempt + 1} attempts")

                return content

            except Exception as e:
                last_exception = e
                first_attempt = False

                if not self._should_retry(e, attempt):
                    if attempt > 0:
                        self.retry_stats["failed_after_retries"] += 1
                    raise

                if attempt < self.max_retries:
                    self.retry_stats["total_retries"] += 1
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Binary request to {endpoint} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.retry_stats["failed_after_retries"] += 1
                    logger.error(f"All {self.max_retries + 1} attempts failed for {endpoint}. Last error: {e}")
                    raise

        # Should never reach here
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in retry logic")

    def get_retry_stats(self) -> dict[str, int]:
        """Get retry statistics.

        Returns:
            Dictionary containing:
            - total_requests: Total number of requests made
            - total_retries: Total number of retry attempts
            - successful_retries: Requests that succeeded after retry
            - failed_after_retries: Requests that failed after all retries

        """
        return self.retry_stats.copy()

    def reset_retry_stats(self) -> None:
        """Reset retry statistics."""
        self.retry_stats = {
            "total_requests": 0,
            "total_retries": 0,
            "successful_retries": 0,
            "failed_after_retries": 0,
        }

    def list_labels(
        self,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        label_hierarchy: str | None = None,
        sources: list[str] | None = None,
        licence: str | None = None,
        location: str | None = None,
        labeler: str | None = None,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        min_duration: float | None = None,
        max_frequency: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict:
        """List labels with filtering options.

        Args:
            limit: Maximum number of labels to return.
            offset: Number of labels to skip.
            search: Search term for multiple fields.
            label_hierarchy: Filter by label hierarchy (partial match).
            sources: List of sources to filter by.
            licence: Filter by licence.
            location: Filter by location.
            labeler: Filter by labeler.
            confidence_min: Minimum confidence score.
            confidence_max: Maximum confidence score.
            min_duration: Minimum duration in milliseconds.
            max_frequency: Maximum frequency in Hz.
            from_date: Filter labels created from this date.
            to_date: Filter labels created up to this date.

        Returns:
            API response containing labels list.

        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if search:
            params["search"] = search
        if label_hierarchy:
            params["label_hierarchy"] = label_hierarchy
        if sources:
            params["sources"] = ",".join(sources)
        if licence:
            params["licence"] = licence
        if location:
            params["location"] = location
        if labeler:
            params["labeler"] = labeler
        if confidence_min is not None:
            params["confidence_min"] = confidence_min
        if confidence_max is not None:
            params["confidence_max"] = confidence_max
        if min_duration is not None:
            params["min_duration"] = min_duration
        if max_frequency is not None:
            params["max_frequency"] = max_frequency
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return self._get("/sdk/labels", params=params)

    def get_label(self, label_id: str) -> dict:
        """Get a specific label by ID.

        Args:
            label_id: UUID of the label.

        Returns:
            Label details.

        """
        return self._get(f"/sdk/labels/{label_id}")

    def get_label_audio(self, label_id: str) -> bytes:
        """Get the audio data for a specific label.

        Args:
            label_id: UUID of the label.

        Returns:
            Audio file content as bytes.

        """
        return self._get_binary(f"/sdk/labels/{label_id}/data")

    def get_sound_file(self, annotation_id: str) -> bytes:
        """Get the audio file for a specific annotation.

        Args:
            annotation_id: UUID of the annotation/label.

        Returns:
            Audio file content as bytes.

        """
        return self._get_binary(f"/sound/{annotation_id}")

    def fetch_all_labels(
        self,
        batch_size: int = 100,
        quiet: bool = False,
        **filters: Any,
    ) -> Iterator[dict]:
        """Fetch all labels with pagination.

        Args:
            batch_size: Number of labels per request.
            quiet: If True, suppress progress output.
            **filters: Additional filters to pass to list_labels.

        Yields:
            Label dictionaries.

        """
        offset = 0
        total = None
        last_reported = 0

        while total is None or offset < total:
            response = self.list_labels(limit=batch_size, offset=offset, **filters)
            labels = response.get("labels", [])
            total = response.get("total", 0)

            if not labels:
                break

            yield from labels

            offset += len(labels)

            # Only print progress at 25%, 50%, 75%, 100% or if total < 500
            if not quiet and total > 0:
                progress_pct = (offset / total) * 100
                if total < 500 or offset == total or (progress_pct - last_reported) >= 25:
                    console.print(f"  [dim]Fetched {offset}/{total} labels[/dim]")
                    last_reported = progress_pct

    def discover_class_names(self, **filters: Any) -> list[str]:
        """Discover distinct class names from the API by scanning label hierarchies.

        Fetches all labels (applying any provided filters), extracts the leaf name
        from each ``label_hierarchy`` field, and returns a sorted deduplicated list.

        Args:
            **filters: Filters forwarded to :meth:`fetch_all_labels`
                (e.g. ``sources=["src1"]``, ``label_hierarchy="marine"``).

        Returns:
            Sorted list of unique leaf class names.

        """
        class_names: set[str] = set()
        for label in self.fetch_all_labels(quiet=True, **filters):
            hierarchy = label.get("label_hierarchy", "")
            leaf = extract_leaf_name(hierarchy)
            if leaf:
                class_names.add(leaf)
        return sorted(class_names)


def extract_leaf_name(hierarchy: str) -> str:
    """Extract the leaf (most-specific) name from a label hierarchy string.

    Supports both ``" + "`` and ``" > "`` separators. Returns the last
    segment, stripped of whitespace.

    Args:
        hierarchy: Hierarchy string, e.g. ``"Marine mammals + Whales + Humpback whale"``.

    Returns:
        Leaf name (e.g. ``"Humpback whale"``), or the original string stripped
        if no separator is found. Returns empty string for blank input.

    """
    if not hierarchy or not isinstance(hierarchy, str):
        return ""
    if " + " in hierarchy:
        return hierarchy.split(" + ")[-1].strip()
    if " > " in hierarchy:
        return hierarchy.split(" > ")[-1].strip()
    return hierarchy.strip()


def discover_class_names_from_metadata(metadata_path: Path) -> list[str]:
    """Discover class names from a previously-downloaded metadata JSON file.

    Scans the ``sounds`` entries for event ``label_hierarchy`` fields and
    extracts unique leaf names.

    Args:
        metadata_path: Path to ``download_metadata.json``.

    Returns:
        Sorted list of unique leaf class names.

    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    class_names: set[str] = set()
    for sound in metadata.get("sounds", []):
        for event in sound.get("events", []):
            hierarchy = event.get("label_hierarchy", "")
            leaf = extract_leaf_name(hierarchy)
            if leaf:
                class_names.add(leaf)
    return sorted(class_names)


# =============================================================================
# Event Grouping Logic
# =============================================================================


def group_overlapping_events(
    events: list[LabeledEvent],
    source_file: str,
    sound_prefix: str = "sound",
) -> list[SoundSegment]:
    """Group overlapping events into minimal enclosing sounds.

    Uses an interval merging algorithm to combine events that overlap temporally.
    Each resulting sound contains all overlapping events from the same source file.

    Args:
        events: List of labeled events (all from the same source file).
        source_file: Path to the source audio file.
        sound_prefix: Prefix for sound IDs.

    Returns:
        List of sound segments with grouped events.

    Example:
        >>> events = [
        ...     LabeledEvent("1", "file.wav", 0, 10, "ship"),
        ...     LabeledEvent("2", "file.wav", 4.5, 5.5, "sonar"),
        ...     LabeledEvent("3", "file.wav", 12.5, 13, "whale"),
        ... ]
        >>> sounds = group_overlapping_events(events, "file.wav")
        >>> len(sounds)
        2
        >>> sounds[0].label_hierarchies
        ["ship", "sonar"]

    """
    if not events:
        return []

    # Sort events by start time
    sorted_events = sorted(events, key=lambda e: e.source_start)

    sounds: list[SoundSegment] = []
    current_sound: SoundSegment | None = None

    for event in sorted_events:
        if current_sound is None:
            # Start first sound
            current_sound = SoundSegment(
                events=[event],
                start_s=event.source_start,
                end_s=event.source_end,
                source_file=source_file,
                sound_id=f"{sound_prefix}_{len(sounds):04d}",
            )
        elif event.source_start <= current_sound.end_s:
            # Event overlaps with current sound, extend it
            current_sound.events.append(event)
            current_sound.end_s = max(current_sound.end_s, event.source_end)
        else:
            # No overlap, finalize current sound and start new one
            sounds.append(current_sound)
            current_sound = SoundSegment(
                events=[event],
                start_s=event.source_start,
                end_s=event.source_end,
                source_file=source_file,
                sound_id=f"{sound_prefix}_{len(sounds):04d}",
            )

    # Don't forget the last sound
    if current_sound is not None:
        sounds.append(current_sound)

    return sounds


def parse_label_response(label: dict) -> LabeledEvent:
    """Parse an API label response into a LabeledEvent.

    Args:
        label: Label dictionary from the API.

    Returns:
        Parsed LabeledEvent.

    """
    return LabeledEvent(
        label_id=label["uuid"],
        source_file=label.get("source_path", ""),
        source_start=label.get("source_start", 0.0),
        source_end=label.get("source_end", 0.0),
        label_hierarchy=label.get("label_hierarchy", "unknown"),
        hz_min=label.get("hz_min"),
        hz_max=label.get("hz_max"),
        confidence=label.get("confidence", 1.0),
        labeler=label.get("labeler", ""),
        absolute_time=label.get("absolute_time"),
    )


# =============================================================================
# Dataset Splitting Logic
# =============================================================================


def split_sounds_into_datasets(
    sounds: list[SoundSegment],
    config: SplitConfig,
    dry_run: bool = False,
) -> dict[str, list[SoundSegment]]:
    """Split sounds into train/val/test sets.

    Reserves test_files for test set first, then splits remaining into
    train/val according to train_ratio/val_ratio. Attempts to stratify
    validation set by class if stratify_val is True.

    If test_source_files is specified, those exact source files are used
    for the test set instead of randomly selecting.

    Args:
        sounds: List of all sound segments.
        config: Split configuration.
        dry_run: If True, only log what would be done.

    Returns:
        Dictionary mapping split names to lists of sounds.

    """
    if not sounds:
        return {"train": [], "val": [], "test": []}

    rng = np.random.default_rng(config.seed)

    console.print("\n[cyan]Dataset Splitting:[/cyan]")
    console.print(f"  Total sounds: {len(sounds)}")

    # Check if specific test source files are specified
    if config.test_source_files:
        # Find sounds from specified source files
        test_indices = set()
        remaining_indices = []

        # Pre-compute filename-only versions of test_source_files for matching
        test_source_filenames = {Path(f).name for f in config.test_source_files}
        test_source_paths = set(config.test_source_files)

        # Collect all unique source files for debugging
        all_source_files = {sound.source_file for sound in sounds}

        for i, sound in enumerate(sounds):
            source_filename = Path(sound.source_file).name
            source_path = sound.source_file
            # Match against either the filename OR the full path
            if source_filename in test_source_filenames or source_path in test_source_paths:
                test_indices.add(i)
            else:
                remaining_indices.append(i)

        remaining_indices = np.array(remaining_indices)

        found_test_files = len(test_indices)
        matched_source_files = {sounds[i].source_file for i in test_indices}

        console.print(f"  Test set: {found_test_files} sounds from {len(matched_source_files)} source file(s)")
        if matched_source_files:
            for f in matched_source_files:
                console.print(f"    [green]✓[/green] {f}")
        else:
            console.print("    [yellow]No source files matched![/yellow]")
            console.print(f"    [dim]Available: {all_source_files}[/dim]")

        # Warn about missing files
        found_source_filenames = {Path(f).name for f in matched_source_files}
        missing_files = set()
        for test_file in config.test_source_files:
            test_filename = Path(test_file).name
            if test_file not in matched_source_files and test_filename not in found_source_filenames:
                missing_files.add(test_file)
        if missing_files:
            console.print(f"  [yellow]Warning: Could not find: {missing_files}[/yellow]")
    else:
        # Shuffle sounds for random selection
        sound_indices = np.arange(len(sounds))
        rng.shuffle(sound_indices)

        # Reserve test files
        test_count = min(config.test_files, len(sounds))
        test_indices = set(sound_indices[:test_count].tolist())
        remaining_indices = sound_indices[test_count:]

        console.print(f"  Test set: {test_count} sounds (randomly selected)")

    # Split remaining into train/val
    if len(remaining_indices) > 0:
        n_remaining = len(remaining_indices)
        n_train = int(n_remaining * config.train_ratio)

        if config.stratify_val and n_remaining > 10:
            # Attempt stratified split
            train_indices, val_indices = _stratified_split(sounds, remaining_indices, n_train, rng, dry_run)
        else:
            # Random split
            train_indices = set(remaining_indices[:n_train].tolist())
            val_indices = set(remaining_indices[n_train:].tolist())
    else:
        train_indices = set()
        val_indices = set()

    # Build result
    splits = {
        "train": [sounds[i] for i in train_indices],
        "val": [sounds[i] for i in val_indices],
        "test": [sounds[i] for i in test_indices],
    }

    console.print(f"  Train set: {len(splits['train'])} sounds ({config.train_ratio:.0%})")
    console.print(f"  Val set: {len(splits['val'])} sounds ({config.val_ratio:.0%})")
    console.print(f"  Test set: {len(splits['test'])} sounds")

    # Verify no overlap between test and train/val source files
    train_sources = {s.source_file for s in splits["train"]}
    val_sources = {s.source_file for s in splits["val"]}
    test_sources = {s.source_file for s in splits["test"]}

    train_test_overlap = train_sources & test_sources
    val_test_overlap = val_sources & test_sources

    if train_test_overlap or val_test_overlap:
        if train_test_overlap:
            console.print(f"  [red]ERROR: Train/Test overlap: {train_test_overlap}[/red]")
        if val_test_overlap:
            console.print(f"  [red]ERROR: Val/Test overlap: {val_test_overlap}[/red]")
    else:
        console.print("  [green]✓ No source file overlap between test and train/val[/green]")

    # Print class distribution in dry run
    if dry_run:
        _print_class_distribution(splits)

    return splits


def _stratified_split(
    sounds: list[SoundSegment],
    indices: np.ndarray,
    n_train: int,
    rng: np.random.Generator,
    dry_run: bool,
) -> tuple[set[int], set[int]]:
    """Perform stratified train/val split with balanced validation.

    Ensures the validation set has roughly equal samples per class,
    regardless of class frequency in the overall dataset.

    Args:
        sounds: List of all sound segments.
        indices: Indices to split.
        n_train: Number of training samples.
        rng: Random number generator.
        dry_run: If True, only log what would be done.

    Returns:
        Tuple of (train_indices, val_indices) as sets.

    """
    # Group by primary label
    label_to_indices: dict[str, list[int]] = {}
    for idx in indices:
        label = sounds[idx].primary_label
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    n_val_target = len(indices) - n_train
    n_classes = len(label_to_indices)

    if n_classes == 0:
        return set(), set()

    # Target equal samples per class in val set
    target_per_class = max(1, n_val_target // n_classes)

    val_indices: list[int] = []
    train_indices: list[int] = []

    # For each class, take target_per_class for val (capped at available - 1 to keep at least 1 in train)
    for label, class_indices in label_to_indices.items():
        rng.shuffle(class_indices)
        n_available = len(class_indices)
        n_val_for_class = min(target_per_class, max(1, n_available - 1))
        val_indices.extend(class_indices[:n_val_for_class])
        train_indices.extend(class_indices[n_val_for_class:])

    if dry_run:
        console.print(f"  [dim]Balanced val: ~{target_per_class} samples/class across {n_classes} classes[/dim]")

    return set(train_indices), set(val_indices)


def _print_class_distribution(splits: dict[str, list[SoundSegment]]) -> None:
    """Print class distribution for each split in a compact table format."""
    # Collect all unique labels across all splits
    all_labels: set[str] = set()
    split_counts: dict[str, dict[str, int]] = {}

    for split_name, split_sounds in splits.items():
        if not split_sounds:
            continue
        split_counts[split_name] = {}
        for sound in split_sounds:
            label = sound.primary_label
            split_counts[split_name][label] = split_counts[split_name].get(label, 0) + 1
            all_labels.add(label)

    if not all_labels:
        return

    # Print compact table
    console.print("\n  [cyan]Class distribution:[/cyan]")

    # Header
    header = "    {:40s}".format("Label")
    for split_name in ["train", "val", "test"]:
        if split_name in split_counts:
            header += f" {split_name.upper():>6s}"
    console.print(f"  [dim]{header}[/dim]")

    # Rows
    for label in sorted(all_labels):
        row = f"    {label[:40]:40s}"
        for split_name in ["train", "val", "test"]:
            if split_name in split_counts:
                count = split_counts[split_name].get(label, 0)
                row += f" {count:>6d}"
        console.print(row)


# =============================================================================
# Download Logic
# =============================================================================


class DataDownloader:
    """Downloads labeled audio data from EKB API."""

    def __init__(
        self,
        api_client: EKBAPIClient,
        config: DownloadConfig,
        dry_run: bool = False,
    ) -> None:
        """Initialize the downloader.

        Args:
            api_client: Configured EKB API client.
            config: Download configuration.
            dry_run: If True, only report what would be downloaded.

        """
        self.api_client = api_client
        self.config = config
        self.dry_run = dry_run
        self.rng = np.random.default_rng(config.advanced.seed)

        # Create output directories
        if not dry_run:
            self._create_output_directories()

    def _create_output_directories(self) -> None:
        """Create output directory structure."""
        base_dir = Path(self.config.output.dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        if self.config.split.enabled:
            # Create split directories
            (base_dir / self.config.output.train_dir).mkdir(exist_ok=True)
            (base_dir / self.config.output.val_dir).mkdir(exist_ok=True)
            (base_dir / self.config.output.test_dir).mkdir(exist_ok=True)
        else:
            # Create single audio directory
            (base_dir / "audio").mkdir(exist_ok=True)

        # Create metadata directory
        (base_dir / self.config.output.metadata_dir).mkdir(exist_ok=True)

    def download_sound(
        self,
        sound: SoundSegment,
        source_label_id: str,
        file_counter: int,
        split: str = "unsplit",
    ) -> DownloadResult:
        """Download audio for a sound segment.

        Args:
            sound: Sound segment to download.
            source_label_id: Label ID to use for fetching audio.
            file_counter: Counter for generating unique filenames.
            split: Which split this belongs to (train/val/test/unsplit).

        Returns:
            DownloadResult with status and metadata.

        """
        # Generate output filename
        source_name = Path(sound.source_file).stem
        output_filename = f"{source_name}_{sound.sound_id}.wav"

        # Determine output directory based on split
        if self.config.split.enabled:
            split_dir_map = {
                "train": self.config.output.train_dir,
                "val": self.config.output.val_dir,
                "test": self.config.output.test_dir,
            }
            split_dir = split_dir_map.get(split, "audio")
            output_path = Path(self.config.output.dir) / split_dir / output_filename
        else:
            output_path = Path(self.config.output.dir) / "audio" / output_filename

        # Prepare event metadata
        events_meta = [
            {
                "label_id": e.label_id,
                "label_hierarchy": e.label_hierarchy,
                "start_s": e.source_start,
                "end_s": e.source_end,
                "duration_s": e.duration_s,
                "hz_min": e.hz_min,
                "hz_max": e.hz_max,
                "confidence": e.confidence,
                "labeler": e.labeler,
            }
            for e in sound.events
        ]

        result = DownloadResult(
            sound_id=sound.sound_id,
            source_file=sound.source_file,
            output_path=output_path,
            start_s=sound.start_s,
            end_s=sound.end_s,
            events=events_meta,
            split=split,
        )

        if self.dry_run:
            result.success = True
            return result

        # Skip download if both .wav and sidecar .json already exist
        sidecar_path = output_path.with_suffix(".json")
        if output_path.exists() and sidecar_path.exists():
            result.success = True
            result.skipped = True
            logger.debug(f"Skipping (already exists): {output_filename}")
            return result

        try:
            # Download the audio file
            audio_data = self.api_client.get_sound_file(source_label_id)

            # Save audio file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_data)

            # Save sidecar metadata JSON alongside the .wav file
            # Each downloaded clip IS the sound segment, so events are file-level.
            # We don't store absolute times because the API may return a clip
            # shorter than the annotated segment; the chunker will use the actual
            # audio duration to create bboxes spanning each chunk.
            sidecar_path = output_path.with_suffix(".json")
            sidecar_events = []
            for evt in events_meta:
                sidecar_events.append(
                    {
                        "label_hierarchy": evt.get("label_hierarchy", ""),
                        "hz_min": evt.get("hz_min") or 0,
                        "hz_max": evt.get("hz_max") or 0,
                        "confidence": evt.get("confidence"),
                        "labeler": evt.get("labeler"),
                        "is_file_level": True,
                    }
                )
            sidecar_data = {
                "uuid": sound.sound_id,
                "source_file": sound.source_file,
                "events": sidecar_events,
            }
            with open(sidecar_path, "w") as f:
                json.dump(sidecar_data, f, indent=2)

            result.success = True
            logger.debug(f"Downloaded: {output_filename} to {split}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.warning(f"Failed to download {sound.sound_id}: {e}")

        return result

    def process_source_file(
        self,
        source_file: str,
        events: list[LabeledEvent],
        split: str = "unsplit",
    ) -> list[DownloadResult]:
        """Process all events from a single source file.

        Args:
            source_file: Path to the source audio file.
            events: List of labeled events from this file.
            split: Which split this belongs to.

        Returns:
            List of download results.

        """
        console.print(f"\n[cyan]Processing: {source_file}[/cyan] [{split}]")
        console.print(f"  [dim]{len(events)} events found[/dim]")

        # Group overlapping events into sounds
        sounds = group_overlapping_events(events, source_file)
        console.print(f"  [dim]Grouped into {len(sounds)} sound segments[/dim]")

        # Show grouping details
        for sound in sounds:
            labels = ", ".join(sorted(sound.label_hierarchies))
            console.print(f"    [dim]{sound.sound_id}: {sound.start_s:.1f}s - {sound.end_s:.1f}s ({labels})[/dim]")

        if self.dry_run:
            # Return mock results for dry run
            return [
                DownloadResult(
                    sound_id=s.sound_id,
                    source_file=source_file,
                    start_s=s.start_s,
                    end_s=s.end_s,
                    events=[
                        {
                            "label_id": e.label_id,
                            "label_hierarchy": e.label_hierarchy,
                            "start_s": e.source_start,
                            "end_s": e.source_end,
                        }
                        for e in s.events
                    ],
                    success=True,
                    split=split,
                )
                for s in sounds
            ]

        # Download each sound
        results: list[DownloadResult] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Downloading sounds...", total=len(sounds))

            for idx, sound in enumerate(sounds):
                source_label_id = sound.events[0].label_id
                result = self.download_sound(sound, source_label_id, idx, split)
                results.append(result)
                progress.update(task, advance=1)

        return results

    def save_metadata(
        self,
        all_results: list[DownloadResult],
        split_results: dict[str, list[DownloadResult]] | None = None,
    ) -> Path:
        """Save download metadata to JSON file.

        Args:
            all_results: List of all download results.
            split_results: Optional dictionary of split-specific results.

        Returns:
            Path to the metadata file.

        """
        metadata = {
            "download_info": {
                "total_sounds": len(all_results),
                "successful": sum(1 for r in all_results if r.success),
                "failed": sum(1 for r in all_results if not r.success),
                "config": self.config.to_dict(),
                "dry_run": self.dry_run,
            },
            "sounds": [r.to_dict() for r in all_results],
        }

        if split_results:
            metadata["splits"] = {
                split: {
                    "count": len(results),
                    "successful": sum(1 for r in results if r.success),
                    "failed": sum(1 for r in results if not r.success),
                }
                for split, results in split_results.items()
            }

        metadata_path = Path(self.config.output.dir) / self.config.output.metadata_dir / "download_metadata.json"

        if not self.dry_run:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return metadata_path

    def run(self) -> list[DownloadResult]:
        """Run the download process.

        Scans all available sources (or specified sources), groups events by
        source file, groups overlapping events within each file, and downloads
        the resulting sound segments. Optionally splits into train/val/test.

        When test_source_files is specified, those files are fetched first
        (regardless of other filters) to ensure they are included in the test set.

        Returns:
            List of download results.

        """
        console.print("\n[bold cyan]EKB Data Downloader[/bold cyan]")

        if self.dry_run:
            console.print("[yellow]DRY RUN MODE - No files will be downloaded[/yellow]")

        # Build filters for train/val data
        filters = {}
        if self.config.filters.sources:
            filters["sources"] = self.config.filters.sources
        if self.config.filters.label_hierarchy:
            filters["label_hierarchy"] = self.config.filters.label_hierarchy
        if self.config.filters.labeler:
            filters["labeler"] = self.config.filters.labeler
        if self.config.filters.from_date:
            filters["from_date"] = self.config.filters.from_date
        if self.config.filters.to_date:
            filters["to_date"] = self.config.filters.to_date
        if self.config.filters.confidence_min is not None:
            filters["confidence_min"] = self.config.filters.confidence_min
        if self.config.filters.confidence_max is not None:
            filters["confidence_max"] = self.config.filters.confidence_max
        if self.config.filters.min_duration is not None:
            filters["min_duration"] = self.config.filters.min_duration
        if self.config.filters.max_frequency is not None:
            filters["max_frequency"] = self.config.filters.max_frequency

        console.print("\n[cyan]Filters:[/cyan]")
        for key, value in filters.items():
            console.print(f"  {key}: {value}")
        console.print(f"  seed: {self.config.advanced.seed}")

        # Track labels separately for test vs train/val
        test_labels: list[dict] = []
        trainval_labels: list[dict] = []

        # Pre-compute test source file matching sets
        test_source_filenames = set()
        test_source_paths = set()
        if self.config.split.enabled and self.config.split.test_source_files:
            test_source_filenames = {Path(f).name for f in self.config.split.test_source_files}
            test_source_paths = set(self.config.split.test_source_files)

        # Step 1: Fetch labels for test_source_files first (if specified)
        if self.config.split.enabled and self.config.split.test_source_files:
            console.print("\n[cyan]Fetching labels for TEST source files...[/cyan]")
            console.print(f"  [dim]Looking for: {self.config.split.test_source_files}[/dim]")

            for test_file in self.config.split.test_source_files:
                # Use the filename for search (API search parameter)
                test_filename = Path(test_file).name
                console.print(f"  [dim]Searching for labels with source file: {test_filename}[/dim]")

                # Fetch labels matching this test file using search
                file_labels: list[dict] = []
                for label in self.api_client.fetch_all_labels(
                    batch_size=self.config.advanced.batch_size,
                    search=test_filename,
                    **{k: v for k, v in filters.items() if k != "sources"},  # Keep other filters except sources
                ):
                    # Verify the label is actually from this source file
                    source_path = label.get("source_path", "")
                    source_filename = Path(source_path).name
                    if source_filename == test_filename or source_path == test_file:
                        file_labels.append(label)

                console.print(f"    [green]Found {len(file_labels)} labels for {test_filename}[/green]")
                test_labels.extend(file_labels)

            if not test_labels:
                console.print("[yellow]Warning: No labels found for specified test source files[/yellow]")
            else:
                console.print(f"[green]Total TEST labels: {len(test_labels)}[/green]")

        # Step 2: Fetch labels for train/val (excluding test source files)
        console.print("\n[cyan]Fetching labels for TRAIN/VAL...[/cyan]")
        fetched_count = 0
        for label in self.api_client.fetch_all_labels(batch_size=self.config.advanced.batch_size, **filters):
            # Skip labels from test source files
            source_path = label.get("source_path", "")
            source_filename = Path(source_path).name

            if source_filename in test_source_filenames or source_path in test_source_paths:
                logger.debug(f"  Skipping label from test source file: {source_path}")
                continue

            trainval_labels.append(label)
            fetched_count += 1

            if self.config.limits.max_labels and fetched_count >= self.config.limits.max_labels:
                console.print(f"  [dim]Reached max_labels limit ({self.config.limits.max_labels})[/dim]")
                break

        console.print(f"[green]Total TRAIN/VAL labels: {len(trainval_labels)}[/green]")

        # Combine all labels
        all_labels = test_labels + trainval_labels

        if not all_labels:
            console.print("[yellow]No labels found matching the criteria[/yellow]")
            return []

        console.print(
            f"\n[green]Total labels: {len(all_labels)} "
            f"(test={len(test_labels)}, train/val={len(trainval_labels)})[/green]"
        )

        # Parse labels into events
        events = [parse_label_response(label) for label in all_labels]

        # Group events by source file
        events_by_source: dict[str, list[LabeledEvent]] = {}
        for event in events:
            source = event.source_file
            if source not in events_by_source:
                events_by_source[source] = []
            events_by_source[source].append(event)

        console.print(f"\n[cyan]Grouped into {len(events_by_source)} source files[/cyan]")

        # Collect all sounds first (before splitting)
        all_sounds: list[SoundSegment] = []
        for source_file, source_events in events_by_source.items():
            sounds = group_overlapping_events(source_events, source_file)
            all_sounds.extend(sounds)

        console.print(f"[green]Total sounds after grouping: {len(all_sounds)}[/green]")

        # Split sounds into datasets if enabled
        split_sounds: dict[str, list[SoundSegment]] = {"unsplit": all_sounds}
        if self.config.split.enabled:
            split_sounds = split_sounds_into_datasets(all_sounds, self.config.split, self.dry_run)

        # Download sounds for each split
        all_results: list[DownloadResult] = []
        split_results: dict[str, list[DownloadResult]] = {}

        for split_name, sounds in split_sounds.items():
            if not sounds:
                continue

            split_results[split_name] = []

            # Group sounds by source file for this split
            sounds_by_source: dict[str, list[SoundSegment]] = {}
            for sound in sounds:
                if sound.source_file not in sounds_by_source:
                    sounds_by_source[sound.source_file] = []
                sounds_by_source[sound.source_file].append(sound)

            console.print(f"\n[cyan]Processing {split_name} split ({len(sounds)} sounds)...[/cyan]")

            # Process each source file
            for source_file, source_sounds in sounds_by_source.items():
                # Get events for these sounds
                source_events = []
                for sound in source_sounds:
                    source_events.extend(sound.events)

                results = self.process_source_file(source_file, source_events, split_name)
                all_results.extend(results)
                split_results[split_name].extend(results)

        # Save metadata
        metadata_path = self.save_metadata(all_results, split_results if self.config.split.enabled else None)
        console.print(f"\n[green]Metadata saved to: {metadata_path}[/green]")

        # Print summary
        total_events_before_grouping = len(events)
        self._print_summary(all_results, split_results, total_events_before_grouping)

        return all_results

    def _print_summary(
        self,
        results: list[DownloadResult],
        split_results: dict[str, list[DownloadResult]],
        total_events_before_grouping: int = 0,
    ) -> None:
        """Print download summary table.

        Args:
            results: List of download results.
            split_results: Dictionary of results by split.
            total_events_before_grouping: Total events before grouping.

        """
        console.print("\n[bold cyan]Download Summary[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        total_sounds = len(results)
        successful = sum(1 for r in results if r.success)
        skipped = sum(1 for r in results if r.skipped)
        failed = total_sounds - successful

        # Event statistics
        total_events_in_sounds = sum(len(r.events) for r in results)
        avg_events_per_sound = total_events_in_sounds / total_sounds if total_sounds > 0 else 0

        # Savings calculation
        if total_events_before_grouping > 0:
            savings = total_events_before_grouping - total_sounds
            savings_pct = (savings / total_events_before_grouping) * 100
        else:
            savings = 0
            savings_pct = 0.0

        table.add_row("Total Events (before grouping)", str(total_events_before_grouping))
        table.add_row("Total Sounds (after grouping)", str(total_sounds))
        table.add_row("Successful", f"[green]{successful}[/green]")
        if skipped > 0:
            table.add_row("Skipped (already exist)", f"[yellow]{skipped}[/yellow]")
        table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
        table.add_row("", "")
        table.add_row("Events contained in sounds", str(total_events_in_sounds))
        table.add_row("Avg Events per Sound", f"{avg_events_per_sound:.2f}")
        table.add_row("", "")
        table.add_row("[bold green]Files Saved by Grouping", f"[bold green]{savings}[/bold green]")
        table.add_row("[bold green]Storage Savings", f"[bold green]{savings_pct:.1f}%[/bold green]")

        # Split breakdown
        if self.config.split.enabled and split_results:
            table.add_row("", "")
            for split_name, split_res in split_results.items():
                split_success = sum(1 for r in split_res if r.success)
                table.add_row(f"{split_name.upper()} set", f"{split_success}/{len(split_res)}")

        # Retry statistics
        retry_stats = self.api_client.get_retry_stats()
        if retry_stats["total_retries"] > 0:
            table.add_row("", "")
            table.add_row("[yellow]Network Retries", f"[yellow]{retry_stats['total_retries']}")
            table.add_row("[green]Successful after retry", f"[green]{retry_stats['successful_retries']}")
            if retry_stats["failed_after_retries"] > 0:
                table.add_row("[red]Failed after all retries", f"[red]{retry_stats['failed_after_retries']}")

        console.print(table)

        # Label hierarchy distribution
        hierarchy_counts: dict[str, int] = {}
        for r in results:
            for event in r.events:
                hierarchy = event.get("label_hierarchy", "unknown")
                hierarchy_counts[hierarchy] = hierarchy_counts.get(hierarchy, 0) + 1

        if hierarchy_counts:
            console.print("\n[cyan]Label Distribution:[/cyan]")
            for hierarchy, count in sorted(hierarchy_counts.items(), key=lambda x: x[1], reverse=True):
                console.print(f"  {hierarchy}: {count}")


# =============================================================================
# CLI
# =============================================================================


def merge_config_with_args(config: DownloadConfig, args: argparse.Namespace) -> DownloadConfig:
    """Merge command-line arguments with YAML config.

    CLI arguments take precedence over YAML config values.

    Args:
        config: Configuration loaded from YAML.
        args: Command-line arguments.

    Returns:
        Merged configuration.

    """
    # API settings
    if args.api_url:
        config.api.url = args.api_url
    if args.token:
        config.api.token = args.token

    # Output settings
    if args.output_dir:
        config.output.dir = args.output_dir

    # Split settings
    if args.split is not None:
        config.split.enabled = args.split
    if args.test_source_files:
        config.split.test_source_files = [s.strip() for s in args.test_source_files.split(",")]
    if args.test_files is not None:
        config.split.test_files = args.test_files
    if args.train_ratio is not None:
        config.split.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        config.split.val_ratio = args.val_ratio
    if args.seed is not None:
        config.split.seed = args.seed
        config.advanced.seed = args.seed

    # Filter settings
    if args.source:
        config.filters.sources = [s.strip() for s in args.source.split(",")]
    if args.label_hierarchy:
        config.filters.label_hierarchy = args.label_hierarchy
    if args.labeler:
        config.filters.labeler = args.labeler
    if args.from_date:
        config.filters.from_date = args.from_date
    if args.to_date:
        config.filters.to_date = args.to_date
    if args.confidence_min is not None:
        config.filters.confidence_min = args.confidence_min
    if args.confidence_max is not None:
        config.filters.confidence_max = args.confidence_max
    if args.min_duration is not None:
        config.filters.min_duration = args.min_duration
    if args.max_frequency is not None:
        config.filters.max_frequency = args.max_frequency

    # Limit settings
    if args.max_labels is not None:
        config.limits.max_labels = args.max_labels

    # Advanced/retry settings
    if hasattr(args, "max_retries") and args.max_retries is not None:
        config.advanced.max_retries = args.max_retries
    if hasattr(args, "retry_backoff") and args.retry_backoff is not None:
        config.advanced.retry_backoff_factor = args.retry_backoff

    return config


def validate_config(config: DownloadConfig) -> list[str]:
    """Validate the configuration.

    Args:
        config: Configuration to validate.

    Returns:
        List of error messages (empty if valid).

    """
    errors = []

    if not config.api.url:
        errors.append("API URL is required (--api-url or config.api.url)")

    if not config.api.token:
        errors.append("API token is required (--token or config.api.token)")

    if config.split.enabled:
        total_ratio = config.split.train_ratio + config.split.val_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            errors.append(f"train_ratio + val_ratio must equal 1.0, got {total_ratio:.4f}")

        if config.split.test_files < 0:
            errors.append(f"test_files must be >= 0, got {config.split.test_files}")

    return errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download human-annotated audio events from EKB API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML configuration file (recommended)
  python run_download_data.py --config config/download.yaml

  # Override specific values from CLI
  python run_download_data.py --config config/download.yaml --token YOUR_TOKEN

  # Create train/val/test splits with random test files
  python run_download_data.py --config config/download.yaml --split --test-files 10

  # Create train/val/test splits with specific test files
  python run_download_data.py --config config/download.yaml --split \
      --test-source-files "file1.wav,file2.wav,file3.wav"

  # Dry run to preview what would be downloaded
  python run_download_data.py --config config/download.yaml --dry-run

  # Use CLI arguments only (legacy mode)
  python run_download_data.py --api-url https://api.example.com/ekb/api \\
      --token YOUR_TOKEN --output-dir data/downloaded

  # Download from specific sources
  python run_download_data.py --api-url https://api.example.com/ekb/api \\
      --token YOUR_TOKEN --source deepship,whale,siren --output-dir data/downloaded

  # Filter by label hierarchy
  python run_download_data.py --api-url https://api.example.com/ekb/api \\
      --token YOUR_TOKEN --label-hierarchy "marine/ship" --output-dir data/ships
        """,
    )

    # Config file argument
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    # API arguments
    parser.add_argument(
        "--api-url",
        help="Base URL of the EKB API (e.g., https://api.example.com/ekb/api)",
    )
    parser.add_argument(
        "--token",
        help="Bearer token for API authentication",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        help="Directory to save downloaded audio files and metadata",
    )

    # Split arguments
    # Use default=None so we can detect if the user explicitly set the flag
    parser.add_argument(
        "--split",
        action="store_true",
        dest="split",
        default=None,
        help="Enable train/val/test splitting",
    )
    parser.add_argument(
        "--no-split",
        action="store_false",
        dest="split",
        help="Disable train/val/test splitting",
    )
    parser.add_argument(
        "--test-source-files",
        help="Comma-separated list of source filenames to use for test set (e.g., 'file1.wav,file2.wav')",
    )
    parser.add_argument(
        "--test-files",
        type=int,
        help="Number of random files for test set (used if --test-source-files not specified)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Train ratio (after reserving test files)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="Validation ratio (after reserving test files)",
    )

    # Filter arguments
    parser.add_argument(
        "--source",
        help="Comma-separated list of source names to filter by",
    )
    parser.add_argument(
        "--label-hierarchy",
        help="Filter by label hierarchy (partial match supported)",
    )
    parser.add_argument(
        "--labeler",
        help="Filter by labeler name/email",
    )
    parser.add_argument(
        "--from-date",
        help="Filter labels created from this date (YYYY-MM-DD or ISO format)",
    )
    parser.add_argument(
        "--to-date",
        help="Filter labels created up to this date (YYYY-MM-DD or ISO format)",
    )
    parser.add_argument(
        "--confidence-min",
        type=float,
        help="Minimum confidence score (0.0-1.0)",
    )
    parser.add_argument(
        "--confidence-max",
        type=float,
        help="Maximum confidence score (0.0-1.0)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum event duration in milliseconds",
    )
    parser.add_argument(
        "--max-frequency",
        type=int,
        help="Maximum frequency in Hz",
    )

    # Control arguments
    parser.add_argument(
        "--max-labels",
        type=int,
        help="Maximum number of labels to download",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retry attempts for failed requests (default: 3)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        help="Backoff factor for exponential retry delay (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if args.config:
        if not args.config.exists():
            console.print(f"[red]Configuration file not found: {args.config}[/red]")
            sys.exit(1)
        config = DownloadConfig.from_yaml(args.config)
        console.print(f"[green]Loaded configuration from: {args.config}[/green]")
    else:
        # Start with default config
        config = DownloadConfig()

    # Merge CLI arguments with config (CLI takes precedence)
    config = merge_config_with_args(config, args)

    # Validate configuration
    errors = validate_config(config)
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        sys.exit(1)

    # Display configuration
    if args.verbose or args.dry_run:
        console.print("\n[cyan]Configuration:[/cyan]")
        console.print(f"  API URL: {config.api.url}")
        console.print(f"  Output dir: {config.output.dir}")
        console.print(f"  Split enabled: {config.split.enabled}")
        if config.split.enabled:
            console.print(f"  Test files: {config.split.test_files}")
            console.print(f"  Train/Val ratio: {config.split.train_ratio:.0%}/{config.split.val_ratio:.0%}")
        console.print(f"  Filters: {config.filters.to_dict()}")
        console.print(
            f"  Retry: max_retries={config.advanced.max_retries}, backoff={config.advanced.retry_backoff_factor}"
        )

    # Initialize API client with retry configuration
    api_client = EKBAPIClient(
        base_url=config.api.url,
        token=config.api.token,
        max_retries=config.advanced.max_retries,
        retry_backoff_factor=config.advanced.retry_backoff_factor,
    )

    # Initialize downloader
    downloader = DataDownloader(
        api_client=api_client,
        config=config,
        dry_run=args.dry_run,
    )

    try:
        # Run download
        results = downloader.run()

        # Exit with error code if any downloads failed
        failed = sum(1 for r in results if not r.success)
        if failed > 0:
            console.print(f"\n[red]Warning: {failed} downloads failed[/red]")
            sys.exit(1)

        if args.dry_run:
            console.print("\n[bold yellow]Dry run complete - no files were downloaded[/bold yellow]")
        else:
            console.print("\n[bold green]Download complete![/bold green]")
        sys.exit(0)

    except requests.exceptions.HTTPError as e:
        console.print(f"\n[red]API Error: {e}[/red]")
        if e.response is not None:
            console.print(f"[dim]Response: {e.response.text}[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
