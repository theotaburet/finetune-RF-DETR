"""Baseline snapshot generator for regression testing.

This module captures the current system behavior before refactoring to enable verification that outputs remain
unchanged.

"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureSnapshot:
    """Snapshot of feature extraction output."""

    shape: tuple[int, ...]
    dtype: str
    mean: float
    std: float
    min_val: float
    max_val: float
    checksum: str  # MD5 of flattened array bytes


@dataclass
class WindowSnapshot:
    """Snapshot of windowing/chunking output."""

    total_duration_ms: float
    num_windows: int
    boundaries: list[tuple[float, float]]
    window_duration_ms: float
    overlap_ms: float


@dataclass
class BboxSnapshot:
    """Snapshot of bounding box alignment output."""

    num_bboxes: int
    bbox_coords: list[list[float]]  # COCO format [x, y, w, h]
    categories: list[str]
    overlap_ratios: list[float]


@dataclass
class BaselineSnapshot:
    """Complete baseline snapshot for regression testing."""

    version: str = "1.0.0"
    audio_file: str = ""
    sample_rate: int = 0
    features: FeatureSnapshot | None = None
    windows: WindowSnapshot | None = None
    bboxes: list[BboxSnapshot] | None = None
    metadata: dict[str, Any] | None = None


def compute_array_checksum(arr: np.ndarray) -> str:
    """Compute MD5 checksum of array for equality testing."""
    return hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()


def capture_feature_snapshot(spectrogram: np.ndarray) -> FeatureSnapshot:
    """Capture snapshot of spectrogram/feature tensor."""
    return FeatureSnapshot(
        shape=spectrogram.shape,
        dtype=str(spectrogram.dtype),
        mean=float(np.mean(spectrogram)),
        std=float(np.std(spectrogram)),
        min_val=float(np.min(spectrogram)),
        max_val=float(np.max(spectrogram)),
        checksum=compute_array_checksum(spectrogram),
    )


def capture_window_snapshot(
    total_duration_ms: float,
    boundaries: list[tuple[float, float]],
    window_duration_ms: float,
    overlap_ms: float,
) -> WindowSnapshot:
    """Capture snapshot of windowing output."""
    return WindowSnapshot(
        total_duration_ms=total_duration_ms,
        num_windows=len(boundaries),
        boundaries=boundaries,
        window_duration_ms=window_duration_ms,
        overlap_ms=overlap_ms,
    )


def capture_bbox_snapshot(bboxes: list) -> BboxSnapshot:
    """Capture snapshot of bbox alignment output."""
    return BboxSnapshot(
        num_bboxes=len(bboxes),
        bbox_coords=[bbox.to_coco_bbox() for bbox in bboxes],
        categories=[bbox.category for bbox in bboxes],
        overlap_ratios=[bbox.overlap_ratio for bbox in bboxes],
    )


def save_snapshot(snapshot: BaselineSnapshot, output_path: Path) -> None:
    """Save baseline snapshot to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict, handling nested dataclasses
    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [to_dict(item) for item in obj]
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    data = to_dict(snapshot)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_snapshot(input_path: Path) -> BaselineSnapshot:
    """Load baseline snapshot from JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    # Reconstruct nested dataclasses
    features = None
    if data.get("features"):
        features = FeatureSnapshot(**data["features"])

    windows = None
    if data.get("windows"):
        # Convert boundaries back to tuples
        data["windows"]["boundaries"] = [tuple(b) for b in data["windows"]["boundaries"]]
        windows = WindowSnapshot(**data["windows"])

    bboxes = None
    if data.get("bboxes"):
        bboxes = [BboxSnapshot(**b) for b in data["bboxes"]]

    return BaselineSnapshot(
        version=data.get("version", "1.0.0"),
        audio_file=data.get("audio_file", ""),
        sample_rate=data.get("sample_rate", 0),
        features=features,
        windows=windows,
        bboxes=bboxes,
        metadata=data.get("metadata"),
    )


def compare_snapshots(
    baseline: BaselineSnapshot,
    current: BaselineSnapshot,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare two snapshots and return differences.

    Returns:
        Dictionary with comparison results. Empty if identical.

    """
    differences = {}

    # Compare features
    if baseline.features and current.features:
        bf, cf = baseline.features, current.features
        if bf.shape != cf.shape:
            differences["features.shape"] = {"baseline": bf.shape, "current": cf.shape}
        if bf.dtype != cf.dtype:
            differences["features.dtype"] = {"baseline": bf.dtype, "current": cf.dtype}
        if not np.isclose(bf.mean, cf.mean, atol=atol):
            differences["features.mean"] = {"baseline": bf.mean, "current": cf.mean}
        if not np.isclose(bf.std, cf.std, atol=atol):
            differences["features.std"] = {"baseline": bf.std, "current": cf.std}
        if bf.checksum != cf.checksum:
            differences["features.checksum"] = {"baseline": bf.checksum, "current": cf.checksum}

    # Compare windows
    if baseline.windows and current.windows:
        bw, cw = baseline.windows, current.windows
        if bw.num_windows != cw.num_windows:
            differences["windows.num_windows"] = {"baseline": bw.num_windows, "current": cw.num_windows}
        if bw.boundaries != cw.boundaries:
            differences["windows.boundaries"] = {"baseline": bw.boundaries, "current": cw.boundaries}

    # Compare bboxes
    if baseline.bboxes and current.bboxes:
        if len(baseline.bboxes) != len(current.bboxes):
            differences["bboxes.count"] = {"baseline": len(baseline.bboxes), "current": len(current.bboxes)}
        else:
            for i, (bb, cb) in enumerate(zip(baseline.bboxes, current.bboxes)):
                if bb.num_bboxes != cb.num_bboxes:
                    differences[f"bboxes[{i}].num_bboxes"] = {"baseline": bb.num_bboxes, "current": cb.num_bboxes}
                if bb.categories != cb.categories:
                    differences[f"bboxes[{i}].categories"] = {"baseline": bb.categories, "current": cb.categories}

    return differences


def generate_baseline_from_audio(
    audio_path: Path,
    metadata_path: Path | None = None,
    config_path: Path | None = None,
) -> BaselineSnapshot:
    """Generate baseline snapshot from an audio file.

    This uses the current (pre-refactor) implementation to capture expected outputs for regression testing.

    """
    from rf_detr_finetuning.audio_chunking import (
        AudioChunker,
        ChunkConfig,
        TimeBasedFFTConfig,
        compute_chunk_boundaries,
        load_chunking_config_from_yaml,
    )

    # Load config
    if config_path and config_path.exists():
        fft_config, chunk_config, preprocessing_config, spec_config = load_chunking_config_from_yaml(config_path)
    else:
        fft_config = TimeBasedFFTConfig()
        chunk_config = ChunkConfig()
        preprocessing_config = None
        spec_config = {"freq_scale": "mel", "fmin": 0.0, "fmax": None}

    # Create chunker and process
    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preprocessing_config,
        freq_scale=spec_config["freq_scale"],
        fmin=spec_config["fmin"],
        fmax=spec_config["fmax"],
    )

    chunks = chunker.chunk_audio_file(audio_path, metadata_path)

    if not chunks:
        return BaselineSnapshot(audio_file=str(audio_path))

    # Capture feature snapshot from first chunk
    first_chunk = chunks[0]
    features = capture_feature_snapshot(first_chunk.spectrogram)

    # Capture window snapshot
    from ezakodio.io import load_audio

    audio_tensor, sample_rate = load_audio(str(audio_path), mono=True, device="cpu")
    audio = audio_tensor.cpu().numpy().flatten()
    total_duration_ms = (len(audio) / sample_rate) * 1000

    boundaries = compute_chunk_boundaries(total_duration_ms, chunk_config)
    windows = capture_window_snapshot(
        total_duration_ms=total_duration_ms,
        boundaries=boundaries,
        window_duration_ms=chunk_config.window_duration_ms,
        overlap_ms=chunk_config.get_overlap_ms(),
    )

    # Capture bbox snapshots for each chunk
    bbox_snapshots = [capture_bbox_snapshot(chunk.bboxes) for chunk in chunks]

    return BaselineSnapshot(
        audio_file=str(audio_path),
        sample_rate=sample_rate,
        features=features,
        windows=windows,
        bboxes=bbox_snapshots,
        metadata={
            "fft_ms": fft_config.fft_ms,
            "hop_ms": fft_config.hop_ms,
            "n_mels": fft_config.n_mels,
            "window_duration_ms": chunk_config.window_duration_ms,
            "overlap_ratio": chunk_config.overlap_ratio,
            "num_chunks": len(chunks),
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate baseline snapshot for regression testing")
    parser.add_argument("--audio", type=Path, required=True, help="Path to audio file")
    parser.add_argument("--metadata", type=Path, help="Path to metadata JSON file")
    parser.add_argument("--config", type=Path, help="Path to config YAML file")
    parser.add_argument("--output", type=Path, default=Path("tests/regression/baseline.json"))
    args = parser.parse_args()

    snapshot = generate_baseline_from_audio(args.audio, args.metadata, args.config)
    save_snapshot(snapshot, args.output)
    print(f"Baseline snapshot saved to: {args.output}")
