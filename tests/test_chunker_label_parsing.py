"""Tests for chunker label parsing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rf_detr_finetuning.dataprocessor import AudioChunker, ChunkConfig, TimeBasedFFTConfig
from rf_detr_finetuning.dataprocessor import chunker as chunker_module


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        ("whistles > odontoceti", "odontoceti"),
        ("whistles + clicks", "clicks"),
        ("odontoceti", "odontoceti"),
    ],
)
def test_chunker_label_hierarchy_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hierarchy: str,
    expected: str,
) -> None:
    """Ensure label_hierarchy uses the leaf label."""
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    metadata_path = tmp_path / "meta.json"

    metadata = {
        "uuid": "test-audio",
        "label_hierarchy": hierarchy,
        "annotation": "fallback",
        "hz_min": 100.0,
        "hz_max": 1000.0,
    }
    metadata_path.write_text(json.dumps(metadata))

    def fake_load_audio_file(_: Path) -> tuple[np.ndarray, int]:
        return np.zeros(1000, dtype=np.float32), 1000

    def fake_compute_spectrogram(_: AudioChunker, __: np.ndarray, ___: int) -> np.ndarray:
        return np.zeros((32, 10), dtype=np.float32)

    monkeypatch.setattr(chunker_module, "load_audio_file", fake_load_audio_file)
    monkeypatch.setattr(AudioChunker, "_compute_spectrogram", fake_compute_spectrogram)

    chunker = AudioChunker(
        fft_config=TimeBasedFFTConfig(hop_ms=10.0, fft_ms=20.0, n_mels=32),
        chunk_config=ChunkConfig(
            window_duration_ms=1000.0,
            overlap_ratio=0.0,
            target_width=10,
            target_height=32,
            min_chunk_content_ratio=0.0,
        ),
    )

    chunks = chunker.chunk_audio_file(audio_path, metadata_path)

    assert len(chunks) == 1
    assert len(chunks[0].bboxes) == 1
    assert chunks[0].bboxes[0].category == expected
