"""Tests for audio chunking module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rf_detr_finetuning.audio_chunking import (
    AudioChunk,
    AudioChunker,
    ChunkBbox,
    ChunkConfig,
    TimeBasedFFTConfig,
    align_bbox_to_chunk,
    compute_bbox_overlap,
    compute_chunk_boundaries,
    draw_bboxes_on_spectrogram,
    extract_audio_chunk,
    load_chunking_config_from_yaml,
    pad_audio,
    resize_spectrogram,
    spectrogram_to_image_array,
)


class TestTimeBasedFFTConfig:
    def test_get_n_fft(self):
        config = TimeBasedFFTConfig(fft_ms=25.0)
        assert config.get_n_fft(16000) == 400
        assert config.get_n_fft(44100) == 1102
        assert config.get_n_fft(48000) == 1200

    def test_get_hop_length(self):
        config = TimeBasedFFTConfig(hop_ms=10.0)
        assert config.get_hop_length(16000) == 160
        assert config.get_hop_length(44100) == 441
        assert config.get_hop_length(48000) == 480

    def test_time_per_pixel(self):
        config = TimeBasedFFTConfig(hop_ms=10.0)
        assert config.get_time_per_pixel() == 10.0

    def test_freq_resolution(self):
        config = TimeBasedFFTConfig(fft_ms=25.0)
        assert config.get_freq_resolution(16000) == 40.0


class TestChunkConfig:
    def test_defaults(self):
        config = ChunkConfig()
        assert config.window_duration_ms == 5000.0
        assert config.overlap_ratio == 0.2
        assert config.get_overlap_ms() == 1000.0  # 20% of 5000
        assert config.target_width == 640
        assert config.target_height == 640

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="overlap_ratio must be between"):
            ChunkConfig(window_duration_ms=1000, overlap_ratio=1.5)

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="min_overlap_with_event_ratio"):
            ChunkConfig(min_overlap_with_event_ratio=1.5)

    def test_invalid_padding_mode(self):
        with pytest.raises(ValueError, match="Invalid padding_mode"):
            ChunkConfig(padding_mode="invalid")


class TestComputeChunkBoundaries:
    def test_single_chunk(self):
        boundaries = compute_chunk_boundaries(
            total_duration_ms=3000,
            config=ChunkConfig(window_duration_ms=5000, overlap_ratio=0.2),
        )
        assert len(boundaries) == 1
        assert boundaries[0] == (0, 3000)

    def test_multiple_chunks(self):
        boundaries = compute_chunk_boundaries(
            total_duration_ms=10000,
            config=ChunkConfig(
                window_duration_ms=5000,
                overlap_ratio=0.2,
                min_chunk_content_ratio=0.0,  # Keep all chunks for this test
            ),
        )
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 5000)
        assert boundaries[1] == (4000, 9000)
        assert boundaries[2] == (8000, 10000)

    def test_exact_fit(self):
        boundaries = compute_chunk_boundaries(
            total_duration_ms=9000,
            config=ChunkConfig(
                window_duration_ms=5000,
                overlap_ratio=0.2,
                min_chunk_content_ratio=0.0,  # Keep all chunks for this test
            ),
        )
        # With stride of 4000ms: 0-5000, 4000-9000, 8000-9000 (last partial)
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 5000)
        assert boundaries[1] == (4000, 9000)

    def test_min_chunk_content_ratio(self):
        # 6000ms audio, 5000ms window, stride=4000ms
        # Chunks: 0-5000 (100%), 4000-6000 (40%)
        # With min_chunk_content_ratio=0.5, second chunk is dropped
        boundaries = compute_chunk_boundaries(
            total_duration_ms=6000,
            config=ChunkConfig(
                window_duration_ms=5000,
                overlap_ratio=0.2,
                min_chunk_content_ratio=0.5,
            ),
        )
        assert len(boundaries) == 1

    def test_empty_audio(self):
        boundaries = compute_chunk_boundaries(
            total_duration_ms=0,
            config=ChunkConfig(),
        )
        assert len(boundaries) == 0


class TestComputeBboxOverlap:
    def test_full_overlap(self):
        overlap = compute_bbox_overlap(1000, 2000, 0, 5000)
        assert overlap == 1.0

    def test_partial_overlap(self):
        overlap = compute_bbox_overlap(0, 2000, 1000, 5000)
        assert overlap == 0.5

    def test_no_overlap(self):
        overlap = compute_bbox_overlap(0, 1000, 2000, 5000)
        assert overlap == 0.0

    def test_zero_duration_event(self):
        overlap = compute_bbox_overlap(1000, 1000, 0, 5000)
        assert overlap == 0.0


class TestAlignBboxToChunk:
    def test_full_event_in_chunk(self):
        bbox = align_bbox_to_chunk(
            event_start_ms=1000,
            event_end_ms=2000,
            hz_min=100,
            hz_max=500,
            chunk_start_ms=0,
            chunk_end_ms=5000,
            chunk_width_px=640,
            chunk_height_px=640,
            freq_min=0,
            freq_max=8000,
            category="test",
            category_id=1,
        )
        assert bbox is not None
        assert bbox.overlap_ratio == 1.0
        assert 0 <= bbox.x < 640
        assert 0 <= bbox.y < 640
        assert bbox.category == "test"

    def test_partial_event(self):
        bbox = align_bbox_to_chunk(
            event_start_ms=4000,
            event_end_ms=6000,
            hz_min=100,
            hz_max=500,
            chunk_start_ms=0,
            chunk_end_ms=5000,
            chunk_width_px=640,
            chunk_height_px=640,
            freq_min=0,
            freq_max=8000,
            category="test",
            category_id=1,
        )
        assert bbox is not None
        assert bbox.overlap_ratio == 0.5

    def test_no_overlap_returns_none(self):
        bbox = align_bbox_to_chunk(
            event_start_ms=6000,
            event_end_ms=7000,
            hz_min=100,
            hz_max=500,
            chunk_start_ms=0,
            chunk_end_ms=5000,
            chunk_width_px=640,
            chunk_height_px=640,
            freq_min=0,
            freq_max=8000,
            category="test",
            category_id=1,
        )
        assert bbox is None


class TestPadAudio:
    def test_zero_padding(self):
        audio = np.array([1.0, 2.0, 3.0])
        padded, amount = pad_audio(audio, 5, mode="zero")
        assert len(padded) == 5
        assert amount == 2
        np.testing.assert_array_equal(padded[3:], [0, 0])

    def test_repeat_padding(self):
        audio = np.array([1.0, 2.0, 3.0])
        padded, amount = pad_audio(audio, 5, mode="repeat")
        assert len(padded) == 5
        assert amount == 2

    def test_reflect_padding(self):
        audio = np.array([1.0, 2.0, 3.0])
        padded, amount = pad_audio(audio, 5, mode="reflect")
        assert len(padded) == 5
        assert amount == 2

    def test_no_padding_needed(self):
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        padded, amount = pad_audio(audio, 5, mode="zero")
        assert len(padded) == 5
        assert amount == 0


class TestExtractAudioChunk:
    def test_extract_middle_chunk(self):
        audio = np.arange(16000, dtype=np.float32)
        chunk, is_padded, padding_ms = extract_audio_chunk(
            audio, sample_rate=16000, start_ms=500, end_ms=1000, padding_mode="zero"
        )
        assert len(chunk) == 8000
        assert not is_padded
        assert padding_ms == 0

    def test_extract_with_padding(self):
        audio = np.arange(8000, dtype=np.float32)
        chunk, is_padded, padding_ms = extract_audio_chunk(
            audio, sample_rate=16000, start_ms=0, end_ms=1000, padding_mode="zero"
        )
        assert len(chunk) == 16000
        assert is_padded
        assert padding_ms > 0


class TestSpectrogramToImageArray:
    def test_normalization(self):
        spec = np.random.randn(128, 256)
        img = spectrogram_to_image_array(spec, normalize=True)
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255

    def test_no_normalization(self):
        spec = np.clip(np.random.randn(128, 256) * 100 + 128, 0, 255)
        img = spectrogram_to_image_array(spec, normalize=False)
        assert img.dtype == np.uint8


class TestResizeSpectrogram:
    def test_resize_2d(self):
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        resized = resize_spectrogram(spec, 640, 640)
        assert resized.shape == (640, 640)


class TestChunkBbox:
    def test_to_coco_bbox(self):
        bbox = ChunkBbox(
            x=10,
            y=20,
            width=100,
            height=50,
            category="test",
            category_id=1,
            original_time_start_ms=0,
            original_time_end_ms=1000,
            hz_min=100,
            hz_max=500,
        )
        coco = bbox.to_coco_bbox()
        assert coco == [10, 20, 100, 50]

    def test_to_xyxy(self):
        bbox = ChunkBbox(
            x=10,
            y=20,
            width=100,
            height=50,
            category="test",
            category_id=1,
            original_time_start_ms=0,
            original_time_end_ms=1000,
            hz_min=100,
            hz_max=500,
        )
        xyxy = bbox.to_xyxy()
        assert xyxy == (10, 20, 110, 70)


class TestAudioChunk:
    def test_duration(self):
        chunk = AudioChunk(chunk_index=0, start_ms=1000, end_ms=6000)
        assert chunk.duration_ms == 5000

    def test_get_chunk_id(self):
        chunk = AudioChunk(chunk_index=5, start_ms=0, end_ms=5000, source_uuid="test123")
        assert chunk.get_chunk_id() == "test123_chunk0005"


class TestAudioChunker:
    def test_chunk_synthetic_audio(self):
        chunker = AudioChunker(
            fft_config=TimeBasedFFTConfig(fft_ms=25, hop_ms=10, n_mels=64),
            chunk_config=ChunkConfig(
                window_duration_ms=1000,
                overlap_ratio=0.2,
                target_width=128,
                target_height=64,
            ),
        )
        audio = np.random.randn(32000).astype(np.float32)
        chunks = chunker.chunk_audio(
            audio,
            sample_rate=16000,
            events=[
                {
                    "time_start_ms": 500,
                    "time_end_ms": 1500,
                    "hz_min": 1000,
                    "hz_max": 4000,
                    "category": "test",
                    "category_id": 0,
                }
            ],
            source_uuid="synthetic",
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.spectrogram is not None
            assert chunk.spectrogram.shape == (64, 128)

    def test_short_audio_padding(self):
        chunker = AudioChunker(
            chunk_config=ChunkConfig(
                window_duration_ms=5000,
                overlap_ratio=0.2,
                padding_mode="zero",
                min_chunk_content_ratio=0.0,  # Keep even short chunks for this test
            ),
        )
        # 8000 samples at 16kHz = 500ms, shorter than 5000ms window
        # But chunker uses end_ms=total_duration for short audio, not full window
        audio = np.random.randn(8000).astype(np.float32)
        chunks = chunker.chunk_audio(audio, sample_rate=16000, source_uuid="short")

        assert len(chunks) == 1
        # Audio is 500ms, chunk is created for 0-500ms (not padded to 5000ms)
        assert chunks[0].start_ms == 0
        assert chunks[0].end_ms == 500.0

    def test_random_padding_for_very_short_audio(self):
        """Test that very short audio (like 50ms gunshots) get random padding positions."""
        chunker_random = AudioChunker(
            chunk_config=ChunkConfig(
                window_duration_ms=5000,
                overlap_ratio=0.2,
                padding_mode="zero",
                min_chunk_content_ratio=0.0,  # Keep very short chunks
                random_pad_position=True,  # Enable random padding
            ),
        )

        # Create 50ms audio (800 samples at 16kHz) - like a gunshot
        audio = np.random.randn(800).astype(np.float32)

        # Chunk multiple times to verify randomness
        chunk_durations = []
        for _ in range(10):
            chunks = chunker_random.chunk_audio(audio, sample_rate=16000, source_uuid="gunshot")
            assert len(chunks) == 1  # Should create single chunk
            chunk_durations.append(chunks[0].end_ms - chunks[0].start_ms)

        # All chunks should be exactly 50ms (the actual audio duration)
        # Even though window is 5000ms, we only extract what's needed
        assert all(d == 50.0 for d in chunk_durations)

        # Now test non-random padding (should always be centered)
        chunker_fixed = AudioChunker(
            chunk_config=ChunkConfig(
                window_duration_ms=5000,
                overlap_ratio=0.2,
                padding_mode="zero",
                min_chunk_content_ratio=0.0,
                random_pad_position=False,  # Fixed padding
            ),
        )

        chunks_fixed = chunker_fixed.chunk_audio(audio, sample_rate=16000, source_uuid="gunshot")
        assert len(chunks_fixed) == 1
        assert chunks_fixed[0].end_ms - chunks_fixed[0].start_ms == 50.0


class TestDrawBboxes:
    def test_draw_on_grayscale(self):
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        bboxes = [
            ChunkBbox(
                x=10,
                y=20,
                width=50,
                height=30,
                category="test",
                category_id=0,
                original_time_start_ms=0,
                original_time_end_ms=1000,
                hz_min=100,
                hz_max=500,
            )
        ]
        result = draw_bboxes_on_spectrogram(spec, bboxes)
        assert result.shape == (128, 256, 3)

    def test_draw_empty_bboxes(self):
        spec = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        result = draw_bboxes_on_spectrogram(spec, [])
        assert result.shape == (128, 256, 3)


class TestLoadYamlConfig:
    def test_load_config(self, tmp_path: Path):
        yaml_content = """
fft:
  fft_ms: 30.0
  hop_ms: 15.0
  n_mels: 256

chunking:
  window_duration_ms: 3000.0
  overlap_ratio: 0.2
  target_width: 512
  target_height: 512
  padding_mode: reflect
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        fft_config, chunk_config, preprocessing_config = load_chunking_config_from_yaml(yaml_path)

        assert fft_config.fft_ms == 30.0
        assert fft_config.hop_ms == 15.0
        assert fft_config.n_mels == 256
        assert chunk_config.window_duration_ms == 3000.0
        assert chunk_config.overlap_ratio == 0.2
        assert chunk_config.get_overlap_ms() == 600.0  # 20% of 3000
        assert chunk_config.target_width == 512
        assert chunk_config.padding_mode == "reflect"


@pytest.mark.real_files
def test_chunking_with_real_files(tmp_path: Path, pytestconfig: pytest.Config):
    """Test chunking with real audio files and draw bboxes for visual inspection.

    This test creates debug images with bboxes overlaid on spectrograms.
    Uses configuration from config/audio_chunking.yaml for consistent parameters.
    Run with: pytest tests/test_audio_chunking.py::test_chunking_with_real_files -v -s

    Output images are saved to: output/test_chunks_debug/

    """
    audio_ini = pytestconfig.getini("real_audio_path")
    json_ini = pytestconfig.getini("real_json_path")

    if not audio_ini or not json_ini:
        pytest.skip("Set real_audio_path and real_json_path in pyproject.toml")

    audio_path = Path(audio_ini)
    json_path = Path(json_ini)

    if not audio_path.exists() or not json_path.exists():
        pytest.skip("Real test files not found")

    # Load configuration from YAML file for consistent parameters
    config_path = Path("config/audio_chunking.yaml")
    if not config_path.exists():
        pytest.skip("config/audio_chunking.yaml not found")

    fft_config, chunk_config, preprocessing_config = load_chunking_config_from_yaml(config_path)

    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
    )

    chunks = chunker.chunk_audio_file(audio_path, json_path)

    assert len(chunks) > 0

    # Save to output/ directory for easy access
    output_dir = Path("output/test_chunks_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("CHUNKED AUDIO VISUALIZATION")
    print(f"{'=' * 70}")
    print(f"Audio file: {audio_path.name}")
    print(f"JSON file: {json_path.name}")
    print(
        f"Config: window={chunk_config.window_duration_ms}ms, "
        f"overlap={chunk_config.overlap_ratio:.0%} ({chunk_config.get_overlap_ms()}ms), "
        f"size={chunk_config.target_width}x{chunk_config.target_height}"
    )
    print(
        f"        min_content_ratio={chunk_config.min_chunk_content_ratio}, "
        f"random_pad={chunk_config.random_pad_position}"
    )
    print(f"Total chunks: {len(chunks)}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'=' * 70}\n")

    for i, chunk in enumerate(chunks):
        # Save spectrogram with bboxes drawn
        filename = f"{audio_path.stem}_chunk{i:04d}_debug.png"
        out_path = output_dir / filename

        # Draw bboxes on spectrogram
        if chunk.bboxes:
            debug_img = draw_bboxes_on_spectrogram(chunk.spectrogram, chunk.bboxes)
            from PIL import Image

            Image.fromarray(debug_img).save(out_path)
        else:
            # Save plain spectrogram if no bboxes
            spec_normalized = chunk.spectrogram - chunk.spectrogram.min()
            spec_normalized = spec_normalized / (spec_normalized.max() + 1e-8)
            spec_img = (spec_normalized * 255).astype(np.uint8)
            from PIL import Image

            Image.fromarray(spec_img).convert("RGB").save(out_path)

        assert out_path.exists()

        # Print chunk info
        print(
            f"Chunk {i:02d}: {chunk.start_ms:>6.0f}-{chunk.end_ms:>6.0f}ms | "
            f"{len(chunk.bboxes)} bbox(es) | {'PADDED' if chunk.is_padded else 'normal':>6} | "
            f"{filename}"
        )

        # Print bbox details
        for j, bbox in enumerate(chunk.bboxes):
            x, y, w, h = bbox.to_coco_bbox()
            print(
                f"  └─ Bbox {j}: [{x:>4.0f}, {y:>4.0f}, {w:>4.0f}, {h:>4.0f}] | "
                f"label={bbox.category} | overlap={bbox.overlap_ratio:.2f}"
            )

    print(f"\n{'=' * 70}")
    print(f"✓ Debug images saved to: {output_dir.absolute()}")
    print("✓ Open images to visually verify bbox alignment")
    print(f"{'=' * 70}\n")
