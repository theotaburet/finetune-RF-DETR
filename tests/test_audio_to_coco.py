"""Tests for the audio_to_coco module."""

import numpy as np
import pytest

from rf_detr_finetuning.audio_to_coco import (
    AudioMetadata,
    BboxFrequencyInfo,
    BoundingBox,
    CategoryRegistry,
    FrequencyMapper,
    SpectrogramConfig,
    TimeMapper,
    decode_bbox_to_frequency,
    spectrogram_to_image,
    validate_coco_dataset,
)


class TestFrequencyMapper:
    """Tests for FrequencyMapper class."""

    def test_hz_to_pixel_boundaries(self):
        """Test frequency mapping at scale boundaries."""
        mapper = FrequencyMapper(n_mels=128, fmin=0, fmax=8000, sample_rate=16000)

        # fmax should map to top (pixel ~0)
        assert mapper.hz_to_pixel(8000) == pytest.approx(0, abs=1)

        # fmin should map to bottom (pixel ~127)
        assert mapper.hz_to_pixel(0) == pytest.approx(128, abs=1)

    def test_pixel_to_hz_roundtrip(self):
        """Verify pixel -> Hz -> pixel is consistent."""
        mapper = FrequencyMapper(n_mels=128, fmin=0, fmax=8000, sample_rate=16000)

        for pixel in [0, 32, 64, 96, 127]:
            hz = mapper.pixel_to_hz(pixel)
            back = mapper.hz_to_pixel(hz)
            assert back == pytest.approx(pixel, abs=0.5), f"Roundtrip failed for pixel {pixel}"

    def test_hz_clamping(self):
        """Test that Hz values are clamped to valid range."""
        mapper = FrequencyMapper(n_mels=128, fmin=100, fmax=4000, sample_rate=16000)

        # Below fmin
        pixel_below = mapper.hz_to_pixel(50)
        pixel_fmin = mapper.hz_to_pixel(100)
        assert pixel_below == pixel_fmin

        # Above fmax
        pixel_above = mapper.hz_to_pixel(5000)
        pixel_fmax = mapper.hz_to_pixel(4000)
        assert pixel_above == pixel_fmax

    def test_normalized_frequency(self):
        """Test normalized frequency conversion."""
        mapper = FrequencyMapper(
            n_mels=128,
            fmin=0,
            fmax=8000,
            sample_rate=16000,
            normalize_frequency=True,
        )

        # 1000 Hz with Nyquist=8000 → 0.125
        assert mapper.hz_to_normalized(1000) == pytest.approx(0.125)

        # Roundtrip
        norm = mapper.hz_to_normalized(4000)
        hz = mapper.normalized_to_hz(norm)
        assert hz == pytest.approx(4000)


class TestTimeMapper:
    """Tests for TimeMapper class."""

    def test_ms_to_pixel_roundtrip(self):
        """Verify ms -> pixel -> ms is consistent."""
        mapper = TimeMapper(total_duration_ms=10000, total_frames=1000, sample_rate=44100, hop_length=512)

        for ms in [0, 2500, 5000, 7500, 10000]:
            pixel = mapper.ms_to_pixel(ms)
            back = mapper.pixel_to_ms(pixel)
            assert back == pytest.approx(ms, rel=0.01)

    def test_frame_timing(self):
        """Test ms_per_frame calculation."""
        mapper = TimeMapper(total_duration_ms=10000, total_frames=1000, sample_rate=44100, hop_length=512)

        expected_ms_per_frame = (512 / 44100) * 1000
        assert mapper.ms_per_frame == pytest.approx(expected_ms_per_frame)


class TestBboxFrequencyInfo:
    """Tests for BboxFrequencyInfo class."""

    def test_frequency_band_classification(self):
        """Test frequency band assignment."""
        test_cases = [
            (50, "infrasonic"),  # < 100 Hz
            (300, "very_low"),  # 100-500 Hz
            (1000, "low"),  # 500-2000 Hz
            (3000, "mid"),  # 2000-5000 Hz
            (7000, "high"),  # 5000-10000 Hz
            (15000, "very_high"),  # > 10000 Hz
        ]

        for hz_center, expected_band in test_cases:
            info = BboxFrequencyInfo(
                hz_min=hz_center - 100,
                hz_max=hz_center + 100,
                hz_center=hz_center,
                time_start_ms=0,
                time_end_ms=1000,
                duration_ms=1000,
                bandwidth_hz=200,
            )
            assert info.get_frequency_band() == expected_band, f"Failed for {hz_center} Hz"

    def test_to_dict(self):
        """Test serialization to dict."""
        info = BboxFrequencyInfo(
            hz_min=100,
            hz_max=500,
            hz_center=300,
            time_start_ms=0,
            time_end_ms=1000,
            duration_ms=1000,
            bandwidth_hz=400,
            sample_rate=16000,
            normalized_freq_min=0.0125,
            normalized_freq_max=0.0625,
            normalized_freq_center=0.0375,
        )

        d = info.to_dict()
        assert d["hz_center"] == 300
        assert d["normalized_freq_center"] == 0.0375
        assert d["frequency_band"] == "very_low"


class TestDecodeBboxToFrequency:
    """Tests for decode_bbox_to_frequency function."""

    def test_decode_full_bbox(self):
        """Test decoding a full-spectrogram bbox."""
        freq_mapper = FrequencyMapper(n_mels=128, fmin=0, fmax=8000, sample_rate=16000)
        time_mapper = TimeMapper(total_duration_ms=10000, total_frames=1000, sample_rate=16000, hop_length=512)

        bbox = BoundingBox(x=0, y=0, width=1000, height=128, category_id=0, category_name="test")

        info = decode_bbox_to_frequency(bbox, freq_mapper, time_mapper)

        # Full height = full frequency range
        assert info.hz_min == pytest.approx(0, abs=100)
        assert info.hz_max == pytest.approx(8000, abs=100)

    def test_normalized_frequencies(self):
        """Test that normalized frequencies are computed."""
        freq_mapper = FrequencyMapper(n_mels=128, fmin=0, fmax=8000, sample_rate=16000)
        time_mapper = TimeMapper(total_duration_ms=10000, total_frames=1000, sample_rate=16000, hop_length=512)

        bbox = BoundingBox(x=0, y=64, width=500, height=32, category_id=0, category_name="test")

        info = decode_bbox_to_frequency(bbox, freq_mapper, time_mapper)

        # Normalized values should be between 0 and 1
        assert 0 <= info.normalized_freq_min <= 1
        assert 0 <= info.normalized_freq_max <= 1
        assert info.sample_rate == 16000


class TestCategoryRegistry:
    """Tests for CategoryRegistry class."""

    def test_get_or_create(self):
        """Test category creation and retrieval."""
        registry = CategoryRegistry()

        id1 = registry.get_or_create("cat_a")
        id2 = registry.get_or_create("cat_b")
        id3 = registry.get_or_create("cat_a")

        assert id1 == 0
        assert id2 == 1
        assert id3 == id1  # Same category, same ID

    def test_frequency_aware_categories(self):
        """Test frequency-based category splitting."""
        registry = CategoryRegistry(frequency_bins=[(0, 1000, "low"), (1000, 5000, "mid"), (5000, 22050, "high")])

        # Low frequency event
        id1, name1 = registry.get_category_with_frequency("whale", 100, 500)
        assert name1 == "whale_low"

        # High frequency event
        id2, name2 = registry.get_category_with_frequency("whale", 8000, 12000)
        assert name2 == "whale_high"
        assert id2 != id1  # Different categories


class TestSpectrogramToImage:
    """Tests for spectrogram_to_image function."""

    def test_output_shape(self):
        """Test that output image has correct dimensions."""
        spec = np.random.randn(128, 500).astype(np.float32)
        img = spectrogram_to_image(spec, normalize=True)

        assert img.size == (500, 128)  # PIL uses (width, height)
        assert img.mode == "RGB"

    def test_normalization(self):
        """Test that normalization produces valid pixel values."""
        spec = np.array([[0, 1], [2, 3]], dtype=np.float32)
        img = spectrogram_to_image(spec, normalize=True)

        pixels = np.array(img)
        assert pixels.min() >= 0
        assert pixels.max() <= 255


class TestValidateCOCO:
    """Tests for validate_coco_dataset function."""

    def test_valid_dataset(self):
        """Test validation passes for valid dataset."""
        coco = {
            "images": [{"id": 1, "width": 640, "height": 480, "file_name": "test.png"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 100, 100]}],
            "categories": [{"id": 0, "name": "test"}],
        }

        is_valid, errors = validate_coco_dataset(coco)
        assert is_valid
        assert len(errors) == 0

    def test_missing_keys(self):
        """Test validation fails for missing required keys."""
        coco = {"images": []}

        is_valid, errors = validate_coco_dataset(coco)
        assert not is_valid
        assert any("Missing required keys" in e for e in errors)

    def test_bbox_out_of_bounds(self):
        """Test validation catches out-of-bounds bbox."""
        coco = {
            "images": [{"id": 1, "width": 100, "height": 100, "file_name": "test.png"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [50, 50, 100, 100]}  # Exceeds bounds
            ],
            "categories": [{"id": 0, "name": "test"}],
        }

        is_valid, errors = validate_coco_dataset(coco)
        assert not is_valid
        assert any("exceeds" in e for e in errors)


class TestSpectrogramConfig:
    """Tests for SpectrogramConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = SpectrogramConfig()

        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.n_mels == 128
        assert config.normalize_frequency is True
        assert config.target_sr is None


class TestAudioMetadata:
    """Tests for AudioMetadata parsing."""

    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "uuid": "test-123",
            "annotation": "whale_call",
            "duration": 5000,
            "hz_min": 100,
            "hz_max": 2000,
            "sample_rate": 44100,
        }

        metadata = AudioMetadata.from_dict(data)

        assert metadata.uuid == "test-123"
        assert metadata.annotation == "whale_call"
        assert metadata.duration == 5000
        assert metadata.hz_min == 100

    def test_missing_fields_have_defaults(self):
        """Test that missing fields use defaults."""
        data = {"uuid": "test", "annotation": "unknown"}

        metadata = AudioMetadata.from_dict(data)

        assert metadata.sample_rate == 44100  # Default
        assert metadata.confidence == 1.0  # Default
