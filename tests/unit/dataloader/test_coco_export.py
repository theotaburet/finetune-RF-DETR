"""Tests for COCO export module.

These tests verify that the COCO export functionality works correctly with the new AudioChunker-based implementation.

"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from rf_detr_finetuning.dataloader.coco_export import (
    CategoryRegistry,
    COCODatasetBuilder,
    convert_audio_to_coco,
    parse_frequency_bins,
    validate_coco_dataset,
)


class TestCategoryRegistry:
    """Tests for CategoryRegistry."""

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

    def test_to_coco_categories(self):
        """Test export to COCO format."""
        registry = CategoryRegistry()
        registry.get_or_create("cat_a")
        registry.get_or_create("cat_b")

        coco_cats = registry.to_coco_categories()

        assert len(coco_cats) == 2
        assert coco_cats[0]["id"] == 0
        assert coco_cats[0]["name"] == "cat_a"
        assert coco_cats[0]["supercategory"] == "audio_event"


class TestCOCODatasetBuilder:
    """Tests for COCODatasetBuilder."""

    def test_add_image(self):
        """Test adding images."""
        builder = COCODatasetBuilder()

        image_id = builder.add_image("test.png", 640, 480)

        assert image_id == 1
        assert len(builder.images) == 1
        assert builder.images[0]["file_name"] == "test.png"
        assert builder.images[0]["width"] == 640
        assert builder.images[0]["height"] == 480

    def test_add_annotation(self):
        """Test adding annotations."""
        builder = COCODatasetBuilder()

        image_id = builder.add_image("test.png", 640, 480)
        ann_id = builder.add_annotation([10, 20, 100, 50], image_id, 0)

        assert ann_id == 1
        assert len(builder.annotations) == 1
        assert builder.annotations[0]["bbox"] == [10, 20, 100, 50]
        assert builder.annotations[0]["category_id"] == 0
        assert builder.annotations[0]["area"] == 5000

    def test_to_dict(self):
        """Test export to dictionary."""
        builder = COCODatasetBuilder()
        builder.add_image("test.png", 640, 480)
        builder.add_annotation([10, 20, 100, 50], 1, 0)

        registry = CategoryRegistry()
        registry.get_or_create("test")
        builder.set_categories(registry)

        data = builder.to_dict()

        assert "info" in data
        assert "licenses" in data
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading COCO dataset."""
        builder = COCODatasetBuilder()
        builder.add_image("test.png", 640, 480)
        builder.add_annotation([10, 20, 100, 50], 1, 0)

        registry = CategoryRegistry()
        registry.get_or_create("test")
        builder.set_categories(registry)

        output_path = tmp_path / "test_coco.json"
        builder.save(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1


class TestParseFrequencyBins:
    """Tests for parse_frequency_bins."""

    def test_parse_single_bin(self):
        """Test parsing single frequency bin."""
        result = parse_frequency_bins("0-500:low")

        assert result == [(0.0, 500.0, "low")]

    def test_parse_multiple_bins(self):
        """Test parsing multiple frequency bins."""
        result = parse_frequency_bins("0-500:low,500-2000:mid,2000-8000:high")

        assert len(result) == 3
        assert result[0] == (0.0, 500.0, "low")
        assert result[1] == (500.0, 2000.0, "mid")
        assert result[2] == (2000.0, 8000.0, "high")

    def test_parse_none(self):
        """Test parsing None."""
        result = parse_frequency_bins(None)

        assert result is None

    def test_parse_empty(self):
        """Test parsing empty string."""
        result = parse_frequency_bins("")

        assert result is None


class TestValidateCocoDataset:
    """Tests for validate_coco_dataset."""

    def test_valid_dataset(self):
        """Test validation of valid dataset."""
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
        assert any("Missing required key" in e for e in errors)

    def test_invalid_bbox(self):
        """Test validation catches invalid bbox."""
        coco = {
            "images": [{"id": 1, "width": 100, "height": 100, "file_name": "test.png"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10]}  # Invalid length
            ],
            "categories": [{"id": 0, "name": "test"}],
        }

        is_valid, errors = validate_coco_dataset(coco)
        assert not is_valid
        assert any("Invalid bbox" in e for e in errors)


class TestConvertAudioToCoco:
    """Tests for convert_audio_to_coco function."""

    def test_convert_synthetic_audio(self, tmp_path: Path):
        """Test conversion with synthetic audio."""
        # Create synthetic audio file
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        sample_rate = 16000
        duration_s = 0.25
        frequency_hz = 440.0

        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * frequency_hz * t)
        samples = (audio * 32767).astype(np.int16)

        audio_path = audio_dir / "sample.wav"
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        # Create metadata
        metadata = {
            "uuid": "sample",
            "label_hierarchy": "tone",
            "annotation": "tone",
            "duration": duration_s * 1000,
            "hz_min": 400.0,
            "hz_max": 480.0,
            "sample_rate": sample_rate,
            "channels": 1,
        }
        json_path = audio_path.with_suffix(".json")
        json_path.write_text(json.dumps(metadata), encoding="utf-8")

        # Convert to COCO
        output_dir = tmp_path / "coco"
        result = convert_audio_to_coco(
            input_dir=audio_dir,
            output_dir=output_dir,
            split_ratios=(1.0, 0.0, 0.0),  # All to train
            random_seed=42,
        )

        assert result == output_dir

        # Check output files
        assert (output_dir / "train").exists()
        assert (output_dir / "train" / "_annotations.coco.json").exists()

        # Load and validate COCO dataset
        with open(output_dir / "train" / "_annotations.coco.json") as f:
            coco_data = json.load(f)

        is_valid, errors = validate_coco_dataset(coco_data)
        assert is_valid, f"Validation errors: {errors}"

        # Check content
        assert len(coco_data["images"]) >= 1
        assert len(coco_data["annotations"]) >= 1
        assert len(coco_data["categories"]) >= 1

    def test_empty_directory(self, tmp_path: Path):
        """Test conversion with empty directory."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        output_dir = tmp_path / "coco"

        with pytest.raises(ValueError, match="No audio files"):
            convert_audio_to_coco(
                input_dir=audio_dir,
                output_dir=output_dir,
            )
