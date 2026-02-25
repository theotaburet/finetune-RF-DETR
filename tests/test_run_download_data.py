"""Tests for run_download_data.py script.

Verifies configuration loading, CLI argument parsing, dataset splitting logic, dry-run functionality, and error
handling.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from run_download_data import (
    DataDownloader,
    DownloadConfig,
    DownloadResult,
    EKBAPIClient,
    LabeledEvent,
    SoundSegment,
    SplitConfig,
    group_overlapping_events,
    merge_config_with_args,
    parse_label_response,
    split_sounds_into_datasets,
    validate_config,
)


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Path:
    """Create a temporary YAML config file."""
    config_data = {
        "api": {
            "url": "http://192.168.50.103:8080/",
            "token": "test_token_123",
        },
        "output": {
            "dir": str(tmp_path / "downloaded"),
            "train_dir": "train",
            "val_dir": "valid",
            "test_dir": "test",
        },
        "split": {
            "enabled": True,
            "test_source_files": ["test1.wav", "test2.wav"],
            "test_files": 5,
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "seed": 42,
            "stratify_val": True,
        },
        "filters": {
            "sources": ["source1", "source2"],
            "label_hierarchy": "marine/ship",
            "labeler": "test_user",
            "from_date": "2024-01-01",
            "to_date": "2024-12-31",
            "confidence_min": 0.5,
            "confidence_max": 1.0,
            "min_duration": 100.0,
            "max_frequency": 20000,
        },
        "limits": {
            "max_labels": 1000,
        },
        "advanced": {
            "seed": 42,
            "batch_size": 50,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def sample_events() -> list[LabeledEvent]:
    """Create sample labeled events for testing."""
    return [
        LabeledEvent(
            label_id="uuid1",
            source_file="audio/file1.wav",
            source_start=0.0,
            source_end=10.0,
            label_hierarchy="marine/ship",
            hz_min=100.0,
            hz_max=1000.0,
        ),
        LabeledEvent(
            label_id="uuid2",
            source_file="audio/file1.wav",
            source_start=4.5,
            source_end=5.5,
            label_hierarchy="marine/sonar",
            hz_min=500.0,
            hz_max=2000.0,
        ),
        LabeledEvent(
            label_id="uuid3",
            source_file="audio/file1.wav",
            source_start=12.5,
            source_end=13.0,
            label_hierarchy="marine/whale",
            hz_min=50.0,
            hz_max=500.0,
        ),
        LabeledEvent(
            label_id="uuid4",
            source_file="audio/file2.wav",
            source_start=0.0,
            source_end=5.0,
            label_hierarchy="marine/ship",
        ),
    ]


@pytest.fixture
def sample_label_response() -> dict:
    """Create sample API label response."""
    return {
        "uuid": "test-uuid-123",
        "source_path": "audio/test.wav",
        "source_start": 10000.0,  # milliseconds (API field name is misleading)
        "source_end": 20000.0,  # milliseconds
        "label_hierarchy": "marine/ship/engine",
        "hz_min": 100.0,
        "hz_max": 1000.0,
        "confidence": 0.95,
        "labeler": "test_user",
        "absolute_time": "2024-01-15T10:30:00Z",
    }


class TestDownloadConfig:
    """Test DownloadConfig class."""

    def test_load_from_yaml(self, tmp_config_file: Path):
        """Test loading configuration from YAML file."""
        config = DownloadConfig.from_yaml(tmp_config_file)

        assert config.api.url == "http://192.168.50.103:8080/"
        assert config.api.token == "test_token_123"
        assert config.split.enabled is True
        assert config.split.test_source_files == ["test1.wav", "test2.wav"]
        assert config.split.test_files == 5
        assert config.split.train_ratio == 0.8
        assert config.filters.sources == ["source1", "source2"]
        assert config.filters.label_hierarchy == "marine/ship"
        assert config.limits.max_labels == 1000
        assert config.advanced.batch_size == 50

    def test_split_config_validation(self):
        """Test SplitConfig validation."""
        # Valid config
        config = SplitConfig(enabled=True, train_ratio=0.7, val_ratio=0.3)
        config.__post_init__()  # Should not raise

        # Invalid ratio sum
        with pytest.raises(ValueError, match="must equal 1.0"):
            SplitConfig(enabled=True, train_ratio=0.5, val_ratio=0.3).__post_init__()

        # Negative test files
        with pytest.raises(ValueError, match="must be >= 0"):
            SplitConfig(enabled=True, test_files=-1).__post_init__()


class TestMergeConfigWithArgs:
    """Test merging CLI args with config."""

    def test_api_settings_override(self):
        """Test API settings are overridden by CLI args."""
        config = DownloadConfig()
        config.api.url = "http://original.com"
        config.api.token = "original_token"

        args = argparse.Namespace(
            api_url="http://override.com",
            token="override_token",
            output_dir=None,
            split=None,
            test_source_files=None,
            test_files=None,
            train_ratio=None,
            val_ratio=None,
            seed=None,
            source=None,
            label_hierarchy=None,
            labeler=None,
            from_date=None,
            to_date=None,
            confidence_min=None,
            confidence_max=None,
            min_duration=None,
            max_frequency=None,
            max_labels=None,
        )

        result = merge_config_with_args(config, args)

        assert result.api.url == "http://override.com"
        assert result.api.token == "override_token"

    def test_split_settings_override(self):
        """Test split settings are overridden by CLI args."""
        config = DownloadConfig()

        args = argparse.Namespace(
            api_url=None,
            token=None,
            output_dir=None,
            split=True,
            test_source_files="file1.wav,file2.wav",
            test_files=20,
            train_ratio=0.7,
            val_ratio=0.3,
            seed=123,
            source=None,
            label_hierarchy=None,
            labeler=None,
            from_date=None,
            to_date=None,
            confidence_min=None,
            confidence_max=None,
            min_duration=None,
            max_frequency=None,
            max_labels=None,
        )

        result = merge_config_with_args(config, args)

        assert result.split.enabled is True
        assert result.split.test_source_files == ["file1.wav", "file2.wav"]
        assert result.split.test_files == 20
        assert result.split.train_ratio == 0.7
        assert result.split.val_ratio == 0.3
        assert result.split.seed == 123


class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = DownloadConfig()
        config.api.url = "http://test.com"
        config.api.token = "token123"

        errors = validate_config(config)
        assert errors == []

    def test_missing_api_url(self):
        """Test validation catches missing API URL."""
        config = DownloadConfig()
        config.api.token = "token123"

        errors = validate_config(config)
        assert any("API URL" in e for e in errors)

    def test_missing_token(self):
        """Test validation catches missing token."""
        config = DownloadConfig()
        config.api.url = "http://test.com"

        errors = validate_config(config)
        assert any("token" in e for e in errors)

    def test_invalid_split_ratios(self):
        """Test validation catches invalid split ratios."""
        config = DownloadConfig()
        config.api.url = "http://test.com"
        config.api.token = "token123"
        config.split.enabled = True
        config.split.train_ratio = 0.5
        config.split.val_ratio = 0.3  # Sum is 0.8, not 1.0

        errors = validate_config(config)
        assert any("must equal 1.0" in e for e in errors)


class TestLabeledEvent:
    """Test LabeledEvent class."""

    def test_overlaps(self):
        """Test overlap detection."""
        event1 = LabeledEvent("1", "file.wav", 0.0, 10.0, "label1")
        event2 = LabeledEvent("2", "file.wav", 5.0, 15.0, "label2")
        event3 = LabeledEvent("3", "file.wav", 10.1, 20.0, "label3")

        assert event1.overlaps(event2) is True
        assert event2.overlaps(event1) is True
        assert event1.overlaps(event3) is False
        assert event3.overlaps(event1) is False


class TestGroupOverlappingEvents:
    """Test event grouping logic."""

    def test_overlapping_events_grouped(self, sample_events):
        """Test that overlapping events are grouped together."""
        # Use only file1 events which have overlaps
        file1_events = [e for e in sample_events if e.source_file == "audio/file1.wav"]
        result = group_overlapping_events(file1_events, "audio/file1.wav")

        # Should have 2 groups: [event1+event2] and [event3]
        assert len(result) == 2

        # First group should contain events 1 and 2 (overlapping)
        assert len(result[0].events) == 2
        assert result[0].start_s == 0.0  # Min of all starts
        assert result[0].end_s == 10.0  # Max of all ends

        # Second group should contain only event 3
        assert len(result[1].events) == 1
        assert result[1].events[0].label_id == "uuid3"

    def test_non_overlapping_events_separate(self, sample_events):
        """Test that non-overlapping events are in separate groups."""
        file2_events = [e for e in sample_events if e.source_file == "audio/file2.wav"]
        result = group_overlapping_events(file2_events, "audio/file2.wav")

        assert len(result) == 1
        assert result[0].events[0].label_id == "uuid4"


class TestParseLabelResponse:
    """Test parsing API label responses."""

    def test_parse_full_response(self, sample_label_response):
        """Test parsing complete label response."""
        event = parse_label_response(sample_label_response)

        assert event.label_id == "test-uuid-123"
        assert event.source_file == "audio/test.wav"
        assert event.source_start == 10.0  # 10000 ms / 1000
        assert event.source_end == 20.0  # 20000 ms / 1000
        assert event.label_hierarchy == "marine/ship/engine"
        assert event.hz_min == 100.0
        assert event.hz_max == 1000.0
        assert event.confidence == 0.95
        assert event.labeler == "test_user"
        assert event.absolute_time == "2024-01-15T10:30:00Z"

    def test_parse_minimal_response(self):
        """Test parsing minimal label response."""
        minimal = {"uuid": "minimal-uuid"}
        event = parse_label_response(minimal)

        assert event.label_id == "minimal-uuid"
        assert event.source_file == ""
        assert event.label_hierarchy == "unknown"
        assert event.confidence == 1.0


class TestSplitSoundsIntoDatasets:
    """Test dataset splitting logic."""

    def create_sample_sounds(self, n: int) -> list[SoundSegment]:
        """Helper to create sample sounds."""
        sounds = []
        for i in range(n):
            event = LabeledEvent(
                label_id=f"uuid{i}",
                source_file=f"audio/file{i % 5}.wav",
                source_start=float(i),
                source_end=float(i + 1),
                label_hierarchy=f"label{i % 3}",
            )
            sounds.append(SoundSegment(events=[event], source_file=f"audio/file{i % 5}.wav"))
        return sounds

    def test_split_random_selection(self):
        """Test random test file selection."""
        sounds = self.create_sample_sounds(20)
        config = SplitConfig(enabled=True, test_files=5, train_ratio=0.8, val_ratio=0.2)

        result = split_sounds_into_datasets(sounds, config, dry_run=True)

        assert len(result["test"]) == 5
        assert len(result["train"]) == 12  # 80% of 15
        assert len(result["val"]) == 3  # 20% of 15

    def test_split_with_source_files(self):
        """Test test set from specified source files."""
        sounds = [
            SoundSegment(
                events=[LabeledEvent("1", "audio/test1.wav", 0, 10, "label")],
                source_file="audio/test1.wav",
            ),
            SoundSegment(
                events=[LabeledEvent("2", "audio/test2.wav", 0, 10, "label")],
                source_file="audio/test2.wav",
            ),
            SoundSegment(
                events=[LabeledEvent("3", "audio/other.wav", 0, 10, "label")],
                source_file="audio/other.wav",
            ),
        ]
        config = SplitConfig(
            enabled=True,
            test_source_files=["test1.wav", "test2.wav"],
            train_ratio=1.0,  # All remaining go to train
            val_ratio=0.0,
        )

        result = split_sounds_into_datasets(sounds, config, dry_run=True)

        # Should find both test files
        assert len(result["test"]) == 2
        assert len(result["train"]) == 1
        assert len(result["val"]) == 0

    def test_split_missing_source_files(self, capsys):
        """Test warning when specified test files are not found."""
        sounds = [
            SoundSegment(
                events=[LabeledEvent("1", "audio/exists.wav", 0, 10, "label")],
                source_file="audio/exists.wav",
            ),
        ]
        config = SplitConfig(
            enabled=True,
            test_source_files=["exists.wav", "missing.wav"],
            train_ratio=1.0,
            val_ratio=0.0,
        )

        result = split_sounds_into_datasets(sounds, config, dry_run=True)

        # Should only find one file
        assert len(result["test"]) == 1

    def test_split_stratification(self):
        """Test stratified validation split."""
        # Create sounds with balanced classes
        sounds = []
        for i in range(20):
            event = LabeledEvent(
                label_id=f"uuid{i}",
                source_file=f"audio/file{i}.wav",
                source_start=float(i),
                source_end=float(i + 1),
                label_hierarchy="class_a" if i < 10 else "class_b",
            )
            sounds.append(SoundSegment(events=[event], source_file=f"audio/file{i}.wav"))

        config = SplitConfig(
            enabled=True,
            test_files=0,
            train_ratio=0.8,
            val_ratio=0.2,
            stratify_val=True,
        )

        result = split_sounds_into_datasets(sounds, config, dry_run=True)

        # Check that val set has both classes
        val_labels = set(s.events[0].label_hierarchy for s in result["val"])
        assert len(val_labels) >= 1  # Should have at least one class


class TestDataDownloader:
    """Test DataDownloader class."""

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client."""
        client = MagicMock(spec=EKBAPIClient)
        client.list_sources.return_value = ["source1", "source2"]
        return client

    @pytest.fixture
    def sample_config(self, tmp_path: Path):
        """Create sample config."""
        config = DownloadConfig()
        config.api.url = "http://test.com"
        config.api.token = "token"
        config.output.dir = str(tmp_path / "output")
        return config

    def test_init_creates_directories(self, mock_api_client, sample_config):
        """Test that initialization creates output directories."""
        DataDownloader(mock_api_client, sample_config, dry_run=False)

        assert Path(sample_config.output.dir).exists()

    def test_dry_run_does_not_create_directories(self, mock_api_client, sample_config):
        """Test that dry-run mode doesn't create directories."""
        output_dir = sample_config.output.dir

        DataDownloader(mock_api_client, sample_config, dry_run=True)

        # Directory should not exist in dry-run mode
        assert not Path(output_dir).exists()


class TestEKBAPIClient:
    """Test EKBAPIClient class."""

    @patch("run_download_data.requests.Session.get")
    def test_list_sources(self, mock_get):
        """Test listing sources from API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sources": [{"name": "source1"}, {"name": "source2"}],
        }
        mock_get.return_value = mock_response

        client = EKBAPIClient("http://api.example.com")
        sources = client.list_sources()

        assert sources == ["source1", "source2"]


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_config_file_not_found(self, tmp_path: Path):
        """Test handling of missing config file."""
        nonexistent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError):
            DownloadConfig.from_yaml(nonexistent)


class TestDryRunOutput:
    """Test that dry-run produces valid output and metadata."""

    @pytest.fixture
    def mock_api_with_data(self):
        """Create mock API client with sample data."""
        client = MagicMock(spec=EKBAPIClient)

        # Mock list_sources
        client.list_sources.return_value = ["source1", "source2"]

        # Mock fetch_all_labels to return sample labels
        sample_labels = [
            {
                "uuid": f"label-{i}",
                "source_path": f"audio/file{i % 3}.wav",
                "source_start": float(i * 10),
                "source_end": float(i * 10 + 5),
                "label_hierarchy": f"class{i % 2}",
                "confidence": 0.9,
            }
            for i in range(10)
        ]

        def mock_fetch_all(**kwargs):
            yield from sample_labels

        client.fetch_all_labels = mock_fetch_all

        # Mock list_labels for pagination
        client.list_labels.return_value = {
            "labels": sample_labels[:5],
            "total": len(sample_labels),
        }

        # Mock retry stats (no retries during dry-run)
        client.get_retry_stats.return_value = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_after_retries": 0,
        }

        return client

    @pytest.fixture
    def dry_run_config(self, tmp_path: Path):
        """Create config for dry-run testing."""
        config = DownloadConfig()
        config.api.url = "http://test.com"
        config.api.token = "token"
        config.output.dir = str(tmp_path / "dryrun_output")
        config.split.enabled = True
        config.split.test_files = 3
        config.split.train_ratio = 0.8
        config.split.val_ratio = 0.2
        config.advanced.batch_size = 5
        return config

    def test_dry_run_produces_results(self, mock_api_with_data, dry_run_config, tmp_path: Path):
        """Test that dry-run produces download results without errors."""
        downloader = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        # Run the download process
        results = downloader.run()

        # Should have results
        assert len(results) > 0
        assert all(isinstance(r, DownloadResult) for r in results)

        # All should be marked as successful in dry-run
        assert all(r.success for r in results)

        # Check result structure
        for result in results:
            assert result.sound_id
            assert result.source_file
            assert isinstance(result.events, list)
            assert result.split in ["train", "val", "test", "unsplit"]

    def test_dry_run_does_not_modify_filesystem(self, mock_api_with_data, dry_run_config, tmp_path: Path):
        """Test that dry-run does not create any files or directories."""
        output_dir = Path(dry_run_config.output.dir)

        # Ensure directory doesn't exist
        assert not output_dir.exists()

        downloader = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        downloader.run()

        # Directory should still not exist
        assert not output_dir.exists()

        # No files should be created
        assert len(list(tmp_path.glob("**/*"))) == 0


class TestExtractLeafName:
    """Tests for extract_leaf_name()."""

    @pytest.mark.parametrize(
        ("hierarchy", "expected"),
        [
            ("Marine mammals + Whales + Humpback whale", "Humpback whale"),
            ("whistles > odontoceti", "odontoceti"),
            ("single_label", "single_label"),
            ("", ""),
            ("  spaced  ", "spaced"),
            ("a + b + c + d", "d"),
        ],
    )
    def test_extract_leaf_name(self, hierarchy, expected):
        """Extract the leaf class name from various hierarchy formats."""
        from run_download_data import extract_leaf_name

        assert extract_leaf_name(hierarchy) == expected


class TestDiscoverClassNamesFromMetadata:
    """Tests for discover_class_names_from_metadata()."""

    def test_discovers_sorted_unique_classes(self, tmp_path):
        """Discover unique leaf class names from download metadata."""
        import json

        from run_download_data import discover_class_names_from_metadata

        metadata = {
            "sounds": [
                {
                    "events": [
                        {"label_hierarchy": "Marine + Whales + Blue whale"},
                        {"label_hierarchy": "Marine + Whales + Humpback whale"},
                    ]
                },
                {
                    "events": [
                        {"label_hierarchy": "Marine + Whales + Blue whale"},
                        {"label_hierarchy": "Anthropogenic + Ship noise"},
                    ]
                },
            ]
        }
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(metadata))

        result = discover_class_names_from_metadata(path)
        assert result == ["Blue whale", "Humpback whale", "Ship noise"]

    def test_empty_metadata(self, tmp_path):
        """Return empty list when metadata has no events."""
        import json

        from run_download_data import discover_class_names_from_metadata

        path = tmp_path / "metadata.json"
        path.write_text(json.dumps({"sounds": []}))

        assert discover_class_names_from_metadata(path) == []
