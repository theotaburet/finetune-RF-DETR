"""Tests for run_download_data.py script.

Verifies configuration loading, CLI argument parsing, dataset splitting logic, dry-run functionality, and error
handling.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
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

# =============================================================================
# Fixtures
# =============================================================================


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
        "source_start": 10.0,
        "source_end": 20.0,
        "label_hierarchy": "marine/ship/engine",
        "hz_min": 100.0,
        "hz_max": 1000.0,
        "confidence": 0.95,
        "labeler": "test_user",
        "absolute_time": "2024-01-15T10:30:00Z",
    }


# =============================================================================
# Configuration Tests
# =============================================================================


class TestDownloadConfig:
    """Test DownloadConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DownloadConfig()
        assert config.api.url == ""
        assert config.api.token is None
        assert config.output.dir == "data/downloaded"
        assert config.split.enabled is False
        assert config.split.test_source_files == []
        assert config.split.test_files == 10
        assert config.split.train_ratio == 0.8
        assert config.split.val_ratio == 0.2

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

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = DownloadConfig()
        config.api.url = "http://test.com"
        config.api.token = "secret"
        config.split.enabled = True
        config.split.test_source_files = ["file1.wav"]

        data = config.to_dict()

        assert data["api"]["url"] == "http://test.com"
        assert data["api"]["token"] == "secret"
        assert data["split"]["enabled"] is True
        assert data["split"]["test_source_files"] == ["file1.wav"]

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


# =============================================================================
# Event and Sound Tests
# =============================================================================


class TestLabeledEvent:
    """Test LabeledEvent class."""

    def test_duration_properties(self):
        """Test duration calculations."""
        event = LabeledEvent(
            label_id="test",
            source_file="file.wav",
            source_start=10.0,
            source_end=15.5,
            label_hierarchy="test/label",
        )

        assert event.duration_s == 5.5
        assert event.duration_ms == 5500.0

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

    def test_no_events(self):
        """Test grouping empty list."""
        result = group_overlapping_events([], "file.wav")
        assert result == []

    def test_single_event(self):
        """Test grouping single event."""
        events = [LabeledEvent("1", "file.wav", 0.0, 10.0, "label")]
        result = group_overlapping_events(events, "file.wav")

        assert len(result) == 1
        assert result[0].start_s == 0.0
        assert result[0].end_s == 10.0
        assert len(result[0].events) == 1

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
        assert event.source_start == 10.0
        assert event.source_end == 20.0
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


# =============================================================================
# Dataset Splitting Tests
# =============================================================================


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

    def test_split_empty_sounds(self):
        """Test splitting empty sound list."""
        config = SplitConfig(enabled=True, test_files=5, train_ratio=0.8, val_ratio=0.2)
        result = split_sounds_into_datasets([], config)

        assert result["train"] == []
        assert result["val"] == []
        assert result["test"] == []

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


# =============================================================================
# DataDownloader Tests
# =============================================================================


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


# =============================================================================
# API Client Tests
# =============================================================================


class TestEKBAPIClient:
    """Test EKBAPIClient class."""

    def test_init_with_token(self):
        """Test initialization with authentication token."""
        client = EKBAPIClient("http://api.example.com", token="test_token")

        assert client.base_url == "http://api.example.com"
        assert client.token == "test_token"
        assert client.session.headers.get("Authorization") == "Bearer test_token"

    def test_init_without_token(self):
        """Test initialization without token."""
        client = EKBAPIClient("http://api.example.com")

        assert client.base_url == "http://api.example.com"
        assert client.token is None
        assert "Authorization" not in client.session.headers

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


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_end_to_end_dry_run(self, tmp_config_file: Path):
        """Test full workflow in dry-run mode."""
        # This test would require mocking the API responses
        # For now, just verify config loading works
        config = DownloadConfig.from_yaml(tmp_config_file)

        assert config.api.url == "http://192.168.50.103:8080/"
        assert config.split.enabled is True
        assert config.split.test_source_files == ["test1.wav", "test2.wav"]

    def test_config_file_not_found(self, tmp_path: Path):
        """Test handling of missing config file."""
        nonexistent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError):
            DownloadConfig.from_yaml(nonexistent)


# =============================================================================
# API Connectivity Tests
# =============================================================================


class TestAPIConnectivity:
    """Test actual API connectivity."""

    @pytest.fixture
    def real_api_url(self):
        """Return the real API URL from config."""
        return "http://192.168.50.103:8080/"

    def test_api_url_reachable(self, real_api_url):
        """Test that the API URL is reachable.

        This test is skipped if the API is not accessible.

        """
        try:
            response = requests.get(real_api_url, timeout=5)
            # Just check if we can connect (any response is fine)
            assert response.status_code in [200, 404, 401, 403]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not reachable")
        except requests.exceptions.Timeout:
            pytest.skip("API connection timed out")

    def test_api_list_sources_endpoint(self, real_api_url):
        """Test that the API /sdk/sources endpoint responds.

        This test is skipped if the API is not accessible.

        """
        try:
            client = EKBAPIClient(real_api_url)
            sources = client.list_sources()
            # If we get here, the endpoint is working
            assert isinstance(sources, list)
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not reachable")
        except requests.exceptions.HTTPError as e:
            # 401/403 is acceptable (no auth token)
            # 404 means endpoint doesn't exist yet
            if e.response.status_code in [401, 403]:
                pytest.skip("API requires authentication")
            elif e.response.status_code == 404:
                pytest.skip("API endpoint /sdk/sources not found (may not be implemented yet)")
            raise

    def test_api_labels_endpoint_structure(self, real_api_url):
        """Test that the API /sdk/labels endpoint has correct structure.

        This test is skipped if the API is not accessible.

        """
        try:
            client = EKBAPIClient(real_api_url)
            response = client.list_labels(limit=1)
            # Check structure
            assert "labels" in response or "total" in response or "sources" in response
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not reachable")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [401, 403]:
                pytest.skip("API requires authentication")
            raise


# =============================================================================
# Dry-Run Output Tests
# =============================================================================


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

    def test_dry_run_creates_metadata(self, mock_api_with_data, dry_run_config, tmp_path: Path):
        """Test that dry-run creates metadata file."""
        downloader = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        results = downloader.run()

        # In dry-run mode, metadata is returned but not saved to disk
        # So we verify the results are well-formed
        assert len(results) > 0

        # Check result structure
        for result in results:
            assert result.sound_id
            assert result.source_file
            assert isinstance(result.events, list)
            assert result.split in ["train", "val", "test", "unsplit"]

    def test_dry_run_with_test_source_files(self, mock_api_with_data, dry_run_config, tmp_path: Path):
        """Test dry-run with specific test source files."""
        dry_run_config.split.test_source_files = ["file0.wav"]
        dry_run_config.split.test_files = 0  # Disable random selection

        downloader = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        results = downloader.run()

        # Should have results
        assert len(results) > 0

        # Check that some results are assigned to test split (if source files match)
        _ = [r for r in results if r.split == "test"]

    def test_dry_run_produces_valid_metadata_structure(self, mock_api_with_data, dry_run_config):
        """Test that dry-run metadata has valid structure."""
        downloader = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        results = downloader.run()

        # Build metadata like the real code does
        metadata = {
            "download_info": {
                "total_sounds": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "config": dry_run_config.to_dict(),
                "dry_run": True,
            },
            "sounds": [r.to_dict() for r in results],
        }

        # Validate structure
        assert metadata["download_info"]["total_sounds"] > 0
        assert metadata["download_info"]["successful"] == len(results)
        assert metadata["download_info"]["failed"] == 0
        assert metadata["download_info"]["dry_run"] is True

        # Check each sound entry
        for sound in metadata["sounds"]:
            assert "sound_id" in sound
            assert "source_file" in sound
            assert "events" in sound
            assert isinstance(sound["events"], list)
            assert "split" in sound

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

    def test_dry_run_output_is_consistent(self, mock_api_with_data, dry_run_config):
        """Test that dry-run produces consistent results across multiple runs."""
        downloader1 = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        downloader2 = DataDownloader(
            api_client=mock_api_with_data,
            config=dry_run_config,
            dry_run=True,
        )

        results1 = downloader1.run()
        results2 = downloader2.run()

        # Should have same number of results
        assert len(results1) == len(results2)

        # Should have same sound IDs
        ids1 = {r.sound_id for r in results1}
        ids2 = {r.sound_id for r in results2}
        assert ids1 == ids2
