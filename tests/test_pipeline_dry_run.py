"""Tests for pipeline dry-run mode.

Verifies that --dry-run validates configuration and reports planned actions without actually executing heavy operations
(training, inference, etc.).

"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from run_pipeline import (
    EvaluateStep,
    InferStep,
    Pipeline,
    PipelineConfig,
    PreprocessStep,
    SplitStep,
    TrainStep,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path: Path) -> PipelineConfig:
    """Build a minimal PipelineConfig pointing at tmp directories."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    chunking_cfg = tmp_path / "chunking.yaml"
    chunking_cfg.write_text(
        yaml.dump(
            {
                "fft": {"hop_ms": 10.0, "fft_ms": 25.0, "n_mels": 128},
                "chunking": {"target_size": 640, "overlap_ms": 1280.0},
            }
        )
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "images").mkdir()

    # Minimal COCO annotation file
    coco = {
        "images": [{"id": i, "file_name": f"images/img_{i:04d}.png", "width": 640, "height": 640} for i in range(10)],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 0,
                "bbox": [10, 10, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            }
            for i in range(10)
        ],
        "categories": [{"id": 0, "name": "event"}],
    }
    with open(processed_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    split_dir = tmp_path / "split_dataset"
    split_dir.mkdir()
    for sub in ("train", "valid", "test"):
        (split_dir / sub).mkdir()
        with open(split_dir / sub / "_annotations.coco.json", "w") as f:
            json.dump(coco, f)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = PipelineConfig(
        name="test-dry-run",
        description="Dry-run test pipeline",
    )
    config.preprocess.audio_dir = str(audio_dir)
    config.preprocess.metadata_dir = str(metadata_dir)
    config.preprocess.output_dir = str(processed_dir)
    config.preprocess.chunking_config = str(chunking_cfg)
    config.preprocess.enabled = True

    config.split.input_dir = str(processed_dir)
    config.split.output_dir = str(split_dir)
    config.split.enabled = True

    config.train.dataset_dir = str(split_dir)
    config.train.output_dir = str(output_dir)
    config.train.enabled = True

    config.evaluate.test_dir = str(split_dir / "test")
    config.evaluate.output_dir = str(output_dir / "eval")
    config.evaluate.weights = str(output_dir / "checkpoint_best.pth")
    config.evaluate.enabled = True

    config.infer.audio_dir = str(audio_dir)
    config.infer.output_dir = str(output_dir / "predictions")
    config.infer.weights = str(output_dir / "checkpoint_best.pth")
    config.infer.chunking_config = str(chunking_cfg)
    config.infer.enabled = True

    return config


@pytest.fixture
def tmp_config_yaml(tmp_path: Path, tmp_config: PipelineConfig) -> Path:
    """Save the tmp_config as a YAML file and return the path."""
    config_path = tmp_path / "pipeline.yaml"
    tmp_config.to_yaml(config_path)
    return config_path


# ---------------------------------------------------------------------------
# Unit tests: individual step dry-run
# ---------------------------------------------------------------------------


class TestPreprocessStepDryRun:
    """PreprocessStep dry-run returns report without processing files."""

    def test_returns_true(self, tmp_config: PipelineConfig) -> None:
        step = PreprocessStep(tmp_config, dry_run=True)
        assert step.run() is True

    def test_populates_results(self, tmp_config: PipelineConfig) -> None:
        step = PreprocessStep(tmp_config, dry_run=True)
        step.run()
        assert "audio_files" in step.results
        assert "output_dir" in step.results

    def test_no_images_created(self, tmp_config: PipelineConfig) -> None:
        output_images = Path(tmp_config.preprocess.output_dir) / "images"
        before = set(output_images.iterdir()) if output_images.exists() else set()
        step = PreprocessStep(tmp_config, dry_run=True)
        step.run()
        after = set(output_images.iterdir()) if output_images.exists() else set()
        assert before == after, "Dry-run should not create image files"


class TestSplitStepDryRun:
    """SplitStep dry-run reports split counts without moving files."""

    def test_returns_true(self, tmp_config: PipelineConfig) -> None:
        step = SplitStep(tmp_config, dry_run=True)
        assert step.run() is True

    def test_reports_split_counts(self, tmp_config: PipelineConfig) -> None:
        step = SplitStep(tmp_config, dry_run=True)
        step.run()
        assert "total_images" in step.results
        assert step.results["total_images"] == 10

    def test_no_files_moved(self, tmp_config: PipelineConfig) -> None:
        output_dir = Path(tmp_config.split.output_dir)
        before = {p.name for p in output_dir.rglob("*.png")}
        step = SplitStep(tmp_config, dry_run=True)
        step.run()
        after = {p.name for p in output_dir.rglob("*.png")}
        assert before == after, "Dry-run should not copy/move image files"


class TestTrainStepDryRun:
    """TrainStep dry-run reports training config without starting training."""

    def test_returns_true(self, tmp_config: PipelineConfig) -> None:
        step = TrainStep(tmp_config, dry_run=True)
        assert step.run() is True

    def test_reports_training_plan(self, tmp_config: PipelineConfig) -> None:
        step = TrainStep(tmp_config, dry_run=True)
        step.run()
        assert "epochs" in step.results
        assert "batch_size" in step.results
        assert "model_size" in step.results

    def test_no_checkpoints_created(self, tmp_config: PipelineConfig) -> None:
        output_dir = Path(tmp_config.train.output_dir)
        before = set(output_dir.glob("*.pth"))
        step = TrainStep(tmp_config, dry_run=True)
        step.run()
        after = set(output_dir.glob("*.pth"))
        assert before == after, "Dry-run should not create checkpoint files"


class TestEvaluateStepDryRun:
    """EvaluateStep dry-run reports evaluation plan."""

    def test_returns_true(self, tmp_config: PipelineConfig) -> None:
        step = EvaluateStep(tmp_config, dry_run=True)
        assert step.run() is True

    def test_reports_eval_config(self, tmp_config: PipelineConfig) -> None:
        step = EvaluateStep(tmp_config, dry_run=True)
        step.run()
        assert "confidence_threshold" in step.results
        assert "test_images" in step.results


class TestInferStepDryRun:
    """InferStep dry-run reports inference plan."""

    def test_returns_true(self, tmp_config: PipelineConfig) -> None:
        step = InferStep(tmp_config, dry_run=True)
        assert step.run() is True

    def test_reports_infer_plan(self, tmp_config: PipelineConfig) -> None:
        step = InferStep(tmp_config, dry_run=True)
        step.run()
        assert "confidence_threshold" in step.results
        assert "output_dir" in step.results


# ---------------------------------------------------------------------------
# Integration test: full pipeline dry-run
# ---------------------------------------------------------------------------


class TestPipelineDryRun:
    """Pipeline.run(dry_run=True) should succeed without GPU or real data."""

    def test_all_steps_dry_run_succeeds(self, tmp_config: PipelineConfig) -> None:
        pipeline = Pipeline(tmp_config)
        result = pipeline.run(run_all=True, dry_run=True)
        assert result is True

    def test_all_steps_produce_results(self, tmp_config: PipelineConfig) -> None:
        pipeline = Pipeline(tmp_config)
        pipeline.run(run_all=True, dry_run=True)
        assert len(pipeline.step_results) == 5
        for step_name in ("preprocess", "split", "train", "evaluate", "infer"):
            assert step_name in pipeline.step_results, f"Missing results for {step_name}"

    def test_single_step_dry_run(self, tmp_config: PipelineConfig) -> None:
        pipeline = Pipeline(tmp_config)
        result = pipeline.run(preprocess=True, dry_run=True)
        assert result is True
        assert "preprocess" in pipeline.step_results

    def test_no_side_effects(self, tmp_config: PipelineConfig) -> None:
        """Dry-run should not create output artifacts."""
        output_dir = Path(tmp_config.train.output_dir)
        before_pth = set(output_dir.glob("*.pth"))

        pipeline = Pipeline(tmp_config)
        pipeline.run(run_all=True, dry_run=True)

        after_pth = set(output_dir.glob("*.pth"))
        assert before_pth == after_pth, "Dry-run should not create checkpoint files"

    def test_dry_run_from_yaml(self, tmp_config_yaml: Path) -> None:
        """Pipeline loaded from YAML runs dry-run successfully."""
        config = PipelineConfig.from_yaml(tmp_config_yaml)
        pipeline = Pipeline(config)
        result = pipeline.run(run_all=True, dry_run=True)
        assert result is True
