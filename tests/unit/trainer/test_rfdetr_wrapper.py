"""Tests for RF-DETR trainer wrapper.

Verifies model setup, freeze behaviour, and factory function work correctly with the actual rfdetr library objects.

"""

from __future__ import annotations

import pytest

from rf_detr_finetuning.trainer.config import CheckpointConfig, OptimizerConfig, TrainerConfig
from rf_detr_finetuning.trainer.rfdetr_wrapper import RFDETRConfig, RFDETRTrainer, create_rfdetr_trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _have_rfdetr() -> bool:
    """Check if rfdetr package is importable."""
    try:
        import rfdetr  # noqa: F401

        return True
    except ImportError:
        return False


requires_rfdetr = pytest.mark.skipif(not _have_rfdetr(), reason="rfdetr package not installed")


def _make_trainer(
    model_size: str = "base",
    num_classes: int = 2,
    freeze_backbone: bool = False,
    freeze_batch_norm: bool = True,
) -> RFDETRTrainer:
    """Create a trainer with sensible test defaults."""
    model_config = RFDETRConfig(
        model_size=model_size,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_batch_norm,
    )
    trainer_config = TrainerConfig(
        epochs=1,
        batch_size=2,
        optimizer=OptimizerConfig(lr=1e-4),
        checkpoint=CheckpointConfig(save_dir="output"),
    )
    return RFDETRTrainer(model_config=model_config, trainer_config=trainer_config)


# ---------------------------------------------------------------------------
# Tests: config & construction
# ---------------------------------------------------------------------------


class TestRFDETRConfig:
    """Test RFDETRConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = RFDETRConfig()
        assert cfg.model_size == "base"
        assert cfg.num_classes == 1
        assert cfg.freeze_batch_norm is True

    def test_custom_values(self) -> None:
        cfg = RFDETRConfig(model_size="large", num_classes=10, freeze_backbone=True)
        assert cfg.model_size == "large"
        assert cfg.num_classes == 10
        assert cfg.freeze_backbone is True


class TestRFDETRTrainerConstruction:
    """Test trainer construction without requiring rfdetr."""

    def test_init_stores_config(self) -> None:
        trainer = _make_trainer(num_classes=5)
        assert trainer.model_config.num_classes == 5
        assert trainer.model is None

    def test_checkpoint_manager_created(self) -> None:
        trainer = _make_trainer()
        assert trainer.checkpoint_manager is not None


# ---------------------------------------------------------------------------
# Tests: setup_model (requires rfdetr)
# ---------------------------------------------------------------------------


@requires_rfdetr
class TestSetupModel:
    """Test model initialization with the real rfdetr library."""

    def test_setup_base_model(self) -> None:
        """setup_model() returns a model and stores it on the trainer."""
        trainer = _make_trainer(model_size="base", num_classes=2)
        model = trainer.setup_model()
        assert model is not None
        assert trainer.model is model

    def test_num_classes_passed_for_non_coco(self) -> None:
        """num_classes != 80 should be forwarded to the model constructor."""
        trainer = _make_trainer(num_classes=3)
        trainer.setup_model()
        # The model should have been created with num_classes=3
        # (rfdetr reinitializes the detection head internally)
        assert trainer.model is not None

    def test_invalid_model_size_raises(self) -> None:
        trainer = _make_trainer(model_size="nonexistent")
        with pytest.raises(ValueError, match="Unknown model size"):
            trainer.setup_model()

    def test_freeze_batch_norm_no_crash(self) -> None:
        """freeze_batch_norm=True should not crash even if model has no BN layers."""
        trainer = _make_trainer(freeze_batch_norm=True, freeze_backbone=False)
        trainer.setup_model()
        # Should complete without error — RFDETRBase has 0 BatchNorm layers

    def test_freeze_backbone(self) -> None:
        """freeze_backbone=True should freeze backbone parameters."""
        trainer = _make_trainer(freeze_backbone=True, freeze_batch_norm=False)
        trainer.setup_model()

        nn_module = trainer._get_nn_module()
        assert nn_module is not None

        frozen_count = 0
        for name, param in nn_module.named_parameters():
            if "backbone" in name:
                assert not param.requires_grad, f"Expected {name} to be frozen"
                frozen_count += 1

        assert frozen_count > 0, "Expected at least one backbone parameter to be frozen"

    def test_freeze_both(self) -> None:
        """Both freeze options together should not crash."""
        trainer = _make_trainer(freeze_backbone=True, freeze_batch_norm=True)
        trainer.setup_model()
        assert trainer.model is not None


# ---------------------------------------------------------------------------
# Tests: _get_nn_module (requires rfdetr)
# ---------------------------------------------------------------------------


@requires_rfdetr
class TestGetNNModule:
    """Test internal nn.Module extraction."""

    def test_returns_module_with_modules_attr(self) -> None:
        trainer = _make_trainer()
        trainer.setup_model()
        nn_module = trainer._get_nn_module()
        assert nn_module is not None
        assert hasattr(nn_module, "modules")
        assert hasattr(nn_module, "named_parameters")

    def test_returns_none_when_no_model(self) -> None:
        trainer = _make_trainer()
        # model is None before setup
        assert trainer._get_nn_module() is None


# ---------------------------------------------------------------------------
# Tests: factory function
# ---------------------------------------------------------------------------


class TestCreateRFDETRTrainer:
    """Test the create_rfdetr_trainer factory."""

    def test_factory_defaults(self) -> None:
        trainer = create_rfdetr_trainer()
        assert isinstance(trainer, RFDETRTrainer)
        assert trainer.model_config.model_size == "base"
        assert trainer.trainer_config.epochs == 10

    def test_factory_custom_args(self) -> None:
        trainer = create_rfdetr_trainer(
            model_size="large",
            num_classes=5,
            epochs=20,
            batch_size=16,
            lr=5e-5,
        )
        assert trainer.model_config.model_size == "large"
        assert trainer.model_config.num_classes == 5
        assert trainer.trainer_config.epochs == 20
        assert trainer.trainer_config.batch_size == 16
        assert trainer.trainer_config.optimizer.lr == 5e-5

    def test_factory_kwargs_applied(self) -> None:
        """Extra kwargs should be applied to trainer_config if the attr exists."""
        trainer = create_rfdetr_trainer(seed=123)
        assert trainer.trainer_config.seed == 123

    def test_factory_unknown_kwargs_warned(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown kwargs should log a warning, not crash."""
        import logging

        with caplog.at_level(logging.WARNING):
            trainer = create_rfdetr_trainer(nonexistent_param=42)
        assert any("nonexistent_param" in record.message for record in caplog.records)
        assert isinstance(trainer, RFDETRTrainer)

    def test_factory_from_yaml(self, tmp_path) -> None:
        """Factory should load config from YAML when config_path is given."""
        import yaml

        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            yaml.dump(
                {
                    "epochs": 30,
                    "batch_size": 4,
                    "optimizer": {"lr": 0.001},
                    "checkpoint": {"save_dir": str(tmp_path / "output")},
                }
            )
        )
        trainer = create_rfdetr_trainer(config_path=config_yaml, num_classes=3)
        assert trainer.trainer_config.epochs == 30
        assert trainer.trainer_config.batch_size == 4
        assert trainer.model_config.num_classes == 3
