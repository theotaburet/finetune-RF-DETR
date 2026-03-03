"""Dataset splitting utilities.

Provides train/val/test splitting with stratification support, deterministic splits for reproducibility, and multi-
loader creation.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting.

    Attributes:
        train_ratio: Fraction of data for training (0-1).
        val_ratio: Fraction of data for validation (0-1).
        test_ratio: Fraction of data for testing (0-1).
        seed: Random seed for reproducibility.
        stratify_by: Key to stratify by (e.g., "category", "source_file").
        min_samples_per_class: Minimum samples per class for stratification.

    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by: str | None = None
    """Key to stratify by (e.g., "category", "source_file").

    Note:
        Currently informational only. Use ``get_label_fn`` in
        :func:`create_stratified_split` to control stratification behaviour.

    """
    min_samples_per_class: int = 1

    def __post_init__(self) -> None:
        """Validate ratios sum to 1."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


def split_dataset(
    dataset: Dataset,
    config: SplitConfig,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test subsets.

    Args:
        dataset: PyTorch dataset to split.
        config: Split configuration.

    Returns:
        Tuple of (train_subset, val_subset, test_subset).

    """
    n = len(dataset)
    indices = np.arange(n)

    # Set seed for reproducibility
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)

    # Calculate split points
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train : n_train + n_val].tolist()
    test_indices = indices[n_train + n_val :].tolist()

    logger.info(f"Split {n} samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def create_stratified_split(
    dataset: Dataset,
    config: SplitConfig,
    get_label_fn: Callable[..., int] | None = None,
) -> tuple[Subset, Subset, Subset]:
    """Create stratified split preserving class distribution.

    Args:
        dataset: Dataset to split.
        config: Split configuration.
        get_label_fn: Function to extract label from dataset item.
            If None, assumes dataset[i][1].labels[0] exists.

    Returns:
        Tuple of (train_subset, val_subset, test_subset).

    """
    n = len(dataset)
    rng = np.random.default_rng(config.seed)

    # Extract labels for stratification
    labels = []
    for i in range(n):
        if get_label_fn is not None:
            label = get_label_fn(dataset[i])
        else:
            from rf_detr_finetuning.dataloader.utils import extract_default_label

            label = extract_default_label(dataset[i])
        labels.append(label)

    labels = np.array(labels)

    # Group indices by label
    unique_labels = np.unique(labels)
    label_to_indices: dict[int, list[int]] = {label: np.where(labels == label)[0].tolist() for label in unique_labels}

    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        rng.shuffle(indices)
        n_label = len(indices)

        n_train = max(config.min_samples_per_class, int(n_label * config.train_ratio))
        n_val = max(1, int(n_label * config.val_ratio))

        # Adjust if we don't have enough samples
        if n_train + n_val > n_label:
            n_train = n_label - 1
            n_val = 1

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    # Shuffle within each split
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    logger.info(
        f"Stratified split {n} samples across {len(unique_labels)} classes: "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def create_train_val_test_loaders(
    dataset: Dataset,
    config: SplitConfig,
    batch_size: int = 8,
    num_workers: int = 4,
    collate_fn: Callable[..., Any] | None = None,
    pin_memory: bool = True,
    stratified: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test splits.

    Args:
        dataset: Dataset to split and load.
        config: Split configuration.
        batch_size: Batch size for all loaders.
        num_workers: Number of worker processes.
        collate_fn: Custom collate function.
        pin_memory: Pin memory for GPU transfer.
        stratified: Use stratified split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    """
    if stratified:
        train_set, val_set, test_set = create_stratified_split(dataset, config)
    else:
        train_set, val_set, test_set = split_dataset(dataset, config)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
