"""Custom samplers for deterministic and balanced sampling.

Provides samplers for:
- Deterministic iteration order (reproducibility)
- Class-balanced sampling (handle imbalanced datasets)

"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import numpy as np
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class DeterministicSampler(Sampler[int]):
    """Sampler that provides deterministic iteration order.

    Useful for reproducible training and debugging.
    Shuffles once at construction based on seed, then iterates in fixed order.

    Args:
        data_source: Dataset to sample from.
        seed: Random seed for shuffling.
        shuffle: Whether to shuffle indices.

    """

    def __init__(
        self,
        data_source: Dataset,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        """Initialize reproducible random sampler.

        Args:
            data_source: Dataset to sample from.
            seed: Random seed for reproducibility.
            shuffle: Whether to shuffle data.

        """
        super().__init__(data_source)
        self.data_source = data_source
        self.seed = seed
        self.shuffle = shuffle

        # Generate fixed order
        n = len(data_source)
        self.indices = list(range(n))

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)

    def __iter__(self) -> Iterator[int]:
        """Return iterator over indices."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.indices)

    def reset(self, new_seed: int | None = None) -> None:
        """Reset sampler with optional new seed.

        Args:
            new_seed: New random seed. If None, uses original seed.

        """
        if new_seed is not None:
            self.seed = new_seed

        if self.shuffle:
            self.indices = list(range(len(self.data_source)))
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.indices)


class BalancedClassSampler(Sampler[int]):
    """Sampler that balances classes by oversampling minority classes.

    Each epoch samples equally from all classes, repeating minority
    class samples as needed.

    Args:
        data_source: Dataset to sample from.
        get_label_fn: Function to extract label from dataset item.
        seed: Random seed.
        samples_per_class: Samples per class per epoch.
            If None, uses maximum class count.

    """

    def __init__(
        self,
        data_source: Dataset,
        get_label_fn: Callable[..., int] | None = None,
        seed: int = 42,
        samples_per_class: int | None = None,
    ) -> None:
        """Initialize balanced sampler.

        Args:
            data_source: Dataset to sample from.
            get_label_fn: Function to extract label from dataset item.
            seed: Random seed.
            samples_per_class: Number of samples per class per epoch.

        """
        super().__init__(data_source)
        self.data_source = data_source
        self.get_label_fn = get_label_fn
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build class indices
        self.class_to_indices: dict[int, list[int]] = {}
        self._build_class_indices()

        # Determine samples per class
        if samples_per_class is not None:
            self.samples_per_class = samples_per_class
        else:
            # Use maximum class count
            self.samples_per_class = max(len(indices) for indices in self.class_to_indices.values())

        logger.info(
            f"BalancedClassSampler: {len(self.class_to_indices)} classes, {self.samples_per_class} samples per class"
        )

    def _build_class_indices(self) -> None:
        """Build mapping from class label to dataset indices."""
        for i in range(len(self.data_source)):
            if self.get_label_fn is not None:
                label = self.get_label_fn(self.data_source[i])
            else:
                from rf_detr_finetuning.dataloader.utils import extract_default_label

                label = extract_default_label(self.data_source[i])

            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(i)

    def __iter__(self) -> Iterator[int]:
        """Generate balanced sample indices."""
        indices = []

        for class_label, class_indices in self.class_to_indices.items():
            # Sample with replacement if needed
            n_class = len(class_indices)
            if n_class >= self.samples_per_class:
                sampled = self.rng.choice(
                    class_indices,
                    size=self.samples_per_class,
                    replace=False,
                )
            else:
                # Oversample minority class
                sampled = self.rng.choice(
                    class_indices,
                    size=self.samples_per_class,
                    replace=True,
                )
            indices.extend(sampled.tolist())

        # Shuffle final indices
        self.rng.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        return len(self.class_to_indices) * self.samples_per_class
