"""Dataloader module for RF-DETR audio detection pipeline.

Provides dataset classes, collate functions, and train/val/test splitting.

"""

from rf_detr_finetuning.dataloader.coco_export import (
    CategoryRegistry,
    COCODatasetBuilder,
    convert_audio_to_coco,
    parse_frequency_bins,
    validate_coco_dataset,
)
from rf_detr_finetuning.dataloader.collate import (
    collate_detections,
    collate_with_targets,
)
from rf_detr_finetuning.dataloader.dataset import (
    AudioChunkDataset,
    COCOAudioDataset,
    InMemoryChunkDataset,
)
from rf_detr_finetuning.dataloader.sampler import (
    BalancedClassSampler,
    DeterministicSampler,
)
from rf_detr_finetuning.dataloader.splitter import (
    SplitConfig,
    create_stratified_split,
    create_train_val_test_loaders,
    split_dataset,
)

__all__ = [
    # Datasets
    "AudioChunkDataset",
    "COCOAudioDataset",
    "InMemoryChunkDataset",
    # Collate functions
    "collate_detections",
    "collate_with_targets",
    # Splitting
    "SplitConfig",
    "split_dataset",
    "create_train_val_test_loaders",
    "create_stratified_split",
    # Samplers
    "DeterministicSampler",
    "BalancedClassSampler",
    # COCO Export
    "COCODatasetBuilder",
    "CategoryRegistry",
    "convert_audio_to_coco",
    "parse_frequency_bins",
    "validate_coco_dataset",
]
