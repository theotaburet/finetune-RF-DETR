# Code Roast TODO

Comprehensive bug tracker from code review of the RF-DETR fine-tuning pipeline.

## P0 - Critical Bugs

- [x] **features.py**: Padding applied to `audio` numpy array but `audio_tensor`
  (used for mel_spectrogram) is never recreated from padded audio. Padding has no
  effect. Also `power` param accepted but never forwarded.
- [x] **rfdetr_wrapper.py**: `RFDETRConfig.num_classes` never passed to model in
  `setup_model()`. `freeze_backbone` / `freeze_batch_norm` declared but unused.
  `**kwargs` in `create_rfdetr_trainer()` accepted but never applied.
- [x] **postprocessor.py**: `freq_per_pixel_hz = max_freq_hz / n_mels` assumes
  linear frequency mapping, but pipeline uses mel-scale spectrograms. All
  frequency estimates are wrong. `_window_to_events` shares same linear bug.
- [x] **dataprocessor/\_\_init\_\_.py**: `__all__` defined twice. Second silently
  overwrites first, dropping `AudioChunker`, augmentation functions,
  `draw_bboxes_on_spectrogram`, etc.

## P1 - Important

- [x] **loop.py**: `CosineAnnealingLR` uses `T_max = len(train_loader) * epochs`
  but doesn't account for `accumulate_grad_batches`. Scheduler completes early.
- [x] **loop.py**: Best metric tracking hardcodes `if val_loss < best_metric`,
  ignoring `CheckpointConfig.best_metric` and `best_mode`.
- [x] **config.py**: `TrainerConfig.from_dict()` calls `config_dict.pop()` which
  mutates the caller's dictionary. `to_dict()` only serializes a subset of fields,
  round-tripping loses data.
- [x] **audio_preprocessing.py**: 608-line near-complete duplicate of
  `dataprocessor/preprocessing.py`. Legacy file has extras (`AudioPreprocessor`,
  `compute_percentile_rms_db`, `preprocess_for_inference`) not in the subpackage.
  Resolved by making legacy module re-export from canonical + add unique extras.

## P2 - Medium

- [x] **batch.py**: `BatchPredictor` accepts `batch_size` param but processes
  images one at a time. Documented as sequential predictor with progress tracking.
- [x] **logger.py**: `TrainingLogger` method signatures don't match
  `TrainerCallback` protocol. `_progress`, `_epoch_task`, `_batch_task` initialized
  but never used.
- [x] **dataset.py**: `AudioChunkDataset.__getitem__` calls `process_file()` which
  doesn't exist (should be `chunk_audio_file`). Uses `bbox.class_id` which doesn't
  exist on `ChunkBbox` (should be `category_id`). Linear scan for index.
- [x] **torch.load() without `weights_only=True`** in: `checkpoint.py:102`,
  `rfdetr_wrapper.py:179`, `inference.py:329`, `checkpoint.py:353`.
- [x] **coco_export.py**: Dead if/else with identical branches (lines 463-466).
  `processed_events` computed but never used. `random.seed()` mutates global state.
  Separator lines in logs.
- [x] **inference.py**: `RFDETRPredictor` passes `nn.Identity()` to
  `Predictor.__init__` as hack. Model class dict duplicated across files.
- [x] **metrics.py**: Hardcoded 640x640 image dimensions. Temp files not cleaned
  up on error. `import os` inside function body. Redundant empty-predictions check.
- [x] **event.py**: `EventList.load()` - JSON turns int keys to strings, breaking
  `class_names` lookup. `AudioEvent.merge()` uses avg score while
  `EventMerger._merge_group()` supports configurable strategy.
- [x] **predictor/\_\_init\_\_.py**: `AudioPredictionResult` not exported.
- [x] **sampler.py**: `callable` (lowercase) used as type hint instead of
  `Callable`.
- [x] **splitter.py**: `SplitConfig.stratify_by` declared but never read.

## P3 - Nits

- [x] **loop.py**: `from torch.cuda.amp import GradScaler, autocast` deprecated
  since PyTorch 2.0. Use `torch.amp.GradScaler` and `torch.amp.autocast`.
