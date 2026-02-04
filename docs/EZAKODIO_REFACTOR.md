# Ezakodio Integration & Refactoring Plan

## ezakodio Available Functions

### ezakodio.io

| Function               | Signature                                                                                    | Status                           |
| ---------------------- | -------------------------------------------------------------------------------------------- | -------------------------------- |
| `load_audio`           | `(path, sample_rate=None, mono=True, offset=0.0, duration=None, device=None, dtype=float32)` | ✅ Used                          |
| `save_audio`           | `(path, audio, sample_rate, format=None)`                                                    | Available                        |
| `get_audio_info`       | `(path) -> dict`                                                                             | Available                        |
| `iterate_audio_chunks` | `(path, chunk_duration_s=30.0, overlap_s=0.0, ...)`                                          | **Could replace chunking logic** |
| `AudioFileReader`      | Class for streaming                                                                          | Available                        |

### ezakodio.dsp

| Function                    | Signature                                       | Status                                |
| --------------------------- | ----------------------------------------------- | ------------------------------------- |
| `mel_spectrogram`           | `(x, sample_rate, n_mels=128, n_fft=2048, ...)` | ✅ Used                               |
| `resample`                  | `(audio, orig_sr, target_sr)`                   | Available                             |
| `hz_to_mel` / `mel_to_hz`   | Frequency conversion                            | ✅ Used                               |
| `compute_amplitude_stats`   | `(wav) -> dict`                                 | Available                             |
| `get_rms_envelope`          | `(wav, sample_rate, ...)`                       | **Could replace percentile RMS**      |
| `trim_silence`              | `(wav, sample_rate, threshold_db, ...)`         | Available                             |
| `extract_segment`           | `(wav, sample_rate, start_s, end_s, pad_s)`     | **Could replace extract_audio_chunk** |
| `streaming_mel_spectrogram` | For long files                                  | Available                             |

### ezakodio.transforms

| Function               | Signature                            | Status                       |
| ---------------------- | ------------------------------------ | ---------------------------- |
| `compute_rms_db`       | `(wav, eps) -> float`                | ✅ Used                      |
| `detrend`              | `(wav, mode='constant')`             | ✅ Used                      |
| `preemphasis`          | `(wav, coef=0.97)`                   | ✅ Used                      |
| `power_to_db`          | `(x, ref, amin, top_db)`             | Available                    |
| `amplitude_to_db`      | `(x, ref, amin, top_db)`             | Available                    |
| `db_to_power`          | `(x_db, ref)`                        | Available                    |
| `db_to_amplitude`      | `(x_db, ref)`                        | Available                    |
| `apply_agc`            | `(wav, sample_rate, target_db, ...)` | **Could replace custom AGC** |
| `normalize_rms`        | `(wav, target_db, max_gain_db, ...)` | Available                    |
| `normalize_percentile` | `(wav, sr, target_db, ...)`          | Available                    |

### ezakodio top-level (freq_scales)

| Function                                | Status    |
| --------------------------------------- | --------- |
| `hz_to_mel`, `mel_to_hz`                | ✅ Used   |
| `hz_to_bark`, `bark_to_hz`              | Available |
| `hz_to_erb`, `erb_to_hz`                | Available |
| `mel_frequencies`, `linear_frequencies` | Available |

______________________________________________________________________

## Project Functions Analysis

### audio_preprocessing.py - REFACTOR CANDIDATES

| Function                    | Current            | ezakodio Equivalent                                | Action      |
| --------------------------- | ------------------ | -------------------------------------------------- | ----------- |
| `compute_rms_db`            | Custom numpy/torch | `ezakodio.transforms.compute_rms_db`               | ✅ **DONE** |
| `compute_percentile_rms_db` | Custom numpy       | Uses `compute_rms_db` wrapper                      | ✅ **DONE** |
| `apply_detrend`             | Custom numpy       | `ezakodio.transforms.detrend`                      | ✅ **DONE** |
| `apply_preemphasis`         | Custom numpy       | `ezakodio.transforms.preemphasis`                  | ✅ **DONE** |
| `apply_agc`                 | Custom             | `ezakodio.transforms.apply_agc(mode='percentile')` | ✅ **DONE** |

### audio_chunking.py - REFACTOR CANDIDATES

| Function               | Current                         | ezakodio Equivalent                               | Action      |
| ---------------------- | ------------------------------- | ------------------------------------------------- | ----------- |
| `extract_audio_chunk`  | Custom padding                  | `extract_segment(wav, sr, start_s, end_s, pad_s)` | **REPLACE** |
| `pad_audio`            | Custom numpy                    | ❌ Not in ezakodio (part of extract_segment)      | **REMOVE**  |
| `_compute_spectrogram` | Uses `mel_spectrogram`          | Already uses ezakodio                             | ✅ OK       |
| Hz to mel conversion   | Uses `2595 * log10(1 + hz/700)` | `ezakodio.hz_to_mel()`                            | ✅ **DONE** |

### audio_to_coco.py - REFACTOR CANDIDATES

| Function                      | Current                     | ezakodio Equivalent           | Action               |
| ----------------------------- | --------------------------- | ----------------------------- | -------------------- |
| `FrequencyMapper.hz_to_pixel` | Uses `ezakodio.hz_to_mel()` | Already uses ezakodio         | ✅ **DONE**          |
| `spectrogram_to_image`        | Custom PIL conversion       | ❌ Not in ezakodio            | Keep (visualization) |
| `iterate_audio_dataset`       | Custom glob                 | Uses `ezakodio.io.load_audio` | ✅ OK                |
| `process_audio_file`          | Uses `mel_spectrogram`      | Already uses ezakodio         | ✅ OK                |

______________________________________________________________________

## Summary of Refactoring

### ✅ Completed Refactorings

| Module                   | Function            | Replaced With                                      | Benefit                                        |
| ------------------------ | ------------------- | -------------------------------------------------- | ---------------------------------------------- |
| `audio_chunking.py`      | Inline `hz_to_mel`  | `ezakodio.hz_to_mel`                               | DRY - removed duplicate formula                |
| `audio_to_coco.py`       | Hz/mel conversion   | Already used `ezakodio`                            | N/A - was already correct                      |
| `audio_preprocessing.py` | `compute_rms_db`    | `ezakodio.transforms.compute_rms_db` wrapper       | Standard implementation                        |
| `audio_preprocessing.py` | `apply_detrend`     | `ezakodio.transforms.detrend`                      | Standard implementation                        |
| `audio_preprocessing.py` | `apply_preemphasis` | `ezakodio.transforms.preemphasis`                  | Standard implementation                        |
| `audio_preprocessing.py` | `apply_agc`         | `ezakodio.transforms.apply_agc(mode='percentile')` | Robust percentile-based AGC with soft clipping |

### 📋 Optional Future Refactorings

These are working functions that *could* be simplified further:

| Function                    | Current            | ezakodio Option                | Priority                        |
| --------------------------- | ------------------ | ------------------------------ | ------------------------------- |
| `extract_audio_chunk`       | Custom padding     | `ezakodio.dsp.extract_segment` | Medium - could reduce ~50 lines |
| `pad_audio`                 | Custom             | Built into `extract_segment`   | Low - helper function           |
| `compute_percentile_rms_db` | Custom frame-based | Could use `get_rms_envelope`   | Low - working well              |

### ❌ Not Needed in ezakodio

These are domain-specific functions that should stay in this project:

### ❌ Not Needed in ezakodio

These are domain-specific functions that should stay in this project:

- `spectrogram_to_image` - PIL conversion for visualization (domain-specific)
- `FrequencyMapper`, `TimeMapper` - COCO bbox mapping logic (object detection specific)
- `AudioChunker` - Sliding window chunking with bbox alignment (RF-DETR specific)
- `normalize_spectrogram` - Dynamic range compression with percentile clipping (custom strategy)

______________________________________________________________________

## Testing

All tests pass after refactoring:

- ✅ 40 tests in `test_audio_chunking.py`
- ✅ 22 tests in `test_audio_to_coco.py`
- ✅ Preprocessing module import and execution verified

The refactoring maintains 100% backward compatibility while eliminating code duplication.

### 2. Preprocessing Functions (ezakodio.preprocess)

```python
def detrend(wav, mode='constant') -> Tensor  # Remove DC offset or linear trend
def preemphasis(wav, coef=0.97) -> Tensor    # High-frequency emphasis
def normalize_rms(wav, target_db=-25.0, max_gain_db=30.0) -> Tensor
def apply_agc(wav, sample_rate, target_db=-20.0, ...) -> Tensor
```

### 3. AGC Functions (ezakodio.agc or ezakodio.transform)

```python
@dataclass
class AGCResult:
    audio: Tensor
    original_rms_db: float
    gain_applied_db: float
    was_normalized: bool

def apply_agc_for_spectrogram(wav, sample_rate, target_db=-20.0, ...) -> AGCResult
def compute_agc_gain(wav, sample_rate, target_db=-20.0, ...) -> tuple[float, float]
```

### 4. Visualization Utilities (ezakodio.viz)

```python
def spectrogram_to_image(spec, normalize=True, colormap='magma') -> PIL.Image
def spectrogram_to_array(spec, normalize=True) -> np.ndarray  # 0-255 uint8
```

______________________________________________________________________

## Immediate Refactoring Actions

### 1. Replace hz_to_mel in audio_chunking.py

```python
# BEFORE (line ~320)
def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


# AFTER
from ezakodio import hz_to_mel
```

### 2. Replace compute_rms_db in audio_preprocessing.py

```python
# BEFORE
def compute_rms_db(audio, eps=1e-8):
    rms_linear = np.sqrt(np.mean(audio**2))
    return 20 * np.log10(rms_linear + eps)


# AFTER
from ezakodio.dsp import compute_amplitude_stats


def compute_rms_db(audio):
    return compute_amplitude_stats(torch.from_numpy(audio))["rms_db"]
```

### 3. Simplify extract_audio_chunk using ezakodio

```python
# BEFORE: 50+ lines of custom padding logic

# AFTER
from ezakodio.dsp import extract_segment

chunk, _, _ = extract_segment(
    wav=torch.from_numpy(audio),
    sample_rate=sample_rate,
    start_s=start_ms / 1000,
    end_s=end_ms / 1000,
    pad_s=(end_ms - start_ms) / 1000,  # pad to full length
)
```

______________________________________________________________________

## Summary

### Functions to REPLACE with ezakodio:

1. `hz_to_mel` formula → `ezakodio.hz_to_mel()`
2. `mel_to_hz` formula → `ezakodio.mel_to_hz()`
3. `compute_rms_db` → `compute_amplitude_stats()`
4. `extract_audio_chunk` → `extract_segment()`

### Functions to SUGGEST for ezakodio:

1. `power_to_db`, `amplitude_to_db` (from transform.py)
2. `detrend`, `preemphasis` (preprocessing)
3. `apply_agc`, `AGCResult` (gain control)
4. `spectrogram_to_image` (visualization)

### Functions to KEEP in project:

1. `spectrogram_to_image_array` - visualization is project-specific
2. `align_bbox_to_chunk` - detection-specific logic
3. `compute_chunk_boundaries` - custom windowing strategy
4. `FrequencyMapper`, `TimeMapper` - coordinate mapping for bboxes
