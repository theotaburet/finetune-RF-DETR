# Event Merging in Underwater Acoustic Analysis

This document explains how to use class-wise event merging for **marine acoustic inference**.

## Overview

The inference pipeline supports two merging strategies for underwater sound analysis:

1. **IoU-based merging** (default): Uses intersection-over-union threshold
2. **Class-wise merging**: Uses Delta_Time (temporal distance) and Delta_Hz (frequency distance) per marine species

Class-wise merging is particularly important for underwater acoustics because:

- Marine mammals produce sounds with vastly different temporal characteristics (clicks vs. long songs)
- Different species occupy different frequency bands (e.g., blue whales at 10-40 Hz vs. dolphins at >100 kHz)
- Anthropogenic noise (ships, sonar) requires different handling than biological sounds

## Configuration File

Create a YAML configuration file optimized for marine species (e.g., `config/merging.yaml`):

```yaml
# Event Merging Configuration for Underwater Acoustics

# Default merge parameters (applied to all classes unless overridden)
# Marine sounds typically have longer durations and slower repetition rates
default:
  delta_time_ms: 2000.0        # Maximum temporal distance to merge (milliseconds)
  delta_freq_hz: 150.0         # Maximum frequency distance to merge (Hz)
  score_strategy: "max"        # How to combine scores: "max", "avg", or "weighted"

# Class-specific merge parameters for marine species
classes:
  # Humpback whale: Complex songs with frequency sweeps (10 Hz - 4 kHz)
  # Long continuous phrases that may have gaps
  0:
    delta_time_ms: 3000         # Allow 3s gaps for continuous songs
    delta_freq_hz: 400          # Tolerance for frequency sweeps
    score_strategy: "max"

  # Killer whale: Pulsed calls and echolocation clicks (1-20 kHz)
  # Short discrete calls, tight temporal grouping
  1:
    delta_time_ms: 800          # Tight tolerance for discrete calls
    delta_freq_hz: 600          # Broadband clicks
    score_strategy: "max"

  # Blue whale: Very long, infrasonic calls (10-40 Hz)
  # Long duration, extremely low frequency
  2:
    delta_time_ms: 8000         # Very long tolerance (8s)
    delta_freq_hz: 30           # Very narrow frequency band
    score_strategy: "avg"

  # Dolphin: Echolocation clicks and whistles (up to 150 kHz)
  # Rapid clicks, very short duration
  3:
    delta_time_ms: 400          # Very tight for rapid clicks
    delta_freq_hz: 800          # Wide frequency range
    score_strategy: "max"

  # Ship noise: Continuous broadband anthropogenic noise
  # Long duration, may vary slowly
  4:
    delta_time_ms: 5000         # Long tolerance for continuous noise
    delta_freq_hz: 100          # Narrow band
    score_strategy: "avg"

  # Fish chorus: Rhythmic, repetitive grunts/knocks (50-1000 Hz)
  # Moderate duration, rhythmic patterns
  5:
    delta_time_ms: 2000
    delta_freq_hz: 200
    score_strategy: "weighted"

# Post-merge filtering
filtering:
  score_threshold: 0.3          # Minimum confidence to keep (0.0-1.0)
  min_duration_ms: 200.0        # Marine sounds are often >200ms
  max_duration_ms: null         # No maximum duration limit
```

## Recommended Settings by Sound Type

| Sound Type       | delta_time_ms | delta_freq_hz | Strategy | Notes                       |
| ---------------- | ------------- | ------------- | -------- | --------------------------- |
| **Blue Whale**   | 5000-10000    | 20-50         | avg      | Very long calls, infrasonic |
| **Fin Whale**    | 3000-5000     | 50-100        | avg      | Long, repetitive calls      |
| **Humpback**     | 2000-4000     | 300-500       | max      | Complex songs with gaps     |
| **Sperm Whale**  | 1000-2000     | 200-400       | max      | Click trains                |
| **Killer Whale** | 500-1000      | 400-800       | max      | Pulsed calls, clicks        |
| **Dolphin**      | 200-500       | 600-1000      | max      | Rapid clicks, whistles      |
| **Ship Noise**   | 3000-5000     | 50-150        | avg      | Continuous broadband        |
| **Sonar**        | 1000-2000     | 100-200       | avg      | Regular pulses              |
| **Fish Chorus**  | 1500-2500     | 100-300       | weighted | Rhythmic patterns           |

## Usage

### run_inference_audio.py (Recommended for Marine Audio)

```bash
# Single hydrophone recording with class-wise merge config
python run_inference_audio.py \
    --audio data/hydrophone/recording.flac \
    --weights output/marine_model.pth \
    --config config/chunking.yaml \
    --merge-config config/merging.yaml \
    --output results/detections.json

# Batch processing multiple recordings
python run_inference_audio.py \
    --audio-dir data/hydrophone/ \
    --weights output/marine_model.pth \
    --config config/chunking.yaml \
    --merge-config config/merging.yaml \
    --output-dir results/ \
    --confidence 0.6

# Without merge config (uses IoU-based merging)
python run_inference_audio.py \
    --audio data/hydrophone/recording.flac \
    --weights output/marine_model.pth \
    --config config/chunking.yaml \
    --iou-threshold 0.5 \
    --output results/detections.json
```

### run_inference.py

```bash
# With class-wise merge config
python run_inference.py \
    --audio data/hydrophone/recording.flac \
    --weights output/marine_model.pth \
    --chunking-config config/chunking.yaml \
    --merge-config config/merging.yaml \
    --output results.json
```

## Parameters for Marine Acoustics

### Delta_Time (Δt)

- **Type**: float (milliseconds)
- **Description**: Maximum temporal gap between detections to merge into one event
- **Marine Considerations**:
  - **Baleen whales** (blue, fin, humpback): Use 3000-10000 ms for long continuous calls
  - **Toothed whales** (dolphin, orca): Use 200-1000 ms for discrete clicks/calls
  - **Ship noise**: Use 3000-5000 ms for continuous broadband noise
  - **Fish**: Use 1000-2500 ms for rhythmic choruses

### Delta_Hz (Δf)

- **Type**: float (Hz)
- **Description**: Maximum frequency distance to merge events
- **Marine Considerations**:
  - **Low-frequency specialists** (blue whale): Use 20-50 Hz (very narrow)
  - **Mid-frequency** (humpback, ship noise): Use 100-300 Hz
  - **High-frequency** (dolphin clicks): Use 500-1000 Hz (broadband)
  - **Frequency sweeps** (humpback songs): Use 300-500 Hz tolerance

### Score Strategy

- **"max"**: Conservative, best for species identification (preserves highest confidence)
- **"avg"**: Balanced, good for long continuous sounds like ship noise or whale songs
- **"weighted"**: Good for rhythmic sounds like fish choruses (weights by duration)

## How It Works for Marine Audio

1. **Hydrophone Data Loading**: Raw acoustic recordings are loaded
2. **Spectrogram Chunking**: Audio is chunked into overlapping spectrograms (FFT-based)
3. **Per-window Detection**: Model detects marine sounds in each window independently
4. **Class-wise Grouping**: Detections grouped by marine species/class
5. **Class-wise Merging**:
   - Baleen whales: Long gaps allowed (continuous songs)
   - Toothed whales: Tight grouping (discrete clicks/calls)
   - Ship noise: Long continuous merging
   - Fish: Rhythmic pattern preservation
6. **Filtering**: Minimum duration threshold (e.g., 200ms) removes short noise bursts
7. **Output**: Merged marine events with species labels

## Marine Acoustic Considerations

### Frequency Ranges by Species

- **Blue Whale**: 10-40 Hz (infrasonic, long-range)
- **Fin Whale**: 20-100 Hz (infrasonic/low)
- **Humpback**: 10 Hz - 4 kHz (complex songs)
- **Sperm Whale**: 100 Hz - 30 kHz (clicks)
- **Killer Whale**: 1-20 kHz (pulsed calls, clicks)
- **Dolphin**: Up to 150 kHz (echolocation clicks)
- **Ship Noise**: 10 Hz - 100 kHz (broadband)

### Temporal Characteristics

- **Blue Whale Calls**: 10-30 seconds (merge tolerance: 5-10s)
- **Humpback Songs**: Minutes to hours (merge tolerance: 3-5s between phrases)
- **Dolphin Clicks**: \<1 ms to 100 ms (merge tolerance: 200-500ms for click trains)
- **Ship Passage**: Minutes to hours (merge tolerance: 3-5s)
- **Fish Choruses**: Rhythmic, seconds apart (merge tolerance: 1-3s)

## Troubleshooting Marine Audio

**Whale calls being split into multiple events:**

- Increase `delta_time_ms` (e.g., from 1000 to 5000 ms)
- Increase `delta_freq_hz` for frequency-sweeping calls (e.g., humpback songs)

**Ship noise creating too many short detections:**

- Increase `min_duration_ms` to 500-1000 ms
- Use `"avg"` score strategy for continuous noise
- Increase `delta_time_ms` to 5000+ ms

**Dolphin clicks not merging into click trains:**

- Decrease `delta_time_ms` to 200-400 ms for tight click trains
- Keep `delta_freq_hz` high (500-1000 Hz) for broadband clicks
- Use `"max"` strategy to preserve high-confidence detections

**Multiple species at same time:**

- Ensure `delta_freq_hz` is appropriate for each species' frequency band
- Use class-specific configs to prevent cross-species merging

**False positives from ambient noise:**

- Increase `score_threshold` to 0.5 or higher
- Increase `min_duration_ms` to filter short noise bursts
- Use higher confidence threshold in inference (e.g., `--confidence 0.7`)

## Example Output

```json
{
  "audio_path": "hydrophone_2024_001.flac",
  "duration_ms": 60000,
  "num_events": 3,
  "class_names": {
    "0": "humpback_whale",
    "1": "ship_noise",
    "2": "dolphin"
  },
  "events": [
    {
      "start_ms": 5000,
      "end_ms": 15000,
      "duration_ms": 10000,
      "class_id": 0,
      "class_name": "humpback_whale",
      "score": 0.89,
      "min_freq_hz": 200,
      "max_freq_hz": 3500,
      "source_windows": [0, 1, 2],
      "metadata": {
        "merged_count": 3,
        "merge_strategy": "max"
      }
    },
    {
      "start_ms": 20000,
      "end_ms": 45000,
      "duration_ms": 25000,
      "class_id": 1,
      "class_name": "ship_noise",
      "score": 0.95,
      "min_freq_hz": 50,
      "max_freq_hz": 5000,
      "source_windows": [3, 4, 5, 6],
      "metadata": {
        "merged_count": 4,
        "merge_strategy": "avg"
      }
    }
  ]
}
```

## References

- [NOAA Fisheries - Marine Mammal Acoustics](https://www.fisheries.noaa.gov/national/marine-mammal-protection/marine-mammal-acoustics)
- [DOSITS - Discovery of Sound in the Sea](https://dosits.org/)
- [Watkins et al. (2000) - Marine Mammal Sounds](https://cis.whoi.edu/science/B/whalesounds/)
