"""Tests for event merging with synthetic audio and debug visualizations.

Tests the full pipeline: synthetic audio generation -> simulated chunking ->
simulated detections -> event merging -> validation + plots.

Run with:
    pytest tests/unit/eventprocessor/test_event_merger.py -v -s

Debug plots are saved to: output/test_merge_debug/

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList
from rf_detr_finetuning.eventprocessor.merger import (
    EventMerger,
    MergeConfig,
    cluster_merge,
    nms_merge,
    temporal_merge,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/test_merge_debug")
_CONFIG_PATH = Path(__file__).parent / "merge_config.yaml"


def _load_merge_config() -> dict:
    """Load merge parameters from the YAML config file next to this test module.

    Returns:
        Dictionary with keys ``merge`` and ``spectrogram``.

    """
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


# -- Synthetic audio helpers --------------------------------------------------


@dataclass
class SyntheticEvent:
    """Ground-truth event used to generate synthetic audio and simulated detections.

    Attributes:
        start_ms: Event start time in milliseconds.
        end_ms: Event end time in milliseconds.
        freq_hz: Center frequency (or start frequency for chirps) in Hz.
        bandwidth_hz: Frequency bandwidth of the event.
        class_id: Class label.
        class_name: Human-readable class name.
        amplitude: Signal amplitude (0-1).
        sound_type: Synthesis model — one of "chirp_up", "chirp_down",
            "chirp_exp", "pulse_train", "harmonic", "am_tone", "fm_tone".

    """

    start_ms: float
    end_ms: float
    freq_hz: float
    bandwidth_hz: float = 200.0
    class_id: int = 0
    class_name: str = "event"
    amplitude: float = 0.8
    sound_type: str = "chirp_up"


@dataclass
class SimulatedDetection:
    """A detection produced by a simulated model on a single window.

    Attributes:
        start_ms: Absolute start time in ms.
        end_ms: Absolute end time in ms.
        min_freq_hz: Lower frequency bound.
        max_freq_hz: Upper frequency bound.
        class_id: Predicted class.
        class_name: Predicted class name.
        score: Confidence score.
        window_index: Which window produced this detection.

    """

    start_ms: float
    end_ms: float
    min_freq_hz: float
    max_freq_hz: float
    class_id: int = 0
    class_name: str = "event"
    score: float = 0.9
    window_index: int = 0

    def to_audio_event(self) -> AudioEvent:
        """Convert to AudioEvent."""
        return AudioEvent(
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            class_id=self.class_id,
            class_name=self.class_name,
            score=self.score,
            min_freq_hz=self.min_freq_hz,
            max_freq_hz=self.max_freq_hz,
            source_windows=[self.window_index],
        )


def _bandpass_signal(
    sig: np.ndarray,
    sample_rate: int,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """Apply a zero-phase FFT bandpass to keep signal within [f_low, f_high].

    Args:
        sig: Input signal array.
        sample_rate: Sample rate in Hz.
        f_low: Lower frequency cutoff in Hz.
        f_high: Upper frequency cutoff in Hz.

    Returns:
        Bandpass-filtered signal (same length, float32).

    """
    n = len(sig)
    spectrum = np.fft.rfft(sig.astype(np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= f_low) & (freqs <= f_high)
    spectrum[~mask] = 0.0
    return np.fft.irfft(spectrum, n=n).astype(np.float32)


def _synthesize_event_signal(
    event: SyntheticEvent,
    t: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Synthesize a single event signal on a pre-allocated time array.

    Each sound type generates energy confined to
    ``[freq_hz - bandwidth_hz/2, freq_hz + bandwidth_hz/2]``
    via a final FFT bandpass step to prevent spectral bleed.

    Args:
        event: Synthetic event parameters.
        t: Full-audio time array in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Signal array of same length as t (zeros outside the event window).

    """
    rng = np.random.default_rng(abs(hash(event.class_name)) % (2**31))

    start_s = event.start_ms / 1000.0
    end_s = event.end_ms / 1000.0
    duration_s = end_s - start_s
    if duration_s <= 0:
        return np.zeros_like(t)

    mask = (t >= start_s) & (t < end_s)
    t_local = t[mask] - start_s  # time relative to event start, in seconds
    n = len(t_local)
    if n == 0:
        return np.zeros_like(t)

    f0 = event.freq_hz
    bw = event.bandwidth_hz
    f_low = max(1.0, f0 - bw / 2)
    f_high = f0 + bw / 2
    stype = event.sound_type

    if stype == "chirp_up":
        # Linear up-sweep: f_low -> f_high
        f_inst = f_low + (bw / duration_s) * t_local
        phase = 2 * np.pi * np.cumsum(f_inst) / sample_rate
        sig = np.sin(phase).astype(np.float32)

    elif stype == "chirp_down":
        # Linear down-sweep: f_high -> f_low
        f_inst = f_high - (bw / duration_s) * t_local
        phase = 2 * np.pi * np.cumsum(f_inst) / sample_rate
        sig = np.sin(phase).astype(np.float32)

    elif stype == "chirp_exp":
        # Exponential sweep within [f_low, f_high]
        f_end = max(f_high, f_low * 1.01)
        k = np.log(f_end / max(f_low, 1.0)) / duration_s
        f_inst = f_low * np.exp(k * t_local)
        phase = 2 * np.pi * np.cumsum(f_inst) / sample_rate
        sig = np.sin(phase).astype(np.float32)

    elif stype == "pulse_train":
        # Narrow-band Gaussian pulses at f0 — bandpass keeps them in band
        pulse_interval_s = max(0.05, duration_s / max(int(duration_s / 0.08), 1))
        sig = np.zeros(n, dtype=np.float32)
        pulse_width_s = min(0.025, pulse_interval_s * 0.4)
        t_pulse = 0.0
        while t_pulse < duration_s:
            center = t_pulse + rng.uniform(-pulse_interval_s * 0.1, pulse_interval_s * 0.1)
            envelope = np.exp(-0.5 * ((t_local - center) / pulse_width_s) ** 2).astype(np.float32)
            carrier = np.sin(2 * np.pi * f0 * t_local).astype(np.float32)
            sig += envelope * carrier
            t_pulse += pulse_interval_s
        peak = np.max(np.abs(sig))
        if peak > 0:
            sig /= peak

    elif stype == "harmonic":
        # Only include harmonics whose frequency falls within [f_low, f_high]
        sig = np.zeros(n, dtype=np.float32)
        for k_h, amp_k in enumerate([1.0, 0.5, 0.25, 0.12], start=1):
            freq_k = k_h * f0
            if freq_k > f_high:
                break
            sig += amp_k * np.sin(2 * np.pi * freq_k * t_local).astype(np.float32)
        peak = np.max(np.abs(sig) + 1e-9)
        sig /= peak

    elif stype == "am_tone":
        # AM: carrier at f0, modulator slow enough to stay in band
        # Sidebands at f0 ± mod_freq; keep mod_freq < bw/2
        mod_freq = min(max(bw / 8, 2.0), bw / 2 - 1.0)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t_local)).astype(np.float32)
        sig = (envelope * np.sin(2 * np.pi * f0 * t_local)).astype(np.float32)

    elif stype == "fm_tone":
        # FM: keep modulation index low so sidebands stay within bandwidth
        # Carson's rule: BW ≈ 2*(deviation + mod_freq); solve for small deviation
        mod_freq = max(bw / 8, 2.0)
        # Modulation index β=1 -> most energy in first few sidebands
        deviation = min(bw / 4, mod_freq)
        phase = (
            2 * np.pi * (f0 * t_local + (deviation / (2 * np.pi * mod_freq)) * np.sin(2 * np.pi * mod_freq * t_local))
        )
        sig = np.sin(phase).astype(np.float32)

    else:
        # Fallback: pure sine at f0
        sig = np.sin(2 * np.pi * f0 * t_local).astype(np.float32)

    # Bandpass to strictly contain energy within declared bandwidth
    if n > 32:
        sig = _bandpass_signal(sig, sample_rate, f_low, f_high)
        peak = np.max(np.abs(sig) + 1e-9)
        sig /= peak

    # Smooth fade-in / fade-out (Hann window on the edges)
    fade_samples = min(int(0.015 * sample_rate), n // 4)
    if fade_samples > 1:
        fade = np.hanning(fade_samples * 2).astype(np.float32)
        sig[:fade_samples] *= fade[:fade_samples]
        sig[-fade_samples:] *= fade[fade_samples:]

    result = np.zeros(len(t), dtype=np.float32)
    result[mask] = event.amplitude * sig
    return result


def generate_synthetic_audio(
    duration_ms: float,
    sample_rate: int,
    events: list[SyntheticEvent],
    noise_level: float = 0.04,
) -> np.ndarray:
    """Generate synthetic audio with bioacoustic-style events embedded in Gaussian noise.

    Supports chirps, pulse trains, harmonic tones, AM and FM tones.

    Args:
        duration_ms: Total audio duration in ms.
        sample_rate: Sample rate in Hz.
        events: List of synthetic events to embed.
        noise_level: Background white Gaussian noise amplitude.

    Returns:
        1D numpy array of audio samples (float32).

    """
    n_samples = int(duration_ms * sample_rate / 1000)
    rng = np.random.default_rng(0)

    # White Gaussian noise background
    audio = (rng.standard_normal(n_samples) * noise_level).astype(np.float32)

    t = np.arange(n_samples) / sample_rate
    for event in events:
        audio += _synthesize_event_signal(event, t, sample_rate)

    return audio


def compute_chunk_windows(
    duration_ms: float,
    window_ms: float,
    overlap_ratio: float,
) -> list[tuple[float, float, int]]:
    """Compute overlapping window boundaries.

    Args:
        duration_ms: Total audio duration.
        window_ms: Window duration in ms.
        overlap_ratio: Overlap between consecutive windows (0-1).

    Returns:
        List of (start_ms, end_ms, window_index) tuples.

    """
    stride_ms = window_ms * (1.0 - overlap_ratio)
    windows = []
    start = 0.0
    idx = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        if (end - start) >= window_ms * 0.5:
            windows.append((start, end, idx))
            idx += 1
        start += stride_ms
    return windows


def simulate_detections_for_event(
    event: SyntheticEvent,
    windows: list[tuple[float, float, int]],
    min_overlap_ratio: float = 0.1,
    score_noise: float = 0.05,
    time_jitter_ms: float = 20.0,
    freq_jitter_hz: float = 30.0,
) -> list[SimulatedDetection]:
    """Simulate model detections for a ground-truth event across windows.

    For each window that overlaps with the event, produces a detection
    clipped to the window boundaries with some random jitter.

    Args:
        event: Ground-truth synthetic event.
        windows: List of (start_ms, end_ms, window_index).
        min_overlap_ratio: Minimum overlap to produce a detection.
        score_noise: Random noise on confidence score.
        time_jitter_ms: Random jitter on time boundaries.
        freq_jitter_hz: Random jitter on frequency boundaries.

    Returns:
        List of simulated detections.

    """
    rng = np.random.default_rng(42)
    detections = []

    for w_start, w_end, w_idx in windows:
        # Compute overlap
        overlap_start = max(event.start_ms, w_start)
        overlap_end = min(event.end_ms, w_end)
        overlap = overlap_end - overlap_start
        event_duration = event.end_ms - event.start_ms

        if event_duration <= 0 or overlap / event_duration < min_overlap_ratio:
            continue

        # Clip detection to window with jitter
        det_start = max(w_start, overlap_start + rng.uniform(-time_jitter_ms, time_jitter_ms))
        det_end = min(w_end, overlap_end + rng.uniform(-time_jitter_ms, time_jitter_ms))
        det_start = max(det_start, w_start)
        det_end = min(det_end, w_end)

        if det_end <= det_start:
            continue

        min_freq = event.freq_hz - event.bandwidth_hz / 2 + rng.uniform(-freq_jitter_hz, freq_jitter_hz)
        max_freq = event.freq_hz + event.bandwidth_hz / 2 + rng.uniform(-freq_jitter_hz, freq_jitter_hz)
        min_freq = max(0.0, min_freq)
        if max_freq <= min_freq:
            max_freq = min_freq + 50.0

        score = min(1.0, max(0.1, event.amplitude + rng.uniform(-score_noise, score_noise)))

        detections.append(
            SimulatedDetection(
                start_ms=det_start,
                end_ms=det_end,
                min_freq_hz=min_freq,
                max_freq_hz=max_freq,
                class_id=event.class_id,
                class_name=event.class_name,
                score=score,
                window_index=w_idx,
            )
        )

    return detections


# -- Plotting helpers ---------------------------------------------------------


def _try_import_matplotlib():
    """Import matplotlib, skip test if not available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        pytest.skip("matplotlib not installed, skipping plot test")


def _compute_spectrogram_for_plot(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    f_max: float | None = None,
) -> np.ndarray:
    """Compute a mel spectrogram in dB for plotting.

    Args:
        audio: 1D audio array.
        sample_rate: Sample rate in Hz.
        n_mels: Number of mel bands.
        f_max: Max frequency (None = Nyquist).

    Returns:
        2D numpy array (n_mels, n_frames) in dB scale, flipped so high freq at top.

    """
    from rf_detr_finetuning.dataprocessor.features import compute_mel_spectrogram_db, flip_spectrogram

    spec_db = compute_mel_spectrogram_db(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        f_max=f_max or sample_rate / 2,
        min_db=-80.0,
    )
    return flip_spectrogram(spec_db)


def _draw_boxes_on_spectrogram(
    ax,
    boxes: list[dict],
    duration_ms: float,
    f_min: float,
    f_max: float,
    n_mels: int,
    use_mel: bool = True,
) -> None:
    """Draw frequency-aware bounding boxes on a spectrogram axis.

    Converts Hz coordinates to mel-pixel Y positions so boxes align with the
    mel-scaled spectrogram image.

    Args:
        ax: Matplotlib axis with spectrogram imshow.
        boxes: List of dicts with keys: start_ms, end_ms, min_freq_hz,
            max_freq_hz, color, linewidth, alpha, linestyle, label.
        duration_ms: Total duration in ms (x-axis extent).
        f_min: Min frequency of the spectrogram in Hz.
        f_max: Max frequency of the spectrogram in Hz.
        n_mels: Number of mel bins.
        use_mel: Whether to use mel scale for Y mapping.

    """
    from ezakodio import hz_to_mel
    from matplotlib.patches import Rectangle

    mel_min_val = hz_to_mel(f_min) if use_mel else f_min
    mel_max_val = hz_to_mel(f_max) if use_mel else f_max
    mel_range = mel_max_val - mel_min_val

    for box in boxes:
        # Convert Hz -> mel -> pixel Y (flipped: high freq at top = y=0)
        box_fmin = box["min_freq_hz"]
        box_fmax = box["max_freq_hz"]

        if use_mel:
            y_top = n_mels * (1.0 - (hz_to_mel(box_fmax) - mel_min_val) / mel_range)
            y_bot = n_mels * (1.0 - (hz_to_mel(box_fmin) - mel_min_val) / mel_range)
        else:
            y_top = n_mels * (1.0 - (box_fmax - f_min) / mel_range)
            y_bot = n_mels * (1.0 - (box_fmin - f_min) / mel_range)

        # Convert time ms -> pixel X
        x_left = box["start_ms"]
        width = box["end_ms"] - box["start_ms"]

        rect = Rectangle(
            (x_left, y_top),
            width,
            y_bot - y_top,
            linewidth=box.get("linewidth", 1.5),
            edgecolor=box.get("color", "red"),
            facecolor=box.get("facecolor", "none"),
            linestyle=box.get("linestyle", "-"),
        )
        ax.add_patch(rect)

        label = box.get("label")
        if label:
            ax.text(
                x_left + 5,
                y_top - 2,
                label,
                fontsize=6,
                color=box.get("color", "red"),
                fontweight="bold",
                va="bottom",
                clip_on=True,
            )


def plot_merge_result(
    duration_ms: float,
    windows: list[tuple[float, float, int]],
    raw_detections: list[SimulatedDetection],
    merged_events: EventList,
    ground_truth: list[SyntheticEvent],
    title: str = "Event Merge Result",
    output_path: Path | None = None,
    audio: np.ndarray | None = None,
    sample_rate: int = 16000,
    n_mels: int = 128,
    f_max: float | None = None,
) -> None:
    """Plot a debug visualization of the merge process with spectrogram views.

    Uses GridSpec to create 3 panels, all showing the mel spectrogram (dB):
    1. Spectrogram with overlapping window boundaries
    2. Spectrogram with raw detections overlaid
    3. Spectrogram with merged events vs ground truth overlaid

    Args:
        duration_ms: Total audio duration.
        windows: Chunk window boundaries.
        raw_detections: Detections before merging.
        merged_events: Events after merging.
        ground_truth: Ground-truth events.
        title: Plot title.
        output_path: Where to save the PNG.
        audio: Audio waveform for spectrogram computation.
        sample_rate: Audio sample rate.
        n_mels: Number of mel bands for spectrogram.
        f_max: Max frequency in Hz (None = Nyquist).

    """
    plt = _try_import_matplotlib()
    import matplotlib.gridspec as gridspec

    if audio is None:
        logger.warning("No audio provided, cannot compute spectrogram for plot")
        return

    effective_f_max = f_max or sample_rate / 2
    spec_db = _compute_spectrogram_for_plot(audio, sample_rate, n_mels=n_mels, f_max=effective_f_max)

    # Shared imshow kwargs
    imshow_kwargs = dict(
        aspect="auto",
        origin="upper",
        cmap="magma",
        extent=[0, duration_ms, n_mels, 0],
        interpolation="nearest",
    )

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # -- Panel 1: Spectrogram + window boundaries --
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(spec_db, **imshow_kwargs)
    cb0 = fig.colorbar(im0, ax=ax0, pad=0.01, aspect=30)
    cb0.set_label("dB", fontsize=9)

    colors_w = plt.cm.Set2(np.linspace(0, 1, max(len(windows), 1)))
    for w_start, w_end, w_idx in windows:
        color = colors_w[w_idx % len(colors_w)]
        ax0.axvline(w_start, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
        ax0.axvline(w_end, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
        ax0.axvspan(w_start, w_end, alpha=0.08, color=color)
        ax0.text(
            (w_start + w_end) / 2,
            2,
            f"W{w_idx}",
            ha="center",
            va="top",
            fontsize=7,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor=color),
        )

    ax0.set_ylabel("Mel bin")
    ax0.set_title(f"Mel Spectrogram (dB) — {len(windows)} overlapping windows", fontsize=11)
    ax0.set_xlim(0, duration_ms)

    # -- Panel 2: Spectrogram + raw detections --
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    im1 = ax1.imshow(spec_db, **imshow_kwargs)
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.01, aspect=30)
    cb1.set_label("dB", fontsize=9)

    class_colors = {}
    cmap_tab = plt.cm.tab10
    det_boxes = []
    for det in raw_detections:
        if det.class_id not in class_colors:
            class_colors[det.class_id] = cmap_tab(len(class_colors) % 10)
        color = class_colors[det.class_id]
        det_boxes.append(
            dict(
                start_ms=det.start_ms,
                end_ms=det.end_ms,
                min_freq_hz=det.min_freq_hz,
                max_freq_hz=det.max_freq_hz,
                color=color,
                facecolor=(*color[:3], 0.25),
                linewidth=1.2,
                label=f"W{det.window_index} ({det.score:.2f})",
            )
        )

    _draw_boxes_on_spectrogram(ax1, det_boxes, duration_ms, 0.0, effective_f_max, n_mels)

    ax1.set_ylabel("Mel bin")
    ax1.set_title(f"Raw Detections ({len(raw_detections)} total)", fontsize=11)
    ax1.set_xlim(0, duration_ms)

    # -- Panel 3: Spectrogram + merged events vs ground truth --
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    im2 = ax2.imshow(spec_db, **imshow_kwargs)
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.01, aspect=30)
    cb2.set_label("dB", fontsize=9)

    gt_boxes = []
    for gt in ground_truth:
        gt_boxes.append(
            dict(
                start_ms=gt.start_ms,
                end_ms=gt.end_ms,
                min_freq_hz=gt.freq_hz - gt.bandwidth_hz / 2,
                max_freq_hz=gt.freq_hz + gt.bandwidth_hz / 2,
                color="lime",
                facecolor=(0.0, 1.0, 0.0, 0.15),
                linewidth=2.0,
                linestyle="--",
                label=f"GT: {gt.class_name}",
            )
        )

    merged_boxes = []
    for event in merged_events:
        merged_count = event.metadata.get("merged_count", 1)
        merged_boxes.append(
            dict(
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                min_freq_hz=event.min_freq_hz or 0,
                max_freq_hz=event.max_freq_hz or effective_f_max,
                color="red",
                facecolor=(1.0, 0.0, 0.0, 0.18),
                linewidth=2.0,
                label=f"{event.class_name} ({event.score:.2f}, n={merged_count})",
            )
        )

    _draw_boxes_on_spectrogram(ax2, gt_boxes + merged_boxes, duration_ms, 0.0, effective_f_max, n_mels)

    ax2.set_ylabel("Mel bin")
    ax2.set_xlabel("Time (ms)")
    ax2.set_title(f"Merged Events ({len(merged_events)}) vs Ground Truth ({len(ground_truth)})", fontsize=11)
    ax2.set_xlim(0, duration_ms)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    plt.close(fig)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def output_dir():
    """Create and return the debug output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# -- Unit tests for IoU methods -----------------------------------------------


class TestTemporalFrequencyIoU:
    """Tests for AudioEvent.temporal_frequency_iou()."""

    def test_identical_events(self):
        """Identical events should have IoU = 1.0."""
        e = AudioEvent(start_ms=0, end_ms=1000, class_id=0, min_freq_hz=500, max_freq_hz=1500)
        assert e.temporal_frequency_iou(e) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping events should have IoU = 0.0."""
        a = AudioEvent(start_ms=0, end_ms=1000, class_id=0, min_freq_hz=500, max_freq_hz=1000)
        b = AudioEvent(start_ms=2000, end_ms=3000, class_id=0, min_freq_hz=500, max_freq_hz=1000)
        assert a.temporal_frequency_iou(b) == pytest.approx(0.0)

    def test_no_freq_overlap(self):
        """Temporal overlap but no frequency overlap -> IoU = 0.0."""
        a = AudioEvent(start_ms=0, end_ms=1000, class_id=0, min_freq_hz=500, max_freq_hz=1000)
        b = AudioEvent(start_ms=0, end_ms=1000, class_id=0, min_freq_hz=2000, max_freq_hz=3000)
        assert a.temporal_frequency_iou(b) == pytest.approx(0.0)

    def test_partial_overlap_2d(self):
        """Partial overlap in both time and frequency."""
        a = AudioEvent(start_ms=0, end_ms=1000, class_id=0, min_freq_hz=0, max_freq_hz=1000)
        b = AudioEvent(start_ms=500, end_ms=1500, class_id=0, min_freq_hz=500, max_freq_hz=1500)
        # Time intersection: 500-1000 = 500ms, Freq intersection: 500-1000 = 500Hz
        # Intersection area = 500 * 500 = 250000
        # Area a = 1000 * 1000 = 1000000, Area b = 1000 * 1000 = 1000000
        # Union = 1000000 + 1000000 - 250000 = 1750000
        # IoU = 250000 / 1750000 = 1/7
        assert a.temporal_frequency_iou(b) == pytest.approx(1 / 7, abs=1e-6)

    def test_fallback_to_temporal_when_no_freq(self):
        """Should fallback to temporal IoU when frequency info is missing."""
        a = AudioEvent(start_ms=0, end_ms=1000, class_id=0)
        b = AudioEvent(start_ms=500, end_ms=1500, class_id=0)
        iou_2d = a.temporal_frequency_iou(b)
        iou_temporal = a.temporal_iou(b)
        assert iou_2d == pytest.approx(iou_temporal)

    def test_contained_event(self):
        """One event fully contains the other."""
        outer = AudioEvent(start_ms=0, end_ms=2000, class_id=0, min_freq_hz=0, max_freq_hz=2000)
        inner = AudioEvent(start_ms=500, end_ms=1000, class_id=0, min_freq_hz=500, max_freq_hz=1000)
        # Intersection = 500 * 500 = 250000
        # outer area = 2000 * 2000 = 4000000, inner area = 500 * 500 = 250000
        # Union = 4000000 + 250000 - 250000 = 4000000
        # IoU = 250000 / 4000000 = 0.0625
        assert outer.temporal_frequency_iou(inner) == pytest.approx(0.0625, abs=1e-6)


# -- Unit tests for Union-Find merger ----------------------------------------


class TestUnionFindMerger:
    """Tests for the Union-Find based EventMerger."""

    def test_empty_events(self):
        """Merging empty list returns empty list."""
        events = EventList(events=[])
        merged = cluster_merge(events)
        assert len(merged) == 0

    def test_single_event(self):
        """Single event passes through unchanged."""
        e = AudioEvent(start_ms=100, end_ms=500, class_id=0, score=0.9)
        events = EventList(events=[e])
        merged = cluster_merge(events, iou_threshold=0.3)
        assert len(merged) == 1
        assert merged[0].start_ms == 100
        assert merged[0].end_ms == 500

    def test_two_overlapping_events_merge(self):
        """Two overlapping same-class events should merge."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.9, min_freq_hz=500, max_freq_hz=1000),
                AudioEvent(start_ms=500, end_ms=1500, class_id=0, score=0.8, min_freq_hz=500, max_freq_hz=1000),
            ]
        )
        merged = cluster_merge(events, iou_threshold=0.2)
        assert len(merged) == 1
        assert merged[0].start_ms == 0
        assert merged[0].end_ms == 1500
        assert merged[0].score == 0.9  # max strategy

    def test_different_classes_not_merged(self):
        """Events of different classes should not merge even if overlapping."""
        events = EventList(
            events=[
                AudioEvent(
                    start_ms=0,
                    end_ms=1000,
                    class_id=0,
                    class_name="whistle",
                    score=0.9,
                    min_freq_hz=500,
                    max_freq_hz=1000,
                ),
                AudioEvent(
                    start_ms=200,
                    end_ms=1200,
                    class_id=1,
                    class_name="click",
                    score=0.8,
                    min_freq_hz=500,
                    max_freq_hz=1000,
                ),
            ]
        )
        merged = cluster_merge(events, iou_threshold=0.2, merge_same_class_only=True)
        assert len(merged) == 2

    def test_three_window_chain_merge(self):
        """Three detections forming a chain should all merge transitively.

        This is the key test case: A overlaps B, B overlaps C, but A does NOT
        overlap C. The Union-Find approach should still merge all three.

        """
        # Window layout: stride=2560ms, window=3200ms
        # A: 1000-3200, B: 2560-5760, C: 5120-6500
        events = EventList(
            events=[
                AudioEvent(
                    start_ms=1000,
                    end_ms=3200,
                    class_id=0,
                    score=0.85,
                    min_freq_hz=800,
                    max_freq_hz=1200,
                    source_windows=[0],
                ),
                AudioEvent(
                    start_ms=2560,
                    end_ms=5760,
                    class_id=0,
                    score=0.9,
                    min_freq_hz=800,
                    max_freq_hz=1200,
                    source_windows=[1],
                ),
                AudioEvent(
                    start_ms=5120,
                    end_ms=6500,
                    class_id=0,
                    score=0.75,
                    min_freq_hz=800,
                    max_freq_hz=1200,
                    source_windows=[2],
                ),
            ]
        )
        # A-B temporal IoU > 0, B-C temporal IoU > 0, but A-C temporal IoU = 0
        # With Union-Find, all three should be in one cluster
        merged = cluster_merge(events, iou_threshold=0.1)
        assert len(merged) == 1
        assert merged[0].start_ms == 1000
        assert merged[0].end_ms == 6500
        assert merged[0].score == 0.9  # max
        assert set(merged[0].source_windows) == {0, 1, 2}

    def test_frequency_separation_prevents_merge(self):
        """Events at same time but different frequency bands should NOT merge with 2D IoU."""
        events = EventList(
            events=[
                AudioEvent(
                    start_ms=0,
                    end_ms=1000,
                    class_id=0,
                    score=0.9,
                    min_freq_hz=500,
                    max_freq_hz=1000,
                ),
                AudioEvent(
                    start_ms=0,
                    end_ms=1000,
                    class_id=0,
                    score=0.8,
                    min_freq_hz=3000,
                    max_freq_hz=4000,
                ),
            ]
        )
        merged = cluster_merge(events, iou_threshold=0.1, use_2d_iou=True)
        assert len(merged) == 2

    def test_gap_tolerance_bridges_events(self):
        """Events with a small temporal gap should merge with gap_tolerance."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.9, min_freq_hz=500, max_freq_hz=1000),
                AudioEvent(start_ms=1050, end_ms=2000, class_id=0, score=0.8, min_freq_hz=500, max_freq_hz=1000),
            ]
        )
        # Without gap tolerance: IoU = 0, should not merge
        merged_no_gap = cluster_merge(events, iou_threshold=0.1, gap_tolerance_ms=0.0)
        assert len(merged_no_gap) == 2

        # With gap tolerance: 50ms gap <= 100ms tolerance, should merge
        merged_gap = cluster_merge(events, iou_threshold=0.1, gap_tolerance_ms=100.0)
        assert len(merged_gap) == 1
        assert merged_gap[0].start_ms == 0
        assert merged_gap[0].end_ms == 2000

    def test_score_threshold_filters(self):
        """Events below score threshold should be filtered after merge."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.3),
                AudioEvent(start_ms=5000, end_ms=6000, class_id=0, score=0.9),
            ]
        )
        config = MergeConfig(iou_threshold=0.3, score_threshold=0.5)
        merger = EventMerger(config)
        merged = merger.merge(events)
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_merge_preserves_metadata(self):
        """Merged event should contain correct metadata."""
        events = EventList(
            events=[
                AudioEvent(
                    start_ms=0,
                    end_ms=1000,
                    class_id=0,
                    class_name="whistle",
                    score=0.7,
                    min_freq_hz=800,
                    max_freq_hz=1200,
                    source_windows=[0],
                ),
                AudioEvent(
                    start_ms=500,
                    end_ms=1500,
                    class_id=0,
                    class_name="whistle",
                    score=0.9,
                    min_freq_hz=750,
                    max_freq_hz=1250,
                    source_windows=[1],
                ),
            ]
        )
        merged = cluster_merge(events, iou_threshold=0.1)
        assert len(merged) == 1
        m = merged[0]
        assert m.class_name == "whistle"  # from highest-scoring
        assert m.min_freq_hz == 750  # min of all
        assert m.max_freq_hz == 1250  # max of all
        assert set(m.source_windows) == {0, 1}
        assert m.metadata["merged_count"] == 2

    def test_avg_score_strategy(self):
        """Test average score merge strategy."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.6, min_freq_hz=500, max_freq_hz=1000),
                AudioEvent(start_ms=500, end_ms=1500, class_id=0, score=0.8, min_freq_hz=500, max_freq_hz=1000),
            ]
        )
        config = MergeConfig(iou_threshold=0.1, merge_strategy="avg", use_2d_iou=True)
        merger = EventMerger(config)
        merged = merger.merge(events)
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(0.7)


# -- Integration tests with synthetic audio + plots --------------------------


class TestMergeWithSyntheticAudio:
    """Integration tests using synthetic audio to verify merge correctness.

    Each test generates synthetic audio, simulates chunking and detections,
    runs the merger, and produces a debug plot.

    Merge parameters default to the values in ``merge_config.yaml`` next to
    this file. Individual tests may pass explicit keyword overrides to
    ``_run_scenario`` when a non-default value is required.

    """

    def _run_scenario(
        self,
        *,
        duration_ms: float,
        sample_rate: int,
        gt_events: list[SyntheticEvent],
        window_ms: float,
        overlap_ratio: float,
        iou_threshold: float | None = None,
        gap_tolerance_ms: float | None = None,
        use_2d_iou: bool | None = None,
        title: str,
        output_path: Path,
    ) -> tuple[list[SimulatedDetection], EventList]:
        """Run a full merge scenario and return (raw_detections, merged_events).

        Merge parameters that are not supplied fall back to the values in
        ``merge_config.yaml``.

        """
        cfg = _load_merge_config()
        merge_cfg = cfg.get("merge", {})
        spec_cfg = cfg.get("spectrogram", {})

        # Apply YAML defaults for any unspecified merge params
        if iou_threshold is None:
            iou_threshold = float(merge_cfg.get("iou_threshold", 0.05))
        if gap_tolerance_ms is None:
            gap_tolerance_ms = float(merge_cfg.get("gap_tolerance_ms", 0.0))
        if use_2d_iou is None:
            use_2d_iou = bool(merge_cfg.get("use_2d_iou", True))

        merge_same_class_only = bool(merge_cfg.get("merge_same_class_only", True))
        merge_strategy: str = str(merge_cfg.get("merge_strategy", "max"))
        score_threshold = float(merge_cfg.get("score_threshold", 0.0))
        n_mels = int(spec_cfg.get("n_mels", 128))

        # Generate audio
        audio = generate_synthetic_audio(duration_ms, sample_rate, gt_events)

        # Compute windows
        windows = compute_chunk_windows(duration_ms, window_ms, overlap_ratio)

        # Simulate detections
        all_detections: list[SimulatedDetection] = []
        for gt in gt_events:
            dets = simulate_detections_for_event(gt, windows)
            all_detections.extend(dets)

        # Convert to AudioEvents
        audio_events = [d.to_audio_event() for d in all_detections]
        event_list = EventList(events=audio_events, duration_ms=duration_ms)

        # Merge using full MergeConfig (respects all YAML params)
        config = MergeConfig(
            iou_threshold=iou_threshold,
            gap_tolerance_ms=gap_tolerance_ms,
            use_2d_iou=use_2d_iou,
            merge_same_class_only=merge_same_class_only,
            merge_strategy=merge_strategy,  # type: ignore[arg-type]
            score_threshold=score_threshold,
        )
        merged = EventMerger(config).merge(event_list)

        # Plot
        plot_merge_result(
            duration_ms=duration_ms,
            windows=windows,
            raw_detections=all_detections,
            merged_events=merged,
            ground_truth=gt_events,
            title=title,
            output_path=output_path,
            audio=audio,
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        return all_detections, merged

    def test_single_short_event(self, output_dir):
        """A short event fitting in one window should produce one merged event."""
        gt = [
            SyntheticEvent(
                start_ms=1500, end_ms=2500, freq_hz=1000, bandwidth_hz=300, class_name="click", sound_type="pulse_train"
            )
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.2,
            title="Single Short Event (fits in 1 window)",
            output_path=output_dir / "01_single_short_event.png",
        )

        assert len(merged) >= 1
        # Should have exactly 1 merged event
        assert len(merged) == 1

    def test_long_event_spanning_3_windows(self, output_dir):
        """A long event spanning 3 windows should merge into one event."""
        # window=3200ms, stride=2560ms -> windows at 0, 2560, 5120
        # Event from 1000-6500ms spans windows 0, 1, 2
        gt = [
            SyntheticEvent(
                start_ms=1000, end_ms=6500, freq_hz=1000, bandwidth_hz=300, class_name="whistle", sound_type="chirp_up"
            )
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.2,
            title="Long Whistle Spanning 3 Windows",
            output_path=output_dir / "02_long_event_3_windows.png",
        )

        assert len(merged) == 1
        m = merged[0]
        # Merged event should approximately cover the ground truth
        assert m.start_ms <= 1100  # within jitter tolerance
        assert m.end_ms >= 6300

    def test_two_separate_events(self, output_dir):
        """Two well-separated events should remain as two merged events."""
        gt = [
            SyntheticEvent(
                start_ms=500, end_ms=2000, freq_hz=800, bandwidth_hz=200, class_name="call_A", sound_type="chirp_exp"
            ),
            SyntheticEvent(
                start_ms=7000, end_ms=9000, freq_hz=2000, bandwidth_hz=400, class_name="call_B", sound_type="fm_tone"
            ),
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.2,
            title="Two Separate Events",
            output_path=output_dir / "03_two_separate_events.png",
        )

        assert len(merged) == 2

    def test_two_overlapping_different_freq(self, output_dir):
        """Two events at different frequencies overlapping in time should stay separate."""
        gt = [
            SyntheticEvent(
                start_ms=1000,
                end_ms=4000,
                freq_hz=800,
                bandwidth_hz=200,
                class_id=0,
                class_name="low_tone",
                sound_type="am_tone",
            ),
            SyntheticEvent(
                start_ms=1500,
                end_ms=4500,
                freq_hz=3000,
                bandwidth_hz=200,
                class_id=0,
                class_name="high_tone",
                sound_type="harmonic",
            ),
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.2,
            use_2d_iou=True,
            title="Two Overlapping Events at Different Frequencies (2D IoU)",
            output_path=output_dir / "04_different_freq_2d_iou.png",
        )

        # With 2D IoU, these should NOT merge (no freq overlap)
        assert len(merged) == 2

    def test_long_event_high_overlap(self, output_dir):
        """Long event with high window overlap (50%) should still merge correctly."""
        gt = [
            SyntheticEvent(
                start_ms=500,
                end_ms=8000,
                freq_hz=1500,
                bandwidth_hz=400,
                class_name="long_whistle",
                sound_type="chirp_down",
            )
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.5,
            title="Long Event with 50% Window Overlap",
            output_path=output_dir / "05_high_overlap_windows.png",
        )

        assert len(merged) == 1
        m = merged[0]
        assert m.start_ms <= 600
        assert m.end_ms >= 7800

    def test_many_events_complex_scene(self, output_dir):
        """Complex scene with multiple events, some overlapping in time."""
        gt = [
            SyntheticEvent(
                start_ms=200,
                end_ms=1800,
                freq_hz=600,
                bandwidth_hz=150,
                class_id=0,
                class_name="A",
                sound_type="pulse_train",
            ),
            SyntheticEvent(
                start_ms=3000,
                end_ms=7000,
                freq_hz=1200,
                bandwidth_hz=300,
                class_id=1,
                class_name="B",
                sound_type="chirp_up",
            ),
            SyntheticEvent(
                start_ms=4000,
                end_ms=5500,
                freq_hz=3500,
                bandwidth_hz=200,
                class_id=2,
                class_name="C",
                sound_type="harmonic",
            ),
            SyntheticEvent(
                start_ms=8000,
                end_ms=9500,
                freq_hz=800,
                bandwidth_hz=200,
                class_id=0,
                class_name="A",
                sound_type="fm_tone",
            ),
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.2,
            title="Complex Scene: 4 Events, Multiple Classes",
            output_path=output_dir / "06_complex_scene.png",
        )

        # Should have 4 separate events (different classes/frequencies)
        assert len(merged) == 4

    def test_back_to_back_events_with_gap_tolerance(self, output_dir):
        """Two back-to-back events should merge with gap tolerance."""
        gt = [
            SyntheticEvent(
                start_ms=1000, end_ms=3000, freq_hz=1000, bandwidth_hz=300, class_name="part1", sound_type="chirp_up"
            ),
            SyntheticEvent(
                start_ms=3100, end_ms=5000, freq_hz=1000, bandwidth_hz=300, class_name="part2", sound_type="chirp_exp"
            ),
        ]

        raw, merged = self._run_scenario(
            duration_ms=10000,
            sample_rate=16000,
            gt_events=gt,
            window_ms=3200,
            overlap_ratio=0.2,
            gap_tolerance_ms=200.0,
            title="Back-to-Back Events with Gap Tolerance (200ms)",
            output_path=output_dir / "07_gap_tolerance.png",
        )

        # The two close events should merge into one
        assert len(merged) == 1

    def test_temporal_only_vs_2d_iou(self, output_dir):
        """Compare temporal-only vs 2D IoU merge on same-time different-freq events."""
        events = EventList(
            events=[
                AudioEvent(
                    start_ms=1000,
                    end_ms=3000,
                    class_id=0,
                    score=0.9,
                    min_freq_hz=500,
                    max_freq_hz=800,
                    source_windows=[0],
                ),
                AudioEvent(
                    start_ms=1200,
                    end_ms=3200,
                    class_id=0,
                    score=0.85,
                    min_freq_hz=3000,
                    max_freq_hz=3500,
                    source_windows=[0],
                ),
            ],
            duration_ms=5000,
        )

        # Temporal-only: should merge (high temporal overlap)
        merged_1d = cluster_merge(events, iou_threshold=0.3, use_2d_iou=False)
        # 2D IoU: should NOT merge (no frequency overlap)
        merged_2d = cluster_merge(events, iou_threshold=0.3, use_2d_iou=True)

        assert len(merged_1d) == 1  # wrongly merged with temporal-only
        assert len(merged_2d) == 2  # correctly separated with 2D IoU


# -- Legacy compatibility tests -----------------------------------------------


class TestLegacyMergerCompatibility:
    """Ensure the new merger is backward-compatible with existing API."""

    def test_nms_merge_function(self):
        """nms_merge() convenience function should work."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.9, min_freq_hz=500, max_freq_hz=1000),
                AudioEvent(start_ms=500, end_ms=1500, class_id=0, score=0.8, min_freq_hz=500, max_freq_hz=1000),
                AudioEvent(start_ms=5000, end_ms=6000, class_id=0, score=0.7, min_freq_hz=500, max_freq_hz=1000),
            ]
        )
        merged = nms_merge(events, iou_threshold=0.2)
        assert len(merged) == 2

    def test_temporal_merge_function(self):
        """temporal_merge() convenience function should work."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.9),
                AudioEvent(start_ms=1050, end_ms=2000, class_id=0, score=0.8),
                AudioEvent(start_ms=5000, end_ms=6000, class_id=0, score=0.7),
            ]
        )
        merged = temporal_merge(events, gap_tolerance_ms=100.0)
        # First two should merge (50ms gap < 100ms tolerance), third stays
        assert len(merged) == 2

    def test_merge_config_defaults(self):
        """Default MergeConfig should have use_2d_iou=True."""
        config = MergeConfig()
        assert config.use_2d_iou is True
        assert config.iou_threshold == 0.5
        assert config.merge_strategy == "max"

    def test_event_list_metadata_preserved(self):
        """Merging should preserve EventList metadata."""
        events = EventList(
            events=[
                AudioEvent(start_ms=0, end_ms=1000, class_id=0, score=0.9),
            ],
            audio_path="/tmp/test.flac",
            duration_ms=10000,
            class_names={0: "whale"},
        )
        merged = cluster_merge(events, iou_threshold=0.3)
        assert merged.audio_path == "/tmp/test.flac"
        assert merged.duration_ms == 10000
        assert merged.class_names == {0: "whale"}
