"""Command-line interface for the RF-DETR training pipeline."""

import logging
import shutil
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import supervision as sv
import yaml

from rf_detr_finetuning.audio_to_coco import (
    SpectrogramConfig,
    convert_audio_to_coco,
    parse_frequency_bins,
)
from rf_detr_finetuning.data import convert_yolo_to_coco
from rf_detr_finetuning.finetune import MAP_MODEL_SIZE, finetune_model
from rf_detr_finetuning.predict import prediction


def download_kaggle_dataset(name: str, dest: str = "data", force: bool = False) -> str:
    """Download a Kaggle dataset into dest using the Kaggle API.

    Args:
        name: Name of the Kaggle dataset to download.
        dest: Destination directory for the downloaded dataset.
        force: Whether to force re-download if the dataset already exists.

    """
    logging.info(f"Starting download of '{name}' into '{dest}'")

    # Local import to keep dependency usage explicit and avoid import-time failures
    try:
        import kagglehub

        kagglehub.login()

        download_path = kagglehub.dataset_download(name, force_download=force)
        logging.info(f"Download complete: {download_path}")

        dest_path = Path(dest)
        dataset_path = dest_path / name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(download_path, dataset_path)
        logging.info(f"Dataset path: {dataset_path}")
        return str(dataset_path)
    except (ImportError, AttributeError) as exc:
        logging.warning(
            "kagglehub unavailable or incompatible; falling back to Kaggle API. Error: %s",
            exc,
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dest_path = Path(dest)
    dataset_path = dest_path / name
    dataset_path.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(name, path=str(dataset_path), unzip=True, force=force)
    logging.info(f"Dataset path: {dataset_path}")
    return str(dataset_path)


def train(config_file: str, dataset: str, model_size: Literal[tuple(MAP_MODEL_SIZE.keys())] = "small") -> None:
    """Train the RF-DETR model using the provided YAML config and dataset path.

    Args:
        config_file: Path to the YAML training configuration file.
        dataset: Path to the prepared dataset directory.
        model_size: Size of the RF-DETR model to use.

    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    finetune_model(model_size=model_size, dataset_path=dataset, config=cfg)

    # After training, try to display the metrics plot if it exists and GUI is available
    metrics_plot = Path("output/metrics_plot.png")
    if not metrics_plot.exists():
        return
    img = plt.imread(str(metrics_plot))
    plt.imshow(img)
    plt.title("Training Metrics")
    try:
        if plt.get_backend().lower() != "agg":
            plt.show()
    except Exception:
        logging.warning("GUI not available, skipping plot display.")


def predict(
    image_path: str,
    model_size: Literal[tuple(MAP_MODEL_SIZE.keys())] = "small",
    model_path: str | None = None,
    confidence: float = 0.5,
    class_names: dict[int, str] = None,
) -> None:
    """Predict on an image using a pretrained or checkpoint RF-DETR model and display the result.

    Args:
        image_path: Path to the input image.
        model_size: Size of the RF-DETR model to use.
        model_path: Path to the model checkpoint or pretrained model name.
        confidence: Confidence threshold for predictions.
        class_names: Optional mapping from class id to class name.

    """
    visual = prediction(
        image_path=image_path,
        model_size=model_size,
        model_path=model_path,
        confidence=confidence,
        class_names=class_names,
    )

    # Display or save the annotated image depending on backend
    if plt.get_backend().lower() == "agg":
        output_path = Path("output/prediction.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if visual.ndim == 3 and visual.shape[2] == 3:
            visual_to_save = visual[:, :, ::-1]
        else:
            visual_to_save = visual
        plt.imsave(str(output_path), visual_to_save)
        logging.info(f"Prediction saved to {output_path}")
        return
    sv.plot_image(visual)


def audio_to_coco(
    input_dir: str,
    output_dir: str,
    split_ratios: str = "0.7,0.2,0.1",
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    target_sr: int | None = None,
    frequency_bins: str | None = None,
    auto_infer_bins: int | None = None,
) -> str:
    """Convert audio dataset with JSON metadata to COCO format spectrograms.

    Args:
        input_dir: Input directory containing audio files (.flac, .wav, etc.) with matching .json metadata.
        output_dir: Output directory for the COCO-format dataset with spectrogram images.
        split_ratios: Train,valid,test split ratios as comma-separated values (default: "0.7,0.2,0.1").
        n_fft: FFT window size (default: 2048).
        hop_length: Hop length for STFT (default: 512).
        n_mels: Number of mel filterbanks (default: 128).
        fmin: Minimum frequency for mel filterbank in Hz (default: 0.0).
        fmax: Maximum frequency for mel filterbank in Hz (default: sr/2).
        target_sr: Target sample rate for resampling (default: keep original).
        frequency_bins: Frequency bins for category splitting, format: 'min1,max1,name1;min2,max2,name2'.
            Example: '0,500,low;500,5000,mid;5000,22050,high' creates separate categories
            for events in different frequency bands (useful for distinguishing ship noise from sonar).
        auto_infer_bins: Automatically infer N frequency bins from the data distribution.
            Overrides frequency_bins if set. Example: 3 for low/mid/high bins.

    Returns:
        Path to the output directory.

    """
    ratios = tuple(float(x) for x in split_ratios.split(","))

    spec_config = SpectrogramConfig(
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        target_sr=target_sr,
    )

    freq_bins = parse_frequency_bins(frequency_bins)

    result = convert_audio_to_coco(
        input_dir=input_dir,
        output_dir=output_dir,
        spec_config=spec_config,
        split_ratios=ratios,
        frequency_bins=freq_bins,
        auto_infer_bins=bool(auto_infer_bins) if auto_infer_bins else False,
    )

    logging.info(f"Audio dataset converted to COCO format at: {result}")
    return str(result)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
        "audio-to-coco": audio_to_coco,
    },
    "train": train,
    "predict": predict,
}
