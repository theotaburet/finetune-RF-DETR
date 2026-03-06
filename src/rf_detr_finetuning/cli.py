"""Command-line interface for the RF-DETR training pipeline."""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import supervision as sv
import yaml
from rich.console import Console

from rf_detr_finetuning.data import convert_yolo_to_coco
from rf_detr_finetuning.dataloader import (
    convert_audio_to_coco,
    parse_frequency_bins,
)
from rf_detr_finetuning.dataprocessor import (
    AudioChunker,
    ChunkConfig,
    TimeBasedFFTConfig,
    draw_bboxes_on_spectrogram,
    load_chunking_config_from_yaml,
)
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
    console = Console()

    # Suppress PyTorch and RF-DETR warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TensorBoard.*")

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    finetune_model(model_size=model_size, dataset_path=dataset, config=cfg)

    # After training, try to display the metrics plot if it exists and GUI is available
    metrics_plot = Path("output/metrics_plot.png")
    if not metrics_plot.exists():
        return

    console.print("\n[cyan]Training metrics plot available at:[/cyan]", metrics_plot)

    try:
        img = plt.imread(str(metrics_plot))
        plt.imshow(img)
        plt.title("Training Metrics")
        if plt.get_backend().lower() != "agg":
            plt.show()
    except Exception:
        pass


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
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    frequency_bins: str | None = None,
) -> str:
    """Convert audio dataset with JSON metadata to COCO format spectrograms.

    Args:
        input_dir: Input directory containing audio files (.flac, .wav, etc.) with matching .json metadata.
        output_dir: Output directory for the COCO-format dataset with spectrogram images.
        split_ratios: Train,valid,test split ratios as comma-separated values (default: "0.7,0.2,0.1").
        n_mels: Number of mel filterbanks (default: 128).
        fmin: Minimum frequency for mel filterbank in Hz (default: 0.0).
        fmax: Maximum frequency for mel filterbank in Hz (default: sr/2).
        frequency_bins: Frequency bins for category splitting, format: 'min1,max1,name1;min2,max2,name2'.
            Example: '0,500,low;500,5000,mid;5000,22050,high' creates separate categories
            for events in different frequency bands (useful for distinguishing ship noise from sonar).

    Returns:
        Path to the output directory.

    """
    ratios = tuple(float(x) for x in split_ratios.split(","))

    # Create AudioChunker from parameters
    fft_config = TimeBasedFFTConfig(
        fft_ms=25.0,
        hop_ms=10.0,
        n_mels=n_mels,
    )

    chunker = AudioChunker(
        fft_config=fft_config,
        fmin=fmin,
        fmax=fmax,
    )

    freq_bins = parse_frequency_bins(frequency_bins)

    result = convert_audio_to_coco(
        input_dir=input_dir,
        output_dir=output_dir,
        chunker=chunker,
        split_ratios=ratios,
        frequency_bins=freq_bins,
    )

    logging.info(f"Audio dataset converted to COCO format at: {result}")
    return str(result)


def chunk_audio(
    input_dir: str,
    output_dir: str,
    config_file: str | None = None,
    window_duration_ms: float = 5000.0,
    overlap_ms: float = 1000.0,
    target_width: int = 640,
    target_height: int = 640,
    fft_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 128,
    padding_mode: str = "zero",
    min_overlap_ratio: float = 0.3,
    draw_bboxes: bool = False,
    debug_output_dir: str | None = None,
) -> str:
    """Chunk audio files into fixed-size spectrograms with aligned bboxes.

    This command processes audio files with JSON metadata (containing bbox annotations)
    and creates fixed-size spectrogram chunks suitable for RF-DETR training.
    Each chunk has bboxes transformed to chunk-relative coordinates.

    Args:
        input_dir: Input directory containing audio files with matching .json metadata.
        output_dir: Output directory for chunked spectrograms and COCO annotations.
        config_file: Path to YAML config file. Overrides other CLI arguments if provided.
        window_duration_ms: Duration of each chunk in milliseconds (default: 5000).
        overlap_ms: Overlap between consecutive chunks in milliseconds (default: 1000).
        target_width: Target width for output spectrograms (default: 640).
        target_height: Target height for output spectrograms (default: 640).
        fft_ms: FFT window duration in milliseconds (default: 25.0).
        hop_ms: Hop duration in milliseconds (default: 10.0).
        n_mels: Number of mel filterbanks (default: 128).
        padding_mode: Padding mode for short audio: 'zero', 'repeat', or 'reflect' (default: 'zero').
        min_overlap_ratio: Minimum overlap ratio with event bbox to include in chunk (default: 0.3).
        draw_bboxes: Whether to draw debug images with bboxes overlaid (default: False).
        debug_output_dir: Output directory for debug bbox images (default: output_dir/debug).

    Returns:
        Path to the output directory.

    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config from YAML or use CLI arguments
    if config_file:
        console.print(f"[cyan]Loading config from:[/cyan] {config_file}")
        fft_config, chunk_config, preprocessing_config, spec_config = load_chunking_config_from_yaml(config_file)
    else:
        fft_config = TimeBasedFFTConfig(
            fft_ms=fft_ms,
            hop_ms=hop_ms,
            n_mels=n_mels,
        )
        overlap_ratio = overlap_ms / window_duration_ms if window_duration_ms > 0 else 0.0
        chunk_config = ChunkConfig(
            window_duration_ms=window_duration_ms,
            overlap_ratio=overlap_ratio,
            target_width=target_width,
            target_height=target_height,
            padding_mode=padding_mode,
            min_overlap_with_event_ratio=min_overlap_ratio,
        )
        preprocessing_config = None  # No preprocessing when using CLI args
        spec_config = {"freq_scale": "mel", "fmin": 0.0, "fmax": None}

    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preprocessing_config,
        freq_scale=spec_config["freq_scale"],
        fmin=spec_config["fmin"],
        fmax=spec_config["fmax"],
    )

    # Find audio files
    input_path = Path(input_dir)
    audio_extensions = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
    audio_files = [f for f in input_path.rglob("*") if f.suffix.lower() in audio_extensions]

    if not audio_files:
        console.print(f"[red]No audio files found in:[/red] {input_dir}")
        return str(output_path)

    console.print(f"[green]Found {len(audio_files)} audio files[/green]")

    # Prepare COCO dataset structure
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    categories_seen: dict[str, int] = {}
    annotation_id = 1
    image_id = 1

    debug_dir = Path(debug_output_dir) if debug_output_dir else output_path / "debug"
    if draw_bboxes:
        debug_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing audio files...", total=len(audio_files))

        for audio_file in audio_files:
            progress.update(task, description=f"[cyan]Chunking: {audio_file.name}")

            try:
                chunks = chunker.chunk_audio_file(audio_file)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process {audio_file}: {e}[/yellow]")
                progress.advance(task)
                continue

            for chunk in chunks:
                # Save spectrogram image
                chunk_filename = f"{audio_file.stem}_chunk{chunk.chunk_index:04d}.png"
                chunk_path = output_path / "images" / chunk_filename
                chunk_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert spectrogram to image (normalize to 0-255)
                spec_min = chunk.spectrogram.min()
                spec_max = chunk.spectrogram.max()
                if spec_max > spec_min:
                    normalized = (chunk.spectrogram - spec_min) / (spec_max - spec_min)
                else:
                    normalized = chunk.spectrogram - spec_min
                img_array = (normalized * 255).astype("uint8")

                # Save using PIL
                from PIL import Image

                img = Image.fromarray(img_array)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(chunk_path)

                # Add to COCO images
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{chunk_filename}",
                        "width": chunk.spectrogram.shape[1],
                        "height": chunk.spectrogram.shape[0],
                    }
                )

                # Add bboxes as annotations
                for bbox in chunk.bboxes:
                    # Get or create category
                    cat_name = bbox.category
                    if cat_name not in categories_seen:
                        categories_seen[cat_name] = len(categories_seen) + 1
                    cat_id = categories_seen[cat_name]

                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": bbox.to_coco_bbox(),
                            "area": bbox.to_coco_bbox()[2] * bbox.to_coco_bbox()[3],
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

                # Draw debug images if requested
                if draw_bboxes and chunk.bboxes:
                    debug_path = debug_dir / f"{audio_file.stem}_chunk{chunk.chunk_index:04d}_debug.png"
                    debug_img = draw_bboxes_on_spectrogram(chunk.spectrogram, chunk.bboxes)
                    Image.fromarray(debug_img).save(debug_path)

                image_id += 1

            progress.advance(task)

    # Build categories list
    coco_categories = [
        {"id": cat_id, "name": cat_name} for cat_name, cat_id in sorted(categories_seen.items(), key=lambda x: x[1])
    ]

    # Save COCO annotations
    coco_dataset = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    import json

    annotations_path = output_path / "_annotations.coco.json"
    with open(annotations_path, "w") as f:
        json.dump(coco_dataset, f, indent=2)

    console.print(f"[green]✓ Created {len(coco_images)} chunks with {len(coco_annotations)} annotations[/green]")
    console.print(f"[green]✓ Categories: {list(categories_seen.keys())}[/green]")
    console.print(f"[green]✓ COCO annotations saved to: {annotations_path}[/green]")
    if draw_bboxes:
        console.print(f"[green]✓ Debug images saved to: {debug_dir}[/green]")

    return str(output_path)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
        "audio-to-coco": audio_to_coco,
        "chunk-audio": chunk_audio,
    },
    "train": train,
    "predict": predict,
}
