"""Command-line interface for the RF-DETR training pipeline."""

import logging
import shutil
from pathlib import Path

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
from rf_detr_finetuning.trainer.rfdetr_wrapper import (
    get_model_sizes,
)


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


def train(config_file: str, dataset: str, model_size: str = "small") -> None:
    """Train the RF-DETR model using the provided YAML config and dataset path.

    Args:
        config_file: Path to the YAML training configuration file.
        dataset: Path to the prepared dataset directory.
        model_size: Size of the RF-DETR model to use (nano, small, base, medium, large).

    """
    from rf_detr_finetuning.trainer import (
        CheckpointConfig,
        OptimizerConfig,
        RFDETRConfig,
        RFDETRTrainer,
        TrainerConfig,
    )

    console = Console()

    valid_sizes = get_model_sizes()
    if model_size.lower() not in valid_sizes:
        console.print(f"[red]Invalid model size: {model_size!r}. Choose from: {valid_sizes}[/red]")
        return

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    # Build structured config from the flat YAML dict
    trainer_config = TrainerConfig(
        epochs=cfg.get("epochs", 10),
        batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("workers", 4),
        optimizer=OptimizerConfig(lr=cfg.get("lr", 1e-4)),
        checkpoint=CheckpointConfig(
            save_dir=cfg.get("project", "output") + "/" + cfg.get("name", "run"),
        ),
    )

    model_config = RFDETRConfig(
        model_size=model_size,
        image_size=cfg.get("imgsz", 640),
    )

    trainer = RFDETRTrainer(model_config=model_config, trainer_config=trainer_config)

    # Pass through any extra keys from the YAML as train_kwargs
    known_keys = {"epochs", "batch_size", "workers", "lr", "project", "name", "imgsz"}
    extra_kwargs = {k: v for k, v in cfg.items() if k not in known_keys}

    trainer.train(dataset_path=dataset, **extra_kwargs)


def predict(
    image_path: str,
    model_size: str = "small",
    model_path: str | None = None,
    confidence: float = 0.5,
    class_names: dict[int, str] | None = None,
) -> None:
    """Predict on an image using a pretrained or checkpoint RF-DETR model and display the result.

    Args:
        image_path: Path to the input image.
        model_size: Size of the RF-DETR model to use (nano, small, base, medium, large).
        model_path: Path to the model checkpoint or pretrained model name.
        confidence: Confidence threshold for predictions.
        class_names: Optional mapping from class id to class name.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    from rf_detr_finetuning.predictor import RFDETRPredictor

    valid_sizes = get_model_sizes()
    if model_size.lower() not in valid_sizes:
        logging.error(f"Invalid model size: {model_size!r}. Choose from: {valid_sizes}")
        return

    # Convert dict class_names to list for RFDETRPredictor
    class_names_list: list[str] | None = None
    if class_names:
        max_id = max(class_names.keys())
        class_names_list = [class_names.get(i, str(i)) for i in range(max_id + 1)]

    predictor = RFDETRPredictor(
        model_size=model_size,
        weights_path=model_path,
        class_names=class_names_list,
    )

    result = predictor.predict(image_path, confidence_threshold=confidence)

    # Build annotated image using supervision
    try:
        import supervision as sv
        from supervision import Color

        sv_detections = result.to_supervision()

        # Build labels
        labels = []
        for det in result.detections:
            if det.class_name:
                labels.append(det.class_name)
            elif class_names and det.class_id in class_names:
                labels.append(class_names[det.class_id])
            else:
                labels.append(str(det.class_id))

        # Load image for annotation
        image = plt.imread(image_path)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        annotated = np.ascontiguousarray(image[:, :, ::-1])
        annotated = sv.BoxAnnotator().annotate(annotated, sv_detections)
        annotated = sv.LabelAnnotator(text_color=Color.RED).annotate(annotated, sv_detections, labels=labels)

        # Display or save
        if plt.get_backend().lower() == "agg":
            output_path = Path("output/prediction.png")
            output_path.mkdir(parents=True, exist_ok=True) if not output_path.parent.exists() else None
            visual_to_save = annotated[:, :, ::-1] if annotated.ndim == 3 and annotated.shape[2] == 3 else annotated
            plt.imsave(str(output_path), visual_to_save)
            logging.info(f"Prediction saved to {output_path}")
        else:
            sv.plot_image(annotated)

    except ImportError:
        logging.warning("supervision not installed; printing detections as text")
        for det in result.detections:
            logging.info(
                "  [%.0f,%.0f,%.0f,%.0f] class=%s score=%.3f",
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                det.class_name or det.class_id,
                det.score,
            )


def audio_to_coco(
    input_dir: str,
    output_dir: str,
    split_ratios: str = "0.7,0.2,0.1",
    n_mels: int = 128,
    fft_ms: float = 25.0,
    hop_ms: float = 10.0,
    fmin: float = 0.0,
    fmax: float | None = None,
    frequency_bins: str | None = None,
    auto_infer_bins: int | None = None,
) -> str:
    """Convert audio dataset with JSON metadata to COCO format spectrograms.

    Args:
        input_dir: Input directory containing audio files (.flac, .wav, etc.) with matching .json metadata.
        output_dir: Output directory for the COCO-format dataset with spectrogram images.
        split_ratios: Train,valid,test split ratios as comma-separated values (default: "0.7,0.2,0.1").
        n_mels: Number of mel filterbanks (default: 128).
        fft_ms: FFT window duration in milliseconds (default: 25.0).
        hop_ms: Hop duration in milliseconds (default: 10.0).
        fmin: Minimum frequency for mel filterbank in Hz (default: 0.0).
        fmax: Maximum frequency for mel filterbank in Hz (default: sr/2).
        frequency_bins: Frequency bins for category splitting, format: 'min1,max1,name1;min2,max2,name2'.
            Example: '0,500,low;500,5000,mid;5000,22050,high' creates separate categories
            for events in different frequency bands (useful for distinguishing ship noise from sonar).
        auto_infer_bins: Automatically infer N frequency bins from the data distribution.
            Overrides frequency_bins if set. Example: 3 for low/mid/high bins.

    Returns:
        Path to the output directory.

    """
    ratios = tuple(float(x) for x in split_ratios.split(","))

    fft_config = TimeBasedFFTConfig(
        fft_ms=fft_ms,
        hop_ms=hop_ms,
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
        fft_config, chunk_config, preprocessing_config = load_chunking_config_from_yaml(config_file)
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

    chunker = AudioChunker(
        fft_config=fft_config,
        chunk_config=chunk_config,
        preprocessing_config=preprocessing_config,
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


def download_ekb_labels(
    api_url: str,
    token: str,
    output_dir: str,
    source: str | None = None,
    label_hierarchy: str | None = None,
    labeler: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    min_duration: float | None = None,
    max_frequency: int | None = None,
    max_labels: int | None = None,
    seed: int | None = None,
    dry_run: bool = False,
) -> None:
    """Download human-annotated audio events from EKB API.

    Groups overlapping events into minimal enclosing sounds. This helps the
    network learn overlapping events without penalizing it wrongly.

    Args:
        api_url: Base URL of the EKB API (e.g., "https://api.example.com/ekb/api").
        token: Bearer token for API authentication.
        output_dir: Directory to save downloaded audio files and metadata.
        source: Comma-separated list of source names to filter by.
        label_hierarchy: Filter by label hierarchy (partial match supported).
        labeler: Filter by labeler name/email.
        from_date: Filter labels created from this date (YYYY-MM-DD).
        to_date: Filter labels created up to this date (YYYY-MM-DD).
        confidence_min: Minimum confidence score (0.0-1.0).
        confidence_max: Maximum confidence score (0.0-1.0).
        min_duration: Minimum event duration in milliseconds.
        max_frequency: Maximum frequency in Hz.
        max_labels: Maximum number of labels to download.
        seed: Random seed for reproducibility.
        dry_run: Preview what would be downloaded without downloading.

    Example:
        # Download ship sounds with overlapping events grouped
        download_ekb_labels(
            api_url="https://api.example.com/ekb/api",
            token="your_token_here",
            output_dir="data/ships",
            label_hierarchy="marine/ship",
            seed=42,
        )

    """
    # Import directly instead of shelling out via subprocess
    from run_download_data import DataDownloader, DownloadConfig

    config = DownloadConfig(
        api_url=api_url,
        api_token=token,
        output_dir=output_dir,
        sources=[s.strip() for s in source.split(",")] if source else [],
        label_hierarchy=label_hierarchy or "",
        labeler=labeler or "",
        from_date=from_date or "",
        to_date=to_date or "",
        confidence_min=confidence_min,
        confidence_max=confidence_max,
        min_duration_ms=min_duration,
        max_frequency_hz=max_frequency,
        max_labels=max_labels,
        seed=seed or 42,
    )

    downloader = DataDownloader(config)
    downloader.run(dry_run=dry_run)


commands = {
    "download": {
        "kaggle-dataset": download_kaggle_dataset,
        "ekb-labels": download_ekb_labels,
    },
    "convert": {
        "yolo-to-coco": convert_yolo_to_coco,
        "audio-to-coco": audio_to_coco,
        "chunk-audio": chunk_audio,
    },
    "train": train,
    "predict": predict,
}
