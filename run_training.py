#!/usr/bin/env python3
"""Training CLI for RF-DETR audio detection.

Usage:
    python run_training.py --config config/audio_train.yaml
    python run_training.py --dataset-dir data/chunked_coco --epochs 20 --batch-size 8

"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Suppress warnings from external libraries
warnings.filterwarnings("ignore", message="torch.meshgrid:.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR for audio event detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to COCO-format dataset directory",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    # Model settings
    parser.add_argument(
        "--model-size",
        choices=["small", "base", "large"],
        default="base",
        help="RF-DETR model size",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained weights (path to .pth file, or None to train from scratch)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of object classes (auto-detected if not specified)",
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for logging",
    )

    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('cuda', 'cpu', 'auto')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def auto_detect_classes(dataset_dir: Path) -> int:
    """Auto-detect number of classes from dataset annotations.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        Number of classes.

    """
    import json

    # Look for annotation file
    candidates = [
        dataset_dir / "_annotations.coco.json",
        dataset_dir / "annotations.json",
        dataset_dir / "train" / "_annotations.coco.json",
    ]

    for ann_path in candidates:
        if ann_path.exists():
            with open(ann_path) as f:
                data = json.load(f)
            categories = data.get("categories", [])
            n_classes = len(categories)
            logger.info(f"Auto-detected {n_classes} classes from {ann_path}")
            return n_classes

    logger.warning("Could not auto-detect classes, defaulting to 1")
    return 1


def main() -> int:
    """Main training entrypoint."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]RF-DETR Audio Detection Training[/bold blue]\n")

    # Validate inputs
    if args.config is None and args.dataset_dir is None:
        console.print("[red]Error:[/red] Either --config or --dataset-dir must be specified")
        return 1

    # Load config if provided
    config_dict = {}
    if args.config and args.config.exists():
        import yaml

        with open(args.config) as f:
            config_dict = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {args.config}")

    # Determine dataset directory
    dataset_dir = args.dataset_dir or Path(config_dict.get("dataset_dir", "."))
    if not dataset_dir.exists():
        console.print(f"[red]Error:[/red] Dataset directory not found: {dataset_dir}")
        return 1

    # Auto-detect classes if needed
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = config_dict.get("num_classes")
    if num_classes is None:
        num_classes = auto_detect_classes(dataset_dir)

    # Create trainer
    from rf_detr_finetuning.trainer import create_rfdetr_trainer

    trainer = create_rfdetr_trainer(
        model_size=args.model_size,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        pretrained_weights=args.pretrained,
    )

    # Setup model
    console.print(f"[cyan]Model:[/cyan] RF-DETR {args.model_size}")
    console.print(f"[cyan]Classes:[/cyan] {num_classes}")
    console.print(f"[cyan]Epochs:[/cyan] {args.epochs}")
    console.print(f"[cyan]Batch size:[/cyan] {args.batch_size}")
    console.print(f"[cyan]Learning rate:[/cyan] {args.lr}")
    console.print()

    trainer.setup_model()

    # Resume if requested
    if args.resume:
        console.print(f"[yellow]Resuming from:[/yellow] {args.resume}")
        trainer.load_weights(args.resume)

    # Run training
    try:
        results = trainer.train(
            dataset_path=dataset_dir,
            output_dir=args.output_dir,
            grad_accum_steps=args.grad_accum,
            workers=args.workers,
            seed=args.seed,
        )
        console.print("\n[bold green]Training complete![/bold green]")

        # Display results
        if results:
            console.print("\n[bold]Results:[/bold]")
            for key, value in results.items():
                console.print(f"  {key}: {value}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Training failed:[/red] {e}")
        logger.exception("Training error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
