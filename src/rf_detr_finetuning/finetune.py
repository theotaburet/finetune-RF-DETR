"""Finetuning utilities for RF-DETR models."""

import warnings

import torch
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

MAP_MODEL_SIZE = {
    "base": RFDETRBase,
    "small": RFDETRSmall,
    "nano": RFDETRNano,
    "large": RFDETRLarge,
    "medium": RFDETRMedium,
}


def finetune_model(model_size: str, dataset_path: str, config: dict) -> dict:
    """Finetune an RF-DETR model on a custom dataset.

    Loads the specified model size, updates the config with dataset path and device,
    and starts the training process. Results are logged and printed.

    Args:
        model_size: Size of the RF-DETR model to use (one of 'base', 'small', 'nano', 'large', 'medium').
        dataset_path: Path to the dataset directory containing images and annotations.
        config: Dictionary of training configuration parameters.

    """
    console = Console()

    # Suppress non-critical warnings
    warnings.filterwarnings("ignore", message=".*positional encodings.*")
    warnings.filterwarnings("ignore", message=".*patch size.*")
    warnings.filterwarnings("ignore", message=".*meshgrid.*")
    warnings.filterwarnings("ignore", message=".*multidimensional indexing.*")
    warnings.filterwarnings("ignore", message=".*lightning.*")

    assert model_size.lower() in MAP_MODEL_SIZE.keys(), f"Model size must be one of {list(MAP_MODEL_SIZE.keys())}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Display training configuration
    table = Table(title="🚀 RF-DETR Training Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("Model Size", model_size.upper())
    table.add_row("Dataset Path", dataset_path)
    table.add_row("Device", device.upper())
    table.add_row("Epochs", str(config.get("epochs", "N/A")))
    table.add_row("Batch Size", str(config.get("batch_size", "N/A")))
    table.add_row("Learning Rate", str(config.get("lr", "N/A")))
    table.add_row("Image Size", str(config.get("imgsz", "N/A")))
    table.add_row("Workers", str(config.get("workers", "N/A")))
    table.add_row("Output Dir", config.get("project", "output") + "/" + config.get("name", "run"))

    console.print()
    console.print(table)
    console.print()

    ModelClass = MAP_MODEL_SIZE[model_size.lower()]
    model = ModelClass()

    config["dataset_dir"] = dataset_path
    config["device"] = device

    console.print(Panel("[bold yellow]Starting training...[/bold yellow]", border_style="yellow"))
    console.print()

    # Just run training normally without stdout capture (it blocks the training)
    # RF-DETR will print its own progress
    results = model.train(**config)

    console.print()
    console.print(Panel("[bold green]✓ Training Complete![/bold green]", border_style="green"))
    console.print()

    return results
