"""CLI entry point for the RF-DETR training pipeline."""

import logging
import warnings

from rich.logging import RichHandler

from rf_detr_finetuning.cli import commands

if __name__ == "__main__":
    # Configure Rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    # Suppress non-critical warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(commands, as_positional=False)
