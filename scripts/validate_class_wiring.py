#!/usr/bin/env python3
"""Validate class ID wiring across configuration files.

This script checks that class IDs are properly wired between:
- config/pipeline.yaml (class_names list)
- config/merging.yaml (classes dict with integer keys)
- Any COCO annotation files

Usage:
    python scripts/validate_class_wiring.py
    python scripts/validate_class_wiring.py --pipeline-config config/pipeline.yaml
    python scripts/validate_class_wiring.py --check-annotations data/processed/train_annotations.json

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_pipeline_config(path: Path) -> dict:
    """Load pipeline configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_merging_config(path: Path) -> dict:
    """Load merging configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_coco_annotations(path: Path) -> dict:
    """Load COCO annotation file."""
    with open(path) as f:
        return json.load(f)


def validate_wiring(
    pipeline_config: dict,
    merging_config: dict | None = None,
    coco_annotations: dict | None = None,
) -> tuple[bool, list[str]]:
    """Validate class ID wiring.

    Args:
        pipeline_config: Pipeline configuration dictionary.
        merging_config: Optional merging configuration dictionary.
        coco_annotations: Optional COCO annotations dictionary.

    Returns:
        Tuple of (is_valid, list of messages).

    """
    messages = []
    is_valid = True

    # Get class names from pipeline config
    class_names = pipeline_config.get("class_names", [])

    if not class_names:
        messages.append("❌ ERROR: No class_names defined in pipeline config")
        is_valid = False
        return is_valid, messages

    messages.append(f"✓ Pipeline has {len(class_names)} classes: {class_names}")

    # Check merging config
    if merging_config:
        class_params = merging_config.get("classes", {})

        if not class_params:
            messages.append("⚠️  WARNING: No class-specific params in merging config")
            messages.append("   Will use default params for all classes")
        else:
            # Check that all class IDs are covered
            expected_ids = set(range(len(class_names)))
            actual_ids = set(int(k) for k in class_params.keys())

            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids

            if missing:
                missing_names = [class_names[i] for i in sorted(missing)]
                messages.append(f"❌ ERROR: Missing merge config for class_ids: {sorted(missing)}")
                messages.append(f"   Classes without merge params: {missing_names}")
                is_valid = False

            if extra:
                messages.append(f"⚠️  WARNING: Extra merge config for unknown class_ids: {sorted(extra)}")
                messages.append("   These will be ignored")

            if not missing and not extra:
                messages.append(f"✓ All {len(class_names)} classes have merge configuration")

            # Show mapping
            messages.append("\nClass ID mapping:")
            for i, name in enumerate(class_names):
                if i in class_params:
                    params = class_params[i]
                    dt = params.get("delta_time_ms", "default")
                    df = params.get("delta_freq_hz", "default")
                    messages.append(f"  {i}: {name:20s} → Δt={dt}ms, Δf={df}Hz")
                else:
                    messages.append(f"  {i}: {name:20s} → (using defaults)")

    # Check COCO annotations
    if coco_annotations:
        categories = coco_annotations.get("categories", [])

        if not categories:
            messages.append("⚠️  WARNING: No categories in COCO annotations")
        else:
            # COCO uses 1-based IDs
            coco_names = {cat["id"]: cat["name"] for cat in categories}
            coco_ids = set(coco_names.keys())

            # Expected COCO IDs (1-based)
            expected_coco_ids = set(range(1, len(class_names) + 1))

            missing_coco = expected_coco_ids - coco_ids
            extra_coco = coco_ids - expected_coco_ids

            if missing_coco:
                messages.append(f"❌ ERROR: Missing categories in COCO: {sorted(missing_coco)}")
                is_valid = False

            if extra_coco:
                messages.append(f"⚠️  WARNING: Extra categories in COCO: {sorted(extra_coco)}")

            if not missing_coco and not extra_coco:
                messages.append(f"✓ COCO categories match pipeline ({len(categories)} classes)")

            # Check name consistency
            messages.append("\nCOCO to Pipeline mapping:")
            for coco_id, coco_name in sorted(coco_names.items()):
                pipeline_idx = coco_id - 1  # Convert 1-based to 0-based
                if 0 <= pipeline_idx < len(class_names):
                    pipeline_name = class_names[pipeline_idx]
                    match = "✓" if coco_name == pipeline_name else "❌"
                    messages.append(
                        f"  COCO {coco_id} '{coco_name}' → Pipeline {pipeline_idx} '{pipeline_name}' {match}"
                    )
                else:
                    messages.append(f"  COCO {coco_id} '{coco_name}' → (out of range)")

    return is_valid, messages


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate class ID wiring across configuration files")
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--merging-config",
        type=Path,
        default=Path("config/merging.yaml"),
        help="Path to merging config YAML",
    )
    parser.add_argument(
        "--check-annotations",
        type=Path,
        help="Check COCO annotation file consistency",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Class ID Wiring Validation")
    print("=" * 80)
    print()

    # Load configs
    configs_to_check = []

    if args.pipeline_config.exists():
        configs_to_check.append(("Pipeline", args.pipeline_config, load_pipeline_config))
    else:
        print(f"❌ Pipeline config not found: {args.pipeline_config}")
        return 1

    if args.merging_config.exists():
        configs_to_check.append(("Merging", args.merging_config, load_merging_config))
    else:
        print(f"⚠️  Merging config not found: {args.merging_config}")
        print("   Skipping merge config validation")

    # Load them
    pipeline_config = None
    merging_config = None

    for name, path, loader in configs_to_check:
        try:
            config = loader(path)
            if name == "Pipeline":
                pipeline_config = config
            elif name == "Merging":
                merging_config = config
            print(f"✓ Loaded {name} config: {path}")
        except Exception as e:
            print(f"❌ Error loading {name} config: {e}")
            return 1

    # Load COCO if specified
    coco_annotations = None
    if args.check_annotations:
        if args.check_annotations.exists():
            try:
                coco_annotations = load_coco_annotations(args.check_annotations)
                print(f"✓ Loaded COCO annotations: {args.check_annotations}")
            except Exception as e:
                print(f"❌ Error loading COCO annotations: {e}")
                return 1
        else:
            print(f"❌ COCO annotations not found: {args.check_annotations}")
            return 1

    print()

    # Validate
    is_valid, messages = validate_wiring(
        pipeline_config=pipeline_config,
        merging_config=merging_config,
        coco_annotations=coco_annotations,
    )

    # Print messages
    for msg in messages:
        print(msg)

    print()
    print("=" * 80)

    if is_valid:
        print("✓ Class wiring validation PASSED")
        print("\nYou can now run:")
        print("  python run_pipeline.py --config config/pipeline.yaml")
        print("  python run_inference_audio.py --audio ... --merge-config config/merging.yaml ...")
        return 0
    else:
        print("❌ Class wiring validation FAILED")
        print("\nPlease fix the errors above before training or inference.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
