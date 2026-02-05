"""Example: Using frequency information for intelligent audio classification.

This demonstrates how to use the automatically decoded frequency information
from bbox coordinates to create frequency-aware features for model training.

The key insight: bbox pixel coordinates encode Hz and time information.
By decoding them, we can condition classification on frequency content.
"""

import json
from pathlib import Path


def load_coco_with_frequency_features(coco_json_path: Path) -> dict:
    """Load COCO annotations and extract frequency features.

    Args:
        coco_json_path: Path to _annotations.coco.json

    Returns:
        Dict with images, annotations, and extracted frequency features.

    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Extract frequency features for each annotation
    for ann in coco["annotations"]:
        # Get the frequency info that was stored during conversion
        image_id = ann["image_id"]
        image = next(img for img in coco["images"] if img["id"] == image_id)

        if "audio_metadata" in image and "frequency_info" in image["audio_metadata"]:
            freq_info = image["audio_metadata"]["frequency_info"]

            # Add frequency-aware features to annotation
            ann["freq_features"] = {
                "hz_center": freq_info["hz_center"],
                "bandwidth": freq_info["bandwidth_hz"],
                "frequency_band": freq_info["frequency_band"],
                "log_hz_center": __import__("math").log10(max(freq_info["hz_center"], 1.0)),
            }

    return coco


def create_frequency_aware_category(
    base_category: str,
    hz_center: float,
) -> str:
    """Create frequency-aware category name.

    Examples:
        - "whale_call" at 500 Hz -> "whale_call_low"
        - "sonar" at 8000 Hz -> "sonar_high"

    """
    if hz_center < 100:
        suffix = "infrasonic"
    elif hz_center < 500:
        suffix = "very_low"
    elif hz_center < 2000:
        suffix = "low"
    elif hz_center < 5000:
        suffix = "mid"
    elif hz_center < 10000:
        suffix = "high"
    else:
        suffix = "very_high"

    return f"{base_category}_{suffix}"


def extract_frequency_features_for_training(coco_dataset: dict) -> list[dict]:
    """Extract features that can be used as additional inputs to the model.

    These features can be concatenated with visual features before classification.

    Returns:
        List of feature dicts for each annotation.

    """
    features = []

    for ann in coco_dataset["annotations"]:
        if "freq_features" not in ann:
            continue

        freq = ann["freq_features"]

        # Create normalized features (0-1 range)
        feature_vec = {
            "log_hz_center_norm": freq["log_hz_center"] / 5.0,  # log10(22050) ≈ 4.34
            "bandwidth_norm": min(freq["bandwidth"] / 10000.0, 1.0),
            # One-hot encoding of frequency band
            "is_infrasonic": 1.0 if freq["frequency_band"] == "infrasonic" else 0.0,
            "is_very_low": 1.0 if freq["frequency_band"] == "very_low" else 0.0,
            "is_low": 1.0 if freq["frequency_band"] == "low" else 0.0,
            "is_mid": 1.0 if freq["frequency_band"] == "mid" else 0.0,
            "is_high": 1.0 if freq["frequency_band"] == "high" else 0.0,
            "is_very_high": 1.0 if freq["frequency_band"] == "very_high" else 0.0,
        }

        features.append(
            {
                "annotation_id": ann["id"],
                "category_id": ann["category_id"],
                "features": feature_vec,
            }
        )

    return features


# Example usage
if __name__ == "__main__":
    # Load dataset with frequency features
    coco_path = Path("data/geoacoustic_coco/train/_annotations.coco.json")

    if not coco_path.exists():
        print(f"COCO file not found: {coco_path}")
        print("Run audio-to-coco conversion first!")
        exit(1)

    coco = load_coco_with_frequency_features(coco_path)

    print(f"Loaded {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    # Extract features for training
    features = extract_frequency_features_for_training(coco)

    print(f"\nExtracted {len(features)} feature vectors")

    if features:
        print("\nExample feature vector:")
        example = features[0]
        print(f"  Annotation ID: {example['annotation_id']}")
        print(f"  Category ID: {example['category_id']}")
        print(f"  Features: {example['features']}")

    # Show frequency distribution
    print("\nFrequency band distribution:")
    bands = {}
    for ann in coco["annotations"]:
        if "freq_features" in ann:
            band = ann["freq_features"]["frequency_band"]
            bands[band] = bands.get(band, 0) + 1

    for band, count in sorted(bands.items()):
        print(f"  {band}: {count}")

    print("\n✅ Frequency-aware features ready for model training!")
    print("\nNext steps:")
    print("1. Create a custom RF-DETR classifier that takes these features")
    print("2. Concatenate frequency features with visual bbox features")
    print("3. Train the model to learn frequency-dependent patterns")
