#!/usr/bin/env python3
"""Complete audio training workflow example.

This script demonstrates the full pipeline:
1. Chunk audio dataset into fixed-size spectrograms
2. Split into train/val/test
3. Train RF-DETR model
4. Run inference and visualize results

"""

from pathlib import Path

from rf_detr_finetuning import (
    AudioChunker,
    ChunkConfig,
    TimeBasedFFTConfig,
    draw_bboxes_on_spectrogram,
    finetune_model,
    prediction,
)


def step1_chunk_audio():
    """Step 1: Chunk variable-length audio into fixed-size spectrograms."""
    print("=" * 60)
    print("STEP 1: Chunking Audio Dataset")
    print("=" * 60)

    # Configure time-based FFT (sample-rate independent)
    fft_config = TimeBasedFFTConfig(
        fft_ms=25.0,  # 25ms FFT window
        hop_ms=10.0,  # 10ms hop
        n_mels=128,  # Mel filterbanks
    )

    # Configure chunking
    chunk_config = ChunkConfig(
        window_duration_ms=5000.0,  # 5-second chunks
        overlap_ms=1000.0,  # 1-second overlap
        target_width=640,  # RF-DETR standard size
        target_height=640,
        padding_mode="zero",  # Pad short audio with zeros
        min_overlap_with_event_ratio=0.3,  # Event needs 30% in chunk
    )

    chunker = AudioChunker(fft_config=fft_config, chunk_config=chunk_config)

    # Process audio files
    input_dir = Path("data/my_audio_dataset")
    output_dir = Path("data/chunked_coco")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob("*.flac")) + list(input_dir.glob("*.wav"))

    print(f"Found {len(audio_files)} audio files")
    print(f"Output directory: {output_dir}")

    # Chunk each audio file
    all_chunks = []
    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        chunks = chunker.chunk_audio_file(audio_file)
        all_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks")

    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    print(f"✓ Output saved to: {output_dir}")

    # Optional: Draw debug visualizations
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    for i, chunk in enumerate(all_chunks[:5]):  # First 5 chunks
        if chunk.bboxes:
            debug_img = draw_bboxes_on_spectrogram(chunk.spectrogram, chunk.bboxes)
            debug_path = debug_dir / f"chunk_{i:04d}_debug.png"
            from PIL import Image

            Image.fromarray(debug_img).save(debug_path)
            print(f"  Debug image saved: {debug_path}")

    return output_dir


def step2_split_dataset(chunked_dir: Path):
    """Step 2: Split into train/val/test."""
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Dataset")
    print("=" * 60)

    # Use the split script
    import subprocess

    result = subprocess.run(
        [
            "python",
            "scripts/split_coco_dataset.py",
            "--input-coco",
            str(chunked_dir / "_annotations.coco.json"),
            "--input-images",
            str(chunked_dir / "images"),
            "--output-dir",
            str(chunked_dir.parent / "chunked_coco_split"),
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
        ],
        check=True,
    )
    print(f"\n✓ Dataset split completed with return code {result.returncode}")
    print("\n✓ Dataset split complete")
    return chunked_dir.parent / "chunked_coco_split"


def step3_train_model(split_dir: Path):
    """Step 3: Train RF-DETR model."""
    print("\n" + "=" * 60)
    print("STEP 3: Training RF-DETR Model")
    print("=" * 60)

    # Training configuration
    config = {
        "model": {"num_classes": 2},  # UPDATE based on your categories
        "training": {
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 0.0001,
        },
        "data": {
            "train_ann_file": "train/_annotations.coco.json",
            "val_ann_file": "valid/_annotations.coco.json",
        },
    }

    print("Training configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")

    # Train model
    finetune_model(
        model_size="small",
        dataset_path=str(split_dir),
        config=config,
    )

    print("\n✓ Training complete")
    return Path("output/checkpoint_best_total.pth")


def step4_test_and_visualize(split_dir: Path, model_path: Path):
    """Step 4: Test and visualize results."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing and Visualization")
    print("=" * 60)

    test_dir = split_dir / "test"
    output_dir = Path("output/test_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get test images
    test_images = list(test_dir.glob("*.png"))
    test_images = [img for img in test_images if not img.stem.endswith("_debug")][:10]

    print(f"Testing on {len(test_images)} images")

    # Run inference
    for img_path in test_images:
        print(f"Processing: {img_path.name}")

        visual = prediction(
            image_path=str(img_path),
            model_size="small",
            model_path=str(model_path),
            confidence=0.5,
            class_names={1: "class_1", 2: "class_2"},  # UPDATE
        )

        # Save result
        output_path = output_dir / f"{img_path.stem}_pred.png"
        import numpy as np
        from PIL import Image

        Image.fromarray(visual.astype(np.uint8)).save(output_path)
        print(f"  Saved: {output_path}")

    print(f"\n✓ Test predictions saved to: {output_dir}")


def main():
    """Run complete workflow."""
    print("Starting Audio Training Workflow")
    print("=" * 60)

    # Step 1: Chunk audio
    # chunked_dir = step1_chunk_audio()

    # Step 2: Split dataset
    # split_dir = step2_split_dataset(chunked_dir)

    # Step 3: Train model
    # model_path = step3_train_model(split_dir)

    # Step 4: Test and visualize
    # step4_test_and_visualize(split_dir, model_path)

    print("\n" + "=" * 60)
    print("Workflow complete! Check output/ for results.")
    print("=" * 60)

    print("\nNote: Uncomment steps in main() to run the full pipeline.")
    print("This example shows the structure. Adapt paths and configs for your data.")


if __name__ == "__main__":
    main()
