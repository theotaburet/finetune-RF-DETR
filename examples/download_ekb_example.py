#!/usr/bin/env python3
"""Example usage of the EKB data downloader.

This script demonstrates how to use the DataDownloader class programmatically
to download human-annotated audio events with overlapping event grouping.

Example scenario from deepship_0.wav (30s):
    - ship at 0-10s and 15-30s
    - sonar beeps at 4.5-5.5s and 20.5-21.5s
    - humpback whale at 12.5-13s

Resulting sounds (overlapping events grouped):
    - Sound 1: ship (0-10s) + sonar beep (4.5-5.5s)
    - Sound 2: humpback whale (12.5-13s)
    - Sound 3: ship (15-30s) + sonar beep (20.5-21.5s)

"""

from pathlib import Path

# Import the downloader components
from run_download_data import DataDownloader, EKBAPIClient


def main():
    """Run the download example."""
    # Configuration
    API_URL = "https://api.example.com/ekb/api"  # Replace with your API URL
    TOKEN = "your_bearer_token_here"  # Replace with your API token
    OUTPUT_DIR = Path("data/deepship_example")

    # Initialize the API client
    api_client = EKBAPIClient(base_url=API_URL, token=TOKEN)

    # Initialize the downloader
    downloader = DataDownloader(
        api_client=api_client,
        output_dir=OUTPUT_DIR,
        dry_run=True,  # Set to False to actually download
        seed=42,  # For reproducibility
    )

    # Download ship sounds with confidence filtering
    results = downloader.run(
        sources=["deepship"],
        label_hierarchy="marine/ship",
        confidence_min=0.8,
        from_date="2024-01-01",
        max_labels=100,  # Limit for testing
    )

    print(f"\nDownloaded {len(results)} sound segments")

    # Example: Process the metadata
    for result in results[:3]:  # Show first 3
        print(f"\nSound: {result.sound_id}")
        print(f"  Duration: {result.end_s - result.start_s:.1f}s")
        print(f"  Events: {len(result.events)}")
        for event in result.events:
            print(f"    - {event['label_hierarchy']}: {event['start_s']:.1f}s - {event['end_s']:.1f}s")


if __name__ == "__main__":
    main()
