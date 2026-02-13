#!/usr/bin/env python3
"""Generate test events for run_merging.py testing."""

import json
from pathlib import Path

# Create output directory
output_dir = Path("output/test_merging")
output_dir.mkdir(parents=True, exist_ok=True)

# Create fake events in EventList format
events_data = {
    "audio_path": "fake_audio.wav",
    "duration_ms": 11000,
    "num_events": 6,
    "class_names": {0: "bird_call", 1: "engine_noise", 2: "dog_bark"},
    "events": [
        {
            "start_ms": 500,
            "end_ms": 1500,
            "duration_ms": 1000,
            "class_id": 0,
            "class_name": "bird_call",
            "score": 0.85,
            "min_freq_hz": 2000,
            "max_freq_hz": 4000,
            "source_windows": [0],
            "metadata": {},
        },
        {
            "start_ms": 2000,
            "end_ms": 2800,
            "duration_ms": 800,
            "class_id": 1,
            "class_name": "engine_noise",
            "score": 0.92,
            "min_freq_hz": 100,
            "max_freq_hz": 800,
            "source_windows": [0],
            "metadata": {},
        },
        {
            "start_ms": 600,
            "end_ms": 1600,
            "duration_ms": 1000,
            "class_id": 0,
            "class_name": "bird_call",
            "score": 0.88,
            "min_freq_hz": 2000,
            "max_freq_hz": 4000,
            "source_windows": [1],
            "metadata": {},
        },
        {
            "start_ms": 2100,
            "end_ms": 2900,
            "duration_ms": 800,
            "class_id": 1,
            "class_name": "engine_noise",
            "score": 0.89,
            "min_freq_hz": 100,
            "max_freq_hz": 800,
            "source_windows": [1],
            "metadata": {},
        },
        {
            "start_ms": 700,
            "end_ms": 1400,
            "duration_ms": 700,
            "class_id": 0,
            "class_name": "bird_call",
            "score": 0.75,
            "min_freq_hz": 2000,
            "max_freq_hz": 4000,
            "source_windows": [2],
            "metadata": {},
        },
        {
            "start_ms": 900,
            "end_ms": 2500,
            "duration_ms": 1600,
            "class_id": 2,
            "class_name": "dog_bark",
            "score": 0.95,
            "min_freq_hz": 500,
            "max_freq_hz": 1500,
            "source_windows": [3],
            "metadata": {},
        },
    ],
}

output_file = output_dir / "test_events.json"
with open(output_file, "w") as f:
    json.dump(events_data, f, indent=2)

print(f"Created test events file: {output_file}")
print(f"Total events: {len(events_data['events'])}")
print("\nEvents:")
for i, event in enumerate(events_data["events"], 1):
    print(f"  {i}. {event['class_name']}: {event['start_ms']}-{event['end_ms']}ms (score={event['score']:.2f})")
