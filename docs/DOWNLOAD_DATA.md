# Downloading Human-Annotated Data from EKB

The `run_download_data.py` script downloads human-annotated audio events from the EKB (Ezako Knowledge Base) API and groups overlapping events into minimal enclosing sounds. This helps train models to recognize overlapping events without being penalized for misclassifications.

## Why Group Overlapping Events?

Consider a 30-second audio file (`deepship_0.wav`) containing:

- Ship noise at 0-10s and 15-30s
- Sonar beeps at 4.5-5.5s and 20.5-21.5s
- Humpback whale song at 12.5-13s

Without grouping, the sonar beeps would be trained as "ship" sounds because they overlap with ship events. With grouping, we create minimal enclosing sounds:

1. **Sound 1**: ship (0-10s) + sonar beep (4.5-5.5s)
2. **Sound 2**: humpback whale (12.5-13s)
3. **Sound 3**: ship (15-30s) + sonar beep (20.5-21.5s)

This allows the model to learn that overlapping events are valid and distinct.

## Quick Start

### Command Line Usage

```bash
# Download all labels from a source
python run_download_data.py \
    --api-url https://api.example.com/ekb/api \
    --token YOUR_TOKEN \
    --source deepship \
    --output-dir data/downloaded

# Dry run to preview what would be downloaded
python run_download_data.py \
    --api-url https://api.example.com/ekb/api \
    --token YOUR_TOKEN \
    --source deepship \
    --output-dir data/downloaded \
    --dry-run

# Filter by label hierarchy
python run_download_data.py \
    --api-url https://api.example.com/ekb/api \
    --token YOUR_TOKEN \
    --label-hierarchy "marine/ship" \
    --output-dir data/ships

# Multiple filters with date range
python run_download_data.py \
    --api-url https://api.example.com/ekb/api \
    --token YOUR_TOKEN \
    --source source1,source2 \
    --from-date 2024-01-01 \
    --to-date 2024-06-30 \
    --confidence-min 0.8 \
    --seed 42 \
    --output-dir data/downloaded
```

### Python API Usage

```python
from pathlib import Path
from run_download_data import DataDownloader, EKBAPIClient

# Initialize API client
api_client = EKBAPIClient(
    base_url="https://api.example.com/ekb/api", token="your_token"
)

# Initialize downloader
downloader = DataDownloader(
    api_client=api_client,
    output_dir=Path("data/downloaded"),
    dry_run=False,  # Set True to preview only
    seed=42,  # For reproducibility
)

# Download with filters
results = downloader.run(
    sources=["deepship"],
    label_hierarchy="marine/ship",
    confidence_min=0.8,
    from_date="2024-01-01",
)

print(f"Downloaded {len(results)} sound segments")
```

## Command Line Arguments

### Required Arguments

- `--api-url`: Base URL of the EKB API (e.g., `https://api.example.com/ekb/api`)
- `--token`: Bearer token for API authentication
- `--output-dir`: Directory to save downloaded files and metadata

### Filter Arguments

- `--source`: Comma-separated list of source names to filter by
- `--label-hierarchy`: Filter by label hierarchy (supports partial matching)
- `--labeler`: Filter by labeler name/email
- `--from-date`: Filter labels created from this date (YYYY-MM-DD)
- `--to-date`: Filter labels created up to this date (YYYY-MM-DD)
- `--confidence-min`: Minimum confidence score (0.0-1.0)
- `--confidence-max`: Maximum confidence score (0.0-1.0)
- `--min-duration`: Minimum event duration in milliseconds
- `--max-frequency`: Maximum frequency in Hz

### Control Arguments

- `--max-labels`: Maximum number of labels to download
- `--seed`: Random seed for reproducibility
- `--dry-run`: Preview what would be downloaded without actually downloading
- `--verbose, -v`: Enable verbose logging

## Output Structure

```
data/downloaded/
├── audio/
│   ├── deepship_001_sound_0000.wav    # First sound segment
│   ├── deepship_001_sound_0001.wav    # Second sound segment
│   └── ...
└── metadata/
    └── download_metadata.json         # Complete download metadata
```

### Metadata Format

The `download_metadata.json` file contains:

```json
{
  "download_info": {
    "total_sounds": 150,
    "successful": 150,
    "failed": 0,
    "seed": 42,
    "dry_run": false,
    "filters": {
      "sources": ["deepship"],
      "label_hierarchy": "marine/ship"
    }
  },
  "sounds": [
    {
      "sound_id": "sound_0000",
      "source_file": "deepship_001.wav",
      "output_path": "audio/deepship_001_sound_0000.wav",
      "start_s": 0.0,
      "end_s": 10.5,
      "duration_s": 10.5,
      "events": [
        {
          "label_id": "uuid-1",
          "label_hierarchy": "marine/ship/engine",
          "start_s": 0.0,
          "end_s": 10.0,
          "confidence": 0.95
        },
        {
          "label_id": "uuid-2",
          "label_hierarchy": "marine/sonar/ping",
          "start_s": 4.5,
          "end_s": 5.5,
          "confidence": 0.88
        }
      ],
      "success": true
    }
  ]
}
```

## API Endpoints Used

The script uses the following EKB API endpoints:

- `GET /sdk/labels` - List labels with filtering
- `GET /sdk/labels/{label_id}` - Get label details
- `GET /sound/{annotation_id}` - Download audio file

## Algorithm

The overlapping event grouping uses an interval merging algorithm:

1. **Sort events** by start time
2. **Iterate through events**, maintaining a current sound segment
3. **If an event overlaps** with the current sound, extend the sound's end time
4. **If no overlap**, finalize the current sound and start a new one
5. **Result**: Minimal enclosing sounds containing all overlapping events

This ensures that events which share any temporal overlap are grouped together, while isolated events get their own sound segments.

## Error Handling

The script handles various error conditions:

- **API errors**: HTTP errors are caught and reported with response details
- **Download failures**: Individual sound download failures are logged but don't stop the process
- **Missing data**: Empty result sets are handled gracefully

## Tips

1. **Start with dry-run**: Always use `--dry-run` first to preview what will be downloaded
2. **Use seeds for reproducibility**: Set `--seed` when you want consistent results across runs
3. **Filter strategically**: Use `--source` and `--label-hierarchy` to download only relevant data
4. **Check metadata**: Review `download_metadata.json` after downloading to verify results
