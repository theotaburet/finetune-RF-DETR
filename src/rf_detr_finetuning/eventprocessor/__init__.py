"""Event processor module for audio detection post-processing.

Converts window-level detections to full-audio events with:
- Timestamp reconstruction
- Overlapping window handling
- NMS-like merging
- Confidence thresholding
- Event smoothing

"""

from rf_detr_finetuning.eventprocessor.event import (
    AudioEvent,
    EventList,
)
from rf_detr_finetuning.eventprocessor.merger import (
    EventMerger,
    MergeConfig,
    cluster_merge,
    nms_merge,
    temporal_merge,
)
from rf_detr_finetuning.eventprocessor.postprocessor import (
    EventPostProcessor,
    PostProcessorConfig,
    windows_to_events,
)

__all__ = [
    # Event representation
    "AudioEvent",
    "EventList",
    # Merging
    "MergeConfig",
    "EventMerger",
    "cluster_merge",
    "nms_merge",
    "temporal_merge",
    # Post-processing
    "PostProcessorConfig",
    "EventPostProcessor",
    "windows_to_events",
]
