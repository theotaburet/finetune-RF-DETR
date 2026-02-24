"""Event processor module for audio detection post-processing.

Converts window-level detections to full-audio events with:
- Timestamp reconstruction
- Overlapping window handling
- NMS-like merging
- Confidence thresholding
- Event smoothing
- Audio-level evaluation (temporal IoU matching)
- Merge parameter optimization (grid search)

"""

from rf_detr_finetuning.eventprocessor.evaluator import (
    ClassMetrics,
    EvaluationResult,
    MatchResult,
    evaluate_events,
    evaluate_multi_file,
    match_events,
)
from rf_detr_finetuning.eventprocessor.event import (
    AudioEvent,
    EventList,
)
from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
    ClassWiseMerger,
    EventMerger,
    MergeConfig,
    nms_merge,
    temporal_merge,
)
from rf_detr_finetuning.eventprocessor.optimizer import (
    GlobalSearchSpace,
    OptimizationResult,
    SearchSpace,
    config_to_yaml_dict,
    optimize_default_only,
    optimize_per_class,
    save_config_yaml,
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
    "nms_merge",
    "temporal_merge",
    # Class-wise merging
    "ClassMergeParams",
    "ClassWiseMergeConfig",
    "ClassWiseMerger",
    # Post-processing
    "PostProcessorConfig",
    "EventPostProcessor",
    "windows_to_events",
    # Evaluation
    "MatchResult",
    "ClassMetrics",
    "EvaluationResult",
    "match_events",
    "evaluate_events",
    "evaluate_multi_file",
    # Optimization
    "SearchSpace",
    "GlobalSearchSpace",
    "OptimizationResult",
    "optimize_default_only",
    "optimize_per_class",
    "config_to_yaml_dict",
    "save_config_yaml",
]
