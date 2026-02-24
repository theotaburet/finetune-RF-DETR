"""Merge parameter optimizer for audio event detection.

Grid-searches over ClassWiseMergeConfig parameters (delta_time_ms, delta_freq_hz, score_threshold, min_duration_ms) to
maximize F1 on a labeled evaluation set.

The optimizer works on raw (unmerged) window detections. For each parameter combination it applies ClassWiseMerger, then
evaluates against ground truth using temporal IoU matching.

"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from rf_detr_finetuning.eventprocessor.evaluator import EvaluationResult, evaluate_multi_file
from rf_detr_finetuning.eventprocessor.event import AudioEvent, EventList
from rf_detr_finetuning.eventprocessor.merger import (
    ClassMergeParams,
    ClassWiseMergeConfig,
    ClassWiseMerger,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Parameter search space for a single class or default.

    Each list defines the candidate values to try. The optimizer will evaluate
    all combinations (Cartesian product) across these lists.

    Attributes:
        delta_time_ms: Candidate values for temporal merge distance.
        delta_freq_hz: Candidate values for frequency merge distance.
        score_strategy: Candidate score combination strategies.

    """

    delta_time_ms: list[float] = field(default_factory=lambda: [500.0, 1000.0, 2000.0, 4000.0])
    delta_freq_hz: list[float] = field(default_factory=lambda: [100.0, 200.0, 500.0])
    score_strategy: list[str] = field(default_factory=lambda: ["max"])

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations."""
        return len(self.delta_time_ms) * len(self.delta_freq_hz) * len(self.score_strategy)


@dataclass
class GlobalSearchSpace:
    """Search space for global (post-merge) filtering parameters.

    Attributes:
        score_threshold: Candidate minimum score thresholds.
        min_duration_ms: Candidate minimum event durations.

    """

    score_threshold: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    min_duration_ms: list[float] = field(default_factory=lambda: [0.0, 100.0, 200.0])

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations."""
        return len(self.score_threshold) * len(self.min_duration_ms)


@dataclass
class OptimizationResult:
    """Result of merge parameter optimization.

    Attributes:
        best_config: Best ClassWiseMergeConfig found.
        best_f1: Best overall F1 score achieved.
        best_eval: Full evaluation result for the best config.
        trials_evaluated: Number of parameter combinations tried.
        all_trials: List of (config_dict, f1) for every trial (optional).

    """

    best_config: ClassWiseMergeConfig | None = None
    best_f1: float = 0.0
    best_eval: EvaluationResult | None = None
    trials_evaluated: int = 0
    all_trials: list[tuple[dict[str, Any], float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "best_f1": round(self.best_f1, 4),
            "trials_evaluated": self.trials_evaluated,
        }
        if self.best_eval:
            result["best_eval"] = self.best_eval.to_dict()
        return result


def _build_config(
    class_ids: list[int],
    class_params_values: dict[int, dict[str, Any]],
    default_params_values: dict[str, Any],
    score_threshold: float,
    min_duration_ms: float,
) -> ClassWiseMergeConfig:
    """Build a ClassWiseMergeConfig from parameter values.

    Args:
        class_ids: List of class IDs to create per-class params for.
        class_params_values: Per-class parameter values.
        default_params_values: Default parameter values.
        score_threshold: Post-merge score threshold.
        min_duration_ms: Post-merge minimum duration.

    Returns:
        ClassWiseMergeConfig instance.

    """
    default_params = ClassMergeParams(
        delta_time_ms=default_params_values["delta_time_ms"],
        delta_freq_hz=default_params_values["delta_freq_hz"],
        score_strategy=default_params_values.get("score_strategy", "max"),
    )

    class_params = {}
    for cid in class_ids:
        if cid in class_params_values:
            vals = class_params_values[cid]
            class_params[cid] = ClassMergeParams(
                delta_time_ms=vals["delta_time_ms"],
                delta_freq_hz=vals["delta_freq_hz"],
                score_strategy=vals.get("score_strategy", "max"),
            )

    return ClassWiseMergeConfig(
        class_params=class_params,
        default_params=default_params,
        score_threshold=score_threshold,
        min_duration_ms=min_duration_ms,
    )


def _apply_merge(
    raw_events_per_file: list[EventList],
    config: ClassWiseMergeConfig,
) -> list[EventList]:
    """Apply class-wise merging to raw events for each file.

    Args:
        raw_events_per_file: Unmerged events per audio file.
        config: Merge configuration to apply.

    Returns:
        Merged events per file.

    """
    merger = ClassWiseMerger(config)
    return [merger.merge(events) for events in raw_events_per_file]


def optimize_default_only(
    raw_events_per_file: list[EventList],
    ground_truths_per_file: list[list[AudioEvent]],
    default_search: SearchSpace | None = None,
    global_search: GlobalSearchSpace | None = None,
    iou_threshold: float = 0.3,
    metric: str = "f1",
    class_names: dict[int, str] | None = None,
    progress_callback: Any = None,
) -> OptimizationResult:
    """Optimize default merge parameters (shared across all classes).

    Searches over all combinations of default delta_time, delta_freq,
    score_threshold, and min_duration. Per-class overrides are not used.

    Args:
        raw_events_per_file: Unmerged events per audio file.
        ground_truths_per_file: Ground truth events per audio file.
        default_search: Search space for default parameters.
        global_search: Search space for global filtering.
        iou_threshold: IoU threshold for evaluation matching.
        metric: Metric to optimize ("f1", "precision", "recall").
        class_names: Class name mapping.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        OptimizationResult with best parameters found.

    """
    default_search = default_search or SearchSpace()
    global_search = global_search or GlobalSearchSpace()

    total = default_search.num_combinations * global_search.num_combinations
    logger.info(f"Optimizing default params: {total} combinations")

    # Diagnostic: summarize input data
    total_raw = sum(len(e) for e in raw_events_per_file)
    total_gt = sum(len(gt) for gt in ground_truths_per_file)
    logger.info(f"Input: {len(raw_events_per_file)} files, {total_raw} raw predictions, {total_gt} ground truths")

    best = OptimizationResult()
    trial_idx = 0
    max_score_seen = 0.0

    for dt, df, ss in itertools.product(
        default_search.delta_time_ms,
        default_search.delta_freq_hz,
        default_search.score_strategy,
    ):
        for st, md in itertools.product(
            global_search.score_threshold,
            global_search.min_duration_ms,
        ):
            config = _build_config(
                class_ids=[],
                class_params_values={},
                default_params_values={"delta_time_ms": dt, "delta_freq_hz": df, "score_strategy": ss},
                score_threshold=st,
                min_duration_ms=md,
            )

            merged = _apply_merge(raw_events_per_file, config)

            file_pairs = list(zip(merged, ground_truths_per_file))
            eval_result = evaluate_multi_file(file_pairs, iou_threshold=iou_threshold, class_names=class_names)

            score = getattr(eval_result.overall, metric, eval_result.overall.f1)

            # Log first trial details for diagnostics
            if trial_idx == 0:
                total_merged = sum(len(m) for m in merged)
                logger.info(
                    f"First trial diagnostic: "
                    f"dt={dt}ms df={df}Hz st={st} md={md}ms => "
                    f"{total_merged} merged events, "
                    f"TP={eval_result.overall.true_positives} "
                    f"FP={eval_result.overall.false_positives} "
                    f"FN={eval_result.overall.false_negatives} "
                    f"F1={score:.4f}"
                )

            max_score_seen = max(max_score_seen, score)

            if score > best.best_f1:
                best.best_f1 = score
                best.best_config = config
                best.best_eval = eval_result

            trial_idx += 1
            if progress_callback:
                progress_callback(trial_idx, total)

    best.trials_evaluated = trial_idx

    if best.best_config is None:
        logger.warning(
            f"All {trial_idx} trials produced {metric}=0. "
            f"Max {metric} seen: {max_score_seen:.6f}. "
            f"This usually means predictions don't match ground truth "
            f"(check class ID mapping and detection quality)."
        )

    return best


def optimize_per_class(
    raw_events_per_file: list[EventList],
    ground_truths_per_file: list[list[AudioEvent]],
    class_ids: list[int],
    per_class_search: dict[int, SearchSpace] | None = None,
    default_search: SearchSpace | None = None,
    global_search: GlobalSearchSpace | None = None,
    iou_threshold: float = 0.3,
    class_names: dict[int, str] | None = None,
    progress_callback: Any = None,
) -> OptimizationResult:
    """Optimize per-class merge parameters independently.

    Strategy: first optimize global filtering and default params, then
    optimize each class independently while holding others at defaults.
    This avoids the combinatorial explosion of full per-class grid search.

    Args:
        raw_events_per_file: Unmerged events per audio file.
        ground_truths_per_file: Ground truth events per audio file.
        class_ids: Class IDs to optimize.
        per_class_search: Per-class search spaces (defaults used if absent).
        default_search: Search space for default parameters.
        global_search: Search space for global filtering.
        iou_threshold: IoU threshold for evaluation matching.
        class_names: Class name mapping.
        progress_callback: Optional callable(step_name, current, total).

    Returns:
        OptimizationResult with best per-class parameters.

    """
    default_search = default_search or SearchSpace()
    global_search = global_search or GlobalSearchSpace()
    per_class_search = per_class_search or {}
    class_names = class_names or {}

    # Phase 1: optimize defaults
    logger.info("Phase 1: optimizing default parameters")
    default_result = optimize_default_only(
        raw_events_per_file,
        ground_truths_per_file,
        default_search=default_search,
        global_search=global_search,
        iou_threshold=iou_threshold,
        class_names=class_names,
        progress_callback=lambda c, t: progress_callback("defaults", c, t) if progress_callback else None,
    )

    if default_result.best_config is None:
        return default_result

    best_default = default_result.best_config.default_params
    best_score_threshold = default_result.best_config.score_threshold
    best_min_duration = default_result.best_config.min_duration_ms

    logger.info(
        f"Phase 1 best: F1={default_result.best_f1:.4f} "
        f"dt={best_default.delta_time_ms}ms df={best_default.delta_freq_hz}Hz "
        f"score_thr={best_score_threshold} min_dur={best_min_duration}ms"
    )

    # Phase 2: optimize each class independently
    best_class_params: dict[int, dict[str, Any]] = {}

    for cid in class_ids:
        search = per_class_search.get(cid, default_search)
        class_name = class_names.get(cid, f"class_{cid}")
        logger.info(f"Phase 2: optimizing class {cid} ({class_name}): {search.num_combinations} combos")

        best_class_f1 = 0.0
        best_class_vals: dict[str, Any] | None = None

        trial_idx = 0
        for dt, df, ss in itertools.product(
            search.delta_time_ms,
            search.delta_freq_hz,
            search.score_strategy,
        ):
            class_vals = {"delta_time_ms": dt, "delta_freq_hz": df, "score_strategy": ss}
            current_class_params = {**best_class_params, cid: class_vals}

            config = _build_config(
                class_ids=list(current_class_params.keys()),
                class_params_values=current_class_params,
                default_params_values={
                    "delta_time_ms": best_default.delta_time_ms,
                    "delta_freq_hz": best_default.delta_freq_hz,
                    "score_strategy": best_default.score_strategy,
                },
                score_threshold=best_score_threshold,
                min_duration_ms=best_min_duration,
            )

            merged = _apply_merge(raw_events_per_file, config)
            file_pairs = list(zip(merged, ground_truths_per_file))
            eval_result = evaluate_multi_file(file_pairs, iou_threshold=iou_threshold, class_names=class_names)

            # Use per-class F1 for this specific class
            class_metric = eval_result.per_class.get(cid)
            class_f1 = class_metric.f1 if class_metric else 0.0

            if class_f1 > best_class_f1:
                best_class_f1 = class_f1
                best_class_vals = class_vals

            trial_idx += 1
            if progress_callback:
                progress_callback(f"class_{cid}", trial_idx, search.num_combinations)

        if best_class_vals:
            best_class_params[cid] = best_class_vals
            logger.info(
                f"  Class {cid} ({class_name}): F1={best_class_f1:.4f} "
                f"dt={best_class_vals['delta_time_ms']}ms df={best_class_vals['delta_freq_hz']}Hz"
            )
        else:
            logger.info(f"  Class {cid} ({class_name}): no improvement over defaults")

    # Build final config with all optimized parameters
    final_config = _build_config(
        class_ids=list(best_class_params.keys()),
        class_params_values=best_class_params,
        default_params_values={
            "delta_time_ms": best_default.delta_time_ms,
            "delta_freq_hz": best_default.delta_freq_hz,
            "score_strategy": best_default.score_strategy,
        },
        score_threshold=best_score_threshold,
        min_duration_ms=best_min_duration,
    )

    # Final evaluation with optimized config
    merged = _apply_merge(raw_events_per_file, final_config)
    file_pairs = list(zip(merged, ground_truths_per_file))
    final_eval = evaluate_multi_file(file_pairs, iou_threshold=iou_threshold, class_names=class_names)

    result = OptimizationResult(
        best_config=final_config,
        best_f1=final_eval.overall.f1,
        best_eval=final_eval,
        trials_evaluated=default_result.trials_evaluated
        + sum((per_class_search.get(cid, default_search)).num_combinations for cid in class_ids),
    )

    logger.info(f"Final optimized F1={final_eval.overall.f1:.4f}")
    return result


def config_to_yaml_dict(config: ClassWiseMergeConfig, class_names: dict[int, str] | None = None) -> dict[str, Any]:
    """Convert a ClassWiseMergeConfig to a YAML-compatible dict.

    The output format matches config/merging.yaml structure.

    Args:
        config: Merge configuration.
        class_names: Optional class name mapping for comments.

    Returns:
        Dictionary ready for yaml.dump().

    """
    class_names = class_names or {}

    result: dict[str, Any] = {
        "default": {
            "delta_time_ms": config.default_params.delta_time_ms,
            "delta_freq_hz": config.default_params.delta_freq_hz,
            "score_strategy": config.default_params.score_strategy,
        },
        "classes": {},
        "filtering": {
            "score_threshold": config.score_threshold,
            "min_duration_ms": config.min_duration_ms,
            "max_duration_ms": config.max_duration_ms,
        },
    }

    if config.default_params.min_overlap_ratio is not None:
        result["default"]["min_overlap_ratio"] = config.default_params.min_overlap_ratio

    for cid, params in sorted(config.class_params.items()):
        entry: dict[str, Any] = {
            "delta_time_ms": params.delta_time_ms,
            "delta_freq_hz": params.delta_freq_hz,
            "score_strategy": params.score_strategy,
        }
        if params.min_overlap_ratio is not None:
            entry["min_overlap_ratio"] = params.min_overlap_ratio
        result["classes"][cid] = entry

    return result


def save_config_yaml(
    config: ClassWiseMergeConfig,
    output_path: str,
    class_names: dict[int, str] | None = None,
) -> None:
    """Save optimized merge config to YAML file.

    Args:
        config: Optimized merge configuration.
        output_path: Path to write YAML file.
        class_names: Optional class name mapping for comments.

    """
    from pathlib import Path

    class_names = class_names or {}

    yaml_dict = config_to_yaml_dict(config, class_names)

    # Build YAML with comments for class names
    lines = ["# Event Merging Configuration (auto-optimized)", "#"]

    if class_names:
        lines.append("# Class mapping:")
        for cid in sorted(class_names.keys()):
            lines.append(f"#   {cid}: {class_names[cid]}")
        lines.append("#")

    lines.append("")

    # Write default section
    lines.append("# Default merge parameters (applied to all classes unless overridden)")
    lines.append("default:")
    lines.append(f"  delta_time_ms: {yaml_dict['default']['delta_time_ms']}")
    lines.append(f"  delta_freq_hz: {yaml_dict['default']['delta_freq_hz']}")
    if "min_overlap_ratio" in yaml_dict["default"]:
        lines.append(f"  min_overlap_ratio: {yaml_dict['default']['min_overlap_ratio']}")
    else:
        lines.append("  min_overlap_ratio: null")
    lines.append(f'  score_strategy: "{yaml_dict["default"]["score_strategy"]}"')

    # Write class-specific section
    lines.append("")
    lines.append("# Class-specific merge parameters")
    lines.append("classes:")
    for cid, params in sorted(yaml_dict["classes"].items()):
        name_comment = f"  # {class_names[cid]}" if cid in class_names else ""
        lines.append(f"  {cid}:{name_comment}")
        lines.append(f"    delta_time_ms: {params['delta_time_ms']}")
        lines.append(f"    delta_freq_hz: {params['delta_freq_hz']}")
        lines.append(f'    score_strategy: "{params["score_strategy"]}"')
        lines.append("")

    # Write filtering section
    lines.append("# Post-merge filtering")
    lines.append("filtering:")
    lines.append(f"  score_threshold: {yaml_dict['filtering']['score_threshold']}")
    lines.append(f"  min_duration_ms: {yaml_dict['filtering']['min_duration_ms']}")
    max_dur = yaml_dict["filtering"]["max_duration_ms"]
    lines.append(f"  max_duration_ms: {max_dur if max_dur is not None else 'null'}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    logger.info(f"Saved optimized config to {output_path}")
