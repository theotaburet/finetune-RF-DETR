"""Legacy prediction function for RF-DETR models.

.. deprecated:: 0.3.0
    Use :class:`rf_detr_finetuning.predictor.RFDETRPredictor` instead.
    This module is a thin compatibility shim and will be removed in a future release.

"""

from __future__ import annotations

import warnings

import numpy as np


def prediction(
    image_path: str,
    model_size: str,
    model_path: str | None = None,
    confidence: float = 0.5,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Predict on an image and return an annotated BGR numpy array.

    .. deprecated:: 0.3.0
        Use :class:`rf_detr_finetuning.predictor.RFDETRPredictor` instead.

    Args:
        image_path: Path to the input image.
        model_size: Size of the RF-DETR model ('nano', 'small', 'base', 'medium', 'large').
        model_path: Path to the model checkpoint.
        confidence: Confidence threshold for predictions.
        class_names: Optional mapping from class id to class name.

    Returns:
        Annotated image as a BGR uint8 numpy array (for OpenCV/supervision compatibility).

    """
    warnings.warn(
        "prediction() is deprecated and will be removed in v0.3.0. "
        "Use rf_detr_finetuning.predictor.RFDETRPredictor instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    import matplotlib.pyplot as plt
    import supervision as sv
    from supervision import Color

    from rf_detr_finetuning.predictor import RFDETRPredictor

    # Convert dict class_names to list for RFDETRPredictor
    class_names_list: list[str] | None = None
    if class_names:
        max_id = max(class_names.keys())
        class_names_list = [class_names.get(i, str(i)) for i in range(max_id + 1)]

    predictor = RFDETRPredictor(
        model_size=model_size,
        weights_path=model_path,
        class_names=class_names_list,
    )

    result = predictor.predict(image_path, confidence_threshold=confidence)

    # Build labels from detections
    labels = []
    for det in result.detections:
        if det.class_name:
            labels.append(det.class_name)
        elif class_names and det.class_id in class_names:
            labels.append(class_names[det.class_id])
        else:
            labels.append(str(det.class_id))

    # Load image for annotation
    image = plt.imread(image_path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # Annotate (BGR for OpenCV compatibility, matching original behaviour)
    annotated = np.ascontiguousarray(image[:, :, ::-1])
    sv_detections = result.to_supervision()
    annotated = sv.BoxAnnotator().annotate(annotated, sv_detections)
    annotated = sv.LabelAnnotator(text_color=Color.RED).annotate(annotated, sv_detections, labels=labels)

    return annotated
