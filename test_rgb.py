import numpy as np

from rf_detr_finetuning.dataprocessor.normalization import grayscale_to_rgb, normalize_to_range

spec = np.random.rand(128, 64, 3) * 255
spec = spec.astype(np.uint8)
normalized = normalize_to_range(spec, 0, 255)
rgb = grayscale_to_rgb(normalized.astype(np.uint8))
print(rgb.shape)
