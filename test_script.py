import numpy as np
import torch
from ezakodio.viz import spectrogram_to_image

spec = np.random.rand(128, 64).astype(np.float32)
spec_img = spectrogram_to_image(torch.from_numpy(spec), cmap="jet", normalize=True)
print(spec_img.shape, spec_img.dtype)
