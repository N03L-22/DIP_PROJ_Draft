"""Small image helpers shared by the degradation models (smooth noise field, mixing). Blur lives in common/filters.py."""
import numpy as np

from common.filters import gaussian_blur
from common.rng import mulberry32_array


def smooth_noise_field(height, width, sigma, seed):
    """A smooth random field in [0, 1]: white noise blurred by `sigma` pixels, then stretched to fill 0..1.

    Used for the glass veil and the smudge patterns (blobs about `sigma` pixels across).
    """
    white = mulberry32_array(seed, height * width).reshape(height, width)
    smooth = gaussian_blur(white, sigma)
    low, high = smooth.min(), smooth.max()
    return (smooth - low) / max(high - low, 1e-12)


def mix(base, other, weight):
    """Blend two images: result = base * (1 - weight) + other * weight. `weight` may be a (h, w) or (h, w, 1) map."""
    weight = np.asarray(weight)
    if weight.ndim == 2:
        weight = weight[:, :, None]
    return base * (1.0 - weight) + other * weight
