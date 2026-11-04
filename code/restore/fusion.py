"""STAGE 3e - multi-scale fusion and Tarel's veil removal, two stronger classical methods that use no learning.

Fusion (Mertens et al. 2007, used for dehazing and underwater pictures by Ancuti et al.): make several enhanced copies of one
picture, give every pixel of every copy a weight for how contrast-rich, colourful and well-exposed it is, and blend the copies
with a Laplacian pyramid so the blend is smooth. Each copy is good at something; the weights pick the good parts.
"""
import numpy as np
from scipy.ndimage import median_filter, zoom

from common.filters import gaussian_blur
from common.stats import luminance
from restore import baselines, extra
from restore.dcp import dcp_restore


def shrink(array):
    """One step down the Gaussian pyramid: blur a little, then keep every second pixel (rows and columns)."""
    return gaussian_blur(array, 1.0)[::2, ::2]


def enlarge(array, shape):
    """One step up: stretch a pyramid level to the next level's (height, width) with linear interpolation."""
    factors = [shape[0] / array.shape[0], shape[1] / array.shape[1]] + [1] * (array.ndim - 2)
    return zoom(array, factors, order=1)[: shape[0], : shape[1]]


def gaussian_pyramid(array, levels):
    """The picture and `levels - 1` ever smaller, smoother copies of it."""
    pyramid = [array]
    for _ in range(levels - 1):
        pyramid.append(shrink(pyramid[-1]))
    return pyramid


def laplacian_pyramid(array, levels):
    """The detail bands: each level is the Gaussian level minus an enlarged copy of the next, coarser one. The last level is the coarse remainder."""
    gaussians = gaussian_pyramid(array, levels)
    bands = [gaussians[i] - enlarge(gaussians[i + 1], gaussians[i].shape) for i in range(levels - 1)]
    return bands + [gaussians[-1]]


def mertens_weights(image):
    """Per-pixel quality of one picture: contrast x saturation x well-exposedness, each in 0..1 (Mertens et al.)."""
    grey = luminance(image)
    laplacian = np.abs(4 * grey - np.roll(grey, 1, 0) - np.roll(grey, -1, 0) - np.roll(grey, 1, 1) - np.roll(grey, -1, 1))  # edges and texture
    saturation = image.std(axis=2)                                                                                           # colourfulness
    exposure = np.exp(-((image - 0.5) ** 2) / (2 * 0.2 ** 2)).prod(axis=2)                                                   # close to mid-grey is well exposed
    return laplacian * saturation * exposure + 1e-6


def blend(images, levels=5):
    """Blend several pictures of the same scene with Laplacian-pyramid fusion and return one picture."""
    weights = np.stack([mertens_weights(img) for img in images])
    weights = weights / weights.sum(axis=0)                       # the weights of all copies add up to 1 at every pixel
    fused = None
    for img, weight in zip(images, weights):
        bands = laplacian_pyramid(img, levels)
        weight_levels = gaussian_pyramid(weight, levels)           # smooth weights so the seams do not show
        weighted = [band * w[:, :, None] for band, w in zip(bands, weight_levels)]
        fused = weighted if fused is None else [a + b for a, b in zip(fused, weighted)]
    result = fused[-1]
    for band in reversed(fused[:-1]):                              # collapse the pyramid: enlarge and add the detail back
        result = enlarge(result, band.shape) + band
    return np.clip(result, 0.0, 1.0)


def fusion_enhance(image, levels=5):
    """Fuse three enhanced copies of the picture: white-balanced, white-balanced + CLAHE, and white-balanced + gamma-brightened.

    A general-purpose version of the underwater and haze fusion methods. No physical model is used.
    """
    balanced = baselines.gray_world(image)
    return blend([balanced, baselines.clahe(balanced), baselines.auto_gamma(balanced, 0.5)], levels)


def tarel_veil_removal(image, window=0.03, strength=0.95, gamma=0.7):
    """Tarel and Hautiere's fast visibility restoration: estimate the veil with median filters, subtract it, then lift the tones.

    veil base W = the smallest colour at each pixel (haze lifts every colour, so the minimum is mostly veil);
    local average = median of W; local contrast = median of |W - average|; veil V = average - contrast, scaled by `strength`
    and never above W. Restored = (I - V) / (1 - V), then a gamma curve restores the tones. `window` is a fraction of the longer side.
    """
    size = max(3, int(window * max(image.shape[:2])) | 1)
    minimum = image.min(axis=2)
    average = median_filter(minimum, size=size, mode="nearest")
    contrast = median_filter(np.abs(minimum - average), size=size, mode="nearest")
    veil = np.clip(np.minimum(strength * np.maximum(average - contrast, 0.0), minimum), 0.0, 0.99)
    restored = np.clip((image - veil[:, :, None]) / (1.0 - veil[:, :, None]), 0.0, 1.0)
    return restored ** gamma


def fusion_after_dcp(image, levels=5):
    """Fuse the original, the DCP result, the CLAHE of the DCP result and the unsharp-masked DCP result: keeps DCP's haze removal and recovers detail it flattened."""
    restored = dcp_restore(image, refine=True)["restored"]
    return blend([image, restored, baselines.clahe(restored), extra.unsharp_mask(restored)], levels)
