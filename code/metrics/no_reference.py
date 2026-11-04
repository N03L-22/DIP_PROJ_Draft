"""Scores that need no clear reference, written by hand (they are short formulas and part of the analysis).

Needed for water, where no true reference exists, and for judging real photographs. They describe how clear or
colourful a picture looks, not how close it is to the truth, so compare them between methods on the same picture,
never between different pictures. Note: global contrast (the standard deviation of brightness) can RISE under haze that
thickens with distance, because the haze adds a brightness gradient across the picture; entropy, UCIQE and the dark channel
are the more reliable haze indicators.
"""
import numpy as np
from skimage.color import rgb2lab

from common.stats import colour_cast, contrast, dark_channel_mean, luminance


def entropy(image):
    """Shannon entropy of the brightness histogram, in bits. More detail and a wider spread of brightness give a higher value."""
    counts, _ = np.histogram(np.round(luminance(image) * 255), bins=256, range=(0, 255))
    probability = counts[counts > 0] / counts.sum()
    return float(-(probability * np.log2(probability)).sum())


def uciqe(image):
    """UCIQE (Yang and Sowmya, 2015), an underwater-colour-image quality score:
        0.4680 * (spread of chroma) + 0.2745 * (lightness contrast) + 0.2576 * (mean saturation)
    computed in CIELab with lightness and chroma scaled to about 0..1. Higher means clearer and more colourful.
    """
    lab = rgb2lab(image)
    lightness = lab[:, :, 0] / 100.0
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2]) / 128.0
    saturation = chroma / np.sqrt(chroma ** 2 + lightness ** 2 + 1e-12)
    sorted_lightness = np.sort(lightness.ravel())
    one_percent = max(1, sorted_lightness.size // 100)
    lightness_contrast = sorted_lightness[-one_percent:].mean() - sorted_lightness[:one_percent].mean()  # top 1% minus bottom 1%
    return float(0.4680 * chroma.std() + 0.2745 * lightness_contrast + 0.2576 * saturation.mean())


def no_reference_scores(image):
    """Return the five reference-free scores for one picture."""
    return {
        "entropy": entropy(image),
        "contrast": contrast(image),
        "uciqe": uciqe(image),
        "dark_channel": dark_channel_mean(image),  # lower = less haze left
        "colour_cast": colour_cast(image),         # near 0 = balanced colours
    }
