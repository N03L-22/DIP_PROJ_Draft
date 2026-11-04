"""Simple numbers that describe how degraded a picture is. Used to sort pictures into mild / medium / heavy."""
import numpy as np

from common.filters import dark_channel


def luminance(image):
    """Brightness of a float RGB image (h, w, 3) using the usual 0.299 / 0.587 / 0.114 weights."""
    return image @ np.array([0.299, 0.587, 0.114])


def contrast(image):
    """Spread of brightness (standard deviation). Fog, water and dirty glass all lower it."""
    return float(luminance(image).std())


def dark_channel_mean(image, patch=15):
    """Average dark channel: the smallest colour value in each patch, averaged over the picture.

    Near 0 for a clear outdoor picture, and rises towards the fog brightness as haze thickens.
    """
    return float(dark_channel(image, patch).mean())


def colour_cast(image):
    """log(red / average of green and blue). Negative means a blue-green cast (typical underwater), positive means reddish."""
    means = image.reshape(-1, 3).mean(axis=0)
    return float(np.log((means[0] + 1e-6) / ((means[1] + means[2]) / 2 + 1e-6)))
