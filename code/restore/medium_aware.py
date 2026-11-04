"""STAGE 3c - medium-aware variants of the Dark Channel Prior, each a small change to dcp.py for one medium.

DCP assumes light is lost evenly in all colours and that haze is the same colour everywhere. Water, glass and plastic
break those assumptions in different ways, so each variant changes exactly one assumption:
  udcp          water   red is gone, so the dark channel ignores red
  local airlight glass  the film is uneven, so the airlight changes across the picture
  near-global t  glass  the film does not depend on distance, so t should be almost constant
"""
import numpy as np
from scipy.ndimage import maximum_filter

from common.filters import gaussian_blur_fft, patch_minimum
from restore.baselines import clahe, gray_world
from restore.dcp import dcp_restore, estimate_airlight, recover


def udcp(image, patch=15, omega=0.95, t0=0.1):
    """Underwater DCP (Drews et al., 2013): the dark channel uses only green and blue, because red is absorbed first.

    With red already lost, red would always look 'dark' and wrongly say the water is clear. Dropping it fixes that.
    """
    green_blue = image[:, :, 1:]                              # drop the red channel
    dark = patch_minimum(green_blue.min(axis=2), patch)
    airlight = estimate_airlight(image, dark)
    transmission = 1.0 - omega * patch_minimum((green_blue / airlight[1:]).min(axis=2), patch)
    return recover(image, transmission, airlight, t0)


def local_airlight_map(image, size=0.15, smooth=0.08):
    """Airlight that varies across the picture: the local brightest value, blurred, per colour.

    Behind a milky film the brightest thing in a region is roughly the film's own brightness there.
    `size` and `smooth` are fractions of the longer side.
    """
    longer = max(image.shape[:2])
    window = max(3, int(size * longer) | 1)  # an odd window size in pixels
    local_max = maximum_filter(image, size=(window, window, 1), mode="nearest")
    return gaussian_blur_fft(local_max, smooth * longer)


def dcp_local_airlight(image, patch=15, omega=0.95, t0=0.1, size=0.15, smooth=0.08):
    """DCP with the airlight allowed to change from place to place (for an uneven veil on glass).

    size and smooth are the window and blur of local_airlight_map, as fractions of the longer side.
    """
    airlight_map = np.maximum(local_airlight_map(image, size, smooth), 0.05)
    # divide by the local airlight, find t, and recover with the same local airlight
    dark = patch_minimum((image / airlight_map).min(axis=2), patch)
    transmission = 1.0 - omega * dark
    return np.clip((image - airlight_map) / np.maximum(transmission, t0)[:, :, None] + airlight_map, 0.0, 1.0)


def dcp_near_global(image, patch_fraction=0.25, omega=0.95, t0=0.1):
    """DCP with a very large patch, so the transmission is almost one number for the whole picture.

    A film on glass is the same distance from the camera everywhere, so a smooth, near-constant t suits it better than
    the patch-by-patch t that is right for haze at varying distance. The patch is a fraction of the longer side.
    """
    patch = max(15, int(patch_fraction * max(image.shape[:2])) | 1)
    return dcp_restore(image, patch=patch, omega=omega, t0=t0, refine=False)["restored"]


def wb_then_dcp(image, patch=15, omega=0.95, t0=0.1):
    """White-balance first (undo the colour cast of water or tinted plastic), then plain DCP."""
    return dcp_restore(gray_world(image), patch=patch, omega=omega, t0=t0, refine=True)["restored"]


def dcp_then_clahe(image, patch=15, omega=0.95, t0=0.1):
    """DCP removes the haze, then CLAHE restores local contrast that the division by t may have flattened (the proposal's 'postprocess' step)."""
    return clahe(dcp_restore(image, patch=patch, omega=omega, t0=t0, refine=True)["restored"])
