"""Translucent or aged plastic sheet in front of the lens: dimmer, tinted, softer, with a milky veil.

    I = blur(J) * tint * t + A * (1 - t)
t is one number for the whole picture (the sheet does not depend on scene distance).
"""
import numpy as np

from common.filters import gaussian_blur

DEFAULTS = {
    "transmission": 0.70,  # fraction of light that passes (lower = more opaque)
    "tint_red": 1.00,      # per-colour gain; yellowed plastic loses blue
    "tint_green": 0.95,
    "tint_blue": 0.80,
    "blur": 0.006,         # softening, as a fraction of the longer image side
    "airlight": 0.88,      # brightness of the milky veil
}


def apply(clean, depth_map=None, params=None):
    """Return `clean` as seen through the plastic. depth_map is ignored (kept so all models share a signature)."""
    settings = {**DEFAULTS, **(params or {})}
    sigma_pixels = settings["blur"] * max(clean.shape[0], clean.shape[1])
    tint = np.array([settings["tint_red"], settings["tint_green"], settings["tint_blue"]])[None, None, :]
    t = settings["transmission"]
    return gaussian_blur(clean, sigma_pixels) * tint * t + settings["airlight"] * (1.0 - t)
