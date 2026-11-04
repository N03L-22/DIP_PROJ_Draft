"""Milky or inky water: turbidity that SCATTERS (or absorbs) all colours alike, as in a tank with milk added, or after a squid releases ink.

    t = exp(-beta * d)           the same rate for red, green and blue (unlike clear-sea water, where red dies first)
    I_c = J_c * t + A_c * (1 - t)   A_c is the colour of the cloud: light brown-grey for milk, near black for ink

Milk scatters light without absorbing it, so the cloud is bright and the picture fades towards a light colour (like fog).
Ink absorbs, so the cloud is dark and the picture fades towards black. The equation is the same; only A changes.
Optionally the scene light is also blurred (forward scattering), which real milky water showed (sharpness fell faster than contrast).
"""
import numpy as np

from common.filters import gaussian_blur_fft

DEFAULTS = {
    "beta": 1.0,
    "scene_depth": 2.0,                 # distance of the farthest point, as in the haze model
    "airlight": [0.55, 0.52, 0.42],     # colour of the cloud (replaced by the colour measured on the tank photos)
    "blur": 0.0,                        # forward scattering: the direct light is also blurred (sigma as a share of the longer side)
}


def transmission(depth_map, params):
    return np.exp(-params["beta"] * params["scene_depth"] * depth_map)[:, :, None]


def apply(clean, depth_map, params=None):
    settings = {**DEFAULTS, **(params or {})}
    t = transmission(depth_map, settings)
    direct = clean
    if settings["blur"] > 0:  # small-angle scattering smears the scene light on its way through the cloud
        sigma = settings["blur"] * max(clean.shape[:2])
        direct = np.dstack([gaussian_blur_fft(clean[:, :, c], sigma) for c in range(3)])
    return direct * t + np.array(settings["airlight"])[None, None, :] * (1.0 - t)
