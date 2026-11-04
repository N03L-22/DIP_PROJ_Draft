"""Atmospheric haze or fog: light is lost with distance and a bright veil is added.

    t = exp(-beta * d)            how much scene light survives at distance d
    I = J * t + A * (1 - t)       A is the colour of the fog (near white)
"""
import numpy as np

DEFAULTS = {
    "beta": 1.2,        # how quickly fog thickens with distance (bigger = denser fog)
    "scene_depth": 2.0,  # distance of the farthest point, in the same units as 1 / beta
    "airlight": 0.92,   # brightness of the fog, 0 to 1 (same for red, green and blue)
}


def transmission(depth_map, params):
    """Fraction of light that gets through, per pixel. depth_map is 0 (near) to 1 (far)."""
    return np.exp(-params["beta"] * params["scene_depth"] * depth_map)[:, :, None]


def apply(clean, depth_map, params=None):
    """Return the hazy version of `clean` (h, w, 3) given the 0..1 depth map (h, w)."""
    settings = {**DEFAULTS, **(params or {})}
    t = transmission(depth_map, settings)
    return clean * t + settings["airlight"] * (1.0 - t)
