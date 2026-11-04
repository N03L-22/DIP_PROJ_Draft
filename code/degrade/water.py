"""Turbid water: each colour dies off at its own rate, red fastest, and a blue-green veil is added.

    t_c = exp(-beta_c * d)        separate rate for red, green and blue (beta_red > beta_green > beta_blue)
    I_c = J_c * t_c + A_c * (1 - t_c)
"""
import numpy as np

DEFAULTS = {
    "beta_red": 0.9,
    "beta_green": 0.35,
    "beta_blue": 0.15,
    "scene_depth": 3.0,   # distance of the farthest point
    "veil_red": 0.10,     # colour of the water light (blue-green)
    "veil_green": 0.42,
    "veil_blue": 0.50,
}


def transmission(depth_map, params):
    """Per-pixel, per-colour fraction of light that survives, shape (h, w, 3)."""
    betas = np.array([params["beta_red"], params["beta_green"], params["beta_blue"]])
    return np.exp(-betas[None, None, :] * params["scene_depth"] * depth_map[:, :, None])


def apply(clean, depth_map, params=None):
    """Return the underwater version of `clean`."""
    settings = {**DEFAULTS, **(params or {})}
    t = transmission(depth_map, settings)
    veil = np.array([settings["veil_red"], settings["veil_green"], settings["veil_blue"]])[None, None, :]
    return clean * t + veil * (1.0 - t)
