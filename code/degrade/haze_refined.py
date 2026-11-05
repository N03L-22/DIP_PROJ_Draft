"""Haze with the refinements the real pictures asked for (check_model_refinements.py, F66):

    t_c = exp(-beta * ratio_c * scene_depth * d)         per-colour transmission: blue is attenuated most
    I_c = (G * J)_c * t_c + A_c * (1 - t_c)             per-colour (bluish) glow, and the scene blurred a little by forward scattering G

The fitted values on the 185 real pairs: t(R, G, B) = 0.34, 0.30, 0.26 -> beta ratios 0.89 : 0.99 : 1.12; glow A(R, G, B) = 0.61, 0.63, 0.69; blur width 1.3% of the longer side.
The depth still comes from the scene (depth_map 0 near .. 1 far), which is what the 4 x 4 grid of local maps in the fit asked for.
"""
import numpy as np

from common.filters import gaussian_blur_fft

DEFAULTS = {
    "beta": 2.0,
    "scene_depth": 2.0,
    "ratio": [0.89, 0.99, 1.12],
    "airlight": [0.61, 0.63, 0.69],
    "blur": 0.013,
}


def apply(clean, depth_map, params=None):
    settings = {**DEFAULTS, **(params or {})}
    ratio = np.array(settings["ratio"])[None, None, :]
    t = np.exp(-settings["beta"] * ratio * settings["scene_depth"] * depth_map[:, :, None])
    direct = clean
    if settings["blur"] > 0:
        sigma = settings["blur"] * max(clean.shape[:2])
        direct = np.dstack([gaussian_blur_fft(clean[:, :, c], sigma) for c in range(3)])
    return direct * t + np.array(settings["airlight"])[None, None, :] * (1.0 - t)
