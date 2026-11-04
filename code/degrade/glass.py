"""Fogged, wet or dirty glass in front of the lens. Three parts, applied in this order:

1. veil     a smooth, uneven milky film (fog or dust):  I = J * t + A * (1 - t), with t varying slowly
2. smudges  patches where the glass scatters light: the picture is blurred, dimmed and mixed with the veil colour
3. droplets round blobs that locally blur and darken what is behind them

The veil, smudges and droplets are not tied to scene distance, because they sit on the glass.
All positions come from a seeded generator, so the same settings always give the same glass.
"""
import numpy as np

from common.rng import mulberry32_array
from common.filters import gaussian_blur
from degrade.ops import mix, smooth_noise_field

DEFAULTS = {
    "veil": 0.35,            # strength of the milky film, 0 = clear glass
    "veil_size": 0.05,       # size of the uneven patches, as a fraction of the longer side
    "smudge": 0.30,          # strength of the smudge patches
    "smudge_size": 0.03,     # size of the smudge patches (fraction of the longer side)
    "smudge_blur": 0.010,    # how much a smudge softens the scene (fraction of the longer side)
    "droplets": 25,          # number of droplets
    "droplet_min": 0.010,    # smallest droplet radius (fraction of the longer side)
    "droplet_max": 0.040,    # largest droplet radius
    "droplet_blur": 0.006,   # how much a droplet softens what is behind it
    "droplet_strength": 0.80,
    "airlight": 0.85,        # brightness of the film
    "seed": 7,               # change to get a different pattern of patches and droplets
}


def veil_layer(image, settings):
    """Step 1: the uneven milky film."""
    height, width = image.shape[:2]
    longer = max(height, width)
    field = smooth_noise_field(height, width, settings["veil_size"] * longer, settings["seed"] * 3 + 1)
    t = (1.0 - settings["veil"] * (0.4 + 0.6 * field))[:, :, None]  # 1 = clear, lower = foggier
    return image * t + settings["airlight"] * (1.0 - t)


def smudge_layer(image, settings):
    """Step 2: soft patches that scatter the scene."""
    height, width = image.shape[:2]
    longer = max(height, width)
    field = smooth_noise_field(height, width, settings["smudge_size"] * longer, settings["seed"] * 3 + 2)
    patch = np.clip((field - 0.5) * 2.0, 0.0, 1.0)  # only the upper half of the field becomes a smudge
    scattered = gaussian_blur(image, settings["smudge_blur"] * longer) * 0.8 + settings["airlight"] * 0.2
    return mix(image, scattered, settings["smudge"] * patch)


def droplet_mask(height, width, settings):
    """Alpha map (h, w) of round droplets with soft edges: 1 inside a droplet, 0 outside."""
    count = int(settings["droplets"])
    mask = np.zeros((height, width))
    if count <= 0:
        return mask
    longer = max(height, width)
    draws = mulberry32_array(settings["seed"] * 3 + 3, 3 * count).reshape(count, 3)
    rows, cols = np.mgrid[0:height, 0:width]
    for x_draw, y_draw, size_draw in draws:
        centre_x, centre_y = x_draw * width, y_draw * height
        radius = (settings["droplet_min"] + size_draw * (settings["droplet_max"] - settings["droplet_min"])) * longer
        edge = max(1.0, 0.2 * radius)  # width of the soft rim
        distance = np.hypot(cols - centre_x, rows - centre_y)
        mask = np.maximum(mask, np.clip((radius - distance) / edge, 0.0, 1.0))
    return mask


def droplet_layer(image, settings):
    """Step 3: droplets blur and slightly darken whatever is behind them."""
    height, width = image.shape[:2]
    longer = max(height, width)
    seen_through_drop = gaussian_blur(image, settings["droplet_blur"] * longer) * 0.9 + settings["airlight"] * 0.1
    return mix(image, seen_through_drop, settings["droplet_strength"] * droplet_mask(height, width, settings))


def apply(clean, depth_map=None, params=None):
    """Return `clean` seen through the glass. depth_map is ignored (kept so all models share a signature)."""
    settings = {**DEFAULTS, **(params or {})}
    result = veil_layer(clean, settings)
    result = smudge_layer(result, settings)
    return droplet_layer(result, settings)
