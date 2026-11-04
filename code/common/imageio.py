"""Load and save images as float arrays in [0, 1], and turn depth maps into a 0..1 'how far' map."""
import numpy as np
from PIL import Image


def load_image(path, max_side=None):
    """Read an image file as float RGB (height, width, 3) in [0, 1]. Optionally shrink so the longest side is `max_side`."""
    picture = Image.open(path).convert("RGB")
    if max_side is not None and max(picture.size) > max_side:
        scale = max_side / max(picture.size)
        picture = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.LANCZOS)
    return np.asarray(picture, dtype=np.float64) / 255.0


def save_image(path, image):
    """Write a float RGB image in [0, 1] as an 8-bit file (values outside 0..1 are clipped)."""
    Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def load_depth_as_unit_map(path, near_is_bright=True, max_side=None):
    """Read a grey depth picture and return a map in [0, 1] where 0 = nearest and 1 = farthest.

    Depth pictures disagree on whether white means near or far, so `near_is_bright` picks one.
    """
    grey = Image.open(path).convert("F")
    if max_side is not None and max(grey.size) > max_side:
        scale = max_side / max(grey.size)
        grey = grey.resize((round(grey.width * scale), round(grey.height * scale)), Image.BILINEAR)
    values = np.asarray(grey, dtype=np.float64)
    return normalise_depth(values, bigger_is_farther=not near_is_bright)


def normalise_depth(depth, bigger_is_farther=True):
    """Stretch any depth array to [0, 1] using the 1st and 99th percentiles (robust to outliers), 0 = near."""
    low, high = np.percentile(depth, [1, 99])
    unit = np.clip((depth - low) / max(high - low, 1e-12), 0.0, 1.0)
    return unit if bigger_is_farther else 1.0 - unit
