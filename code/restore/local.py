"""Run any restoration method LOCALLY instead of on the whole picture.

A method such as a contrast stretch looks at the whole picture to decide how strongly to act. If the damage differs from
place to place (haze thickens with distance; raindrops hit only parts), one global decision is wrong for much of the picture.
Two ways to give each region its own decision, both usable with ANY method that maps a picture to a picture:

  tiled      cut the picture into a grid of overlapping tiles, run the method on each tile alone, and blend the tiles back
             with smooth weights so no seams show. grid = 1 is the same as the global method.
  segmented  group the pixels into a few regions (for example "thin haze", "medium", "thick") and run the method once per
             region, using only that region's pixels to decide, then blend with soft region weights.

The regions can be found from the picture itself (damage_regions: where the dark channel is high, i.e. where the haze is
thick; sharpness_regions: where fine detail is lost, i.e. inside raindrops) or, as an upper bound on synthetic pictures,
from the true depth (depth_regions).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from common.filters import dark_channel, gaussian_blur


def _clean(picture):
    return np.clip(np.nan_to_num(picture, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def tiled(method, image, grid, overlap=0.5):
    """Apply `method` tile by tile (grid x grid tiles, each widened by `overlap` of its size on every side) and blend with triangle weights."""
    height, width = image.shape[:2]
    tile_h, tile_w = height / grid, width / grid
    total = np.zeros(image.shape, dtype=np.float64)
    weight = np.zeros(image.shape[:2], dtype=np.float64)
    for row in range(grid):
        for column in range(grid):
            top, bottom = int(round(max(0, (row - overlap) * tile_h))), int(round(min(height, (row + 1 + overlap) * tile_h)))
            left, right = int(round(max(0, (column - overlap) * tile_w))), int(round(min(width, (column + 1 + overlap) * tile_w)))
            result = _clean(method(image[top:bottom, left:right]))
            # triangle window: 1 at the centre of the tile, falling towards the edge of the widened tile (never exactly 0)
            centre_y, centre_x = (row + 0.5) * tile_h - top, (column + 0.5) * tile_w - left
            half_y, half_x = max(centre_y, bottom - top - centre_y), max(centre_x, right - left - centre_x)
            window_y = np.clip(1.0 - np.abs(np.arange(bottom - top) + 0.5 - centre_y) / half_y, 1e-3, 1.0)
            window_x = np.clip(1.0 - np.abs(np.arange(right - left) + 0.5 - centre_x) / half_x, 1e-3, 1.0)
            window = window_y[:, None] * window_x[None, :]
            total[top:bottom, left:right] += result * window[:, :, None]
            weight[top:bottom, left:right] += window
    return total / weight[:, :, None]


def segmented(method, image, labels, soft=0.02):
    """Apply `method` once per region and blend. `labels` is an integer map (h, w); `soft` is the blending blur as a share of the picture size.

    For each region, the pixels outside it are replaced by the nearest pixel inside it, so the method sees a picture made of that
    region only (its statistics decide the settings) and no sharp artificial edges.
    """
    height, width = labels.shape
    sigma = soft * max(height, width)
    out = np.zeros(image.shape, dtype=np.float64)
    weights = []
    results = []
    for value in np.unique(labels):
        inside = labels == value
        if inside.sum() < 64:  # too small to say anything: skip the region
            continue
        _, nearest = distance_transform_edt(~inside, return_indices=True)
        filled = image[nearest[0], nearest[1]]
        results.append(_clean(method(filled)))
        weights.append(gaussian_blur(inside.astype(np.float64), sigma) + 1e-6)
    total = np.sum(weights, axis=0)
    for result, weight in zip(results, weights):
        out += result * (weight / total)[:, :, None]
    return out


def quantile_labels(values, count, blur=0.02):
    """Split a map into `count` regions of equal size by value (smoothed first so regions are blobs, not speckle)."""
    smooth = gaussian_blur(values, blur * max(values.shape))
    edges = np.quantile(smooth, np.linspace(0, 1, count + 1)[1:-1])
    return np.digitize(smooth, edges)


def damage_regions(image, count=3):
    """Regions found from the picture alone: where the dark channel is low, medium, high (high = thick haze or glow)."""
    return quantile_labels(dark_channel(image, 15), count)


def sharpness_regions(image, count=2):
    """Regions found from the picture alone: where fine detail is strong or weak (weak = blurred, e.g. inside a raindrop)."""
    grey = image.mean(axis=2)
    energy = np.abs(grey - gaussian_blur(grey, 1.5))
    return quantile_labels(gaussian_blur(energy, 4.0), count, blur=0.0)


def depth_regions(depth, count=3):
    """Upper bound, synthetic pictures only: regions from the TRUE depth (near, middle, far)."""
    return quantile_labels(depth, count)
