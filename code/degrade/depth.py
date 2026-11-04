"""Simple depth maps for scenes that have no measured depth. Values run from 0 (near) to 1 (far).

No learned depth model is used: the proposal allows measured distances, a planar ramp, or 2-3 hand-set layers.
"""
import numpy as np


def ramp_depth(height, width, far_at_top=True):
    """A planar ramp: the ground is near at the bottom of the picture and the horizon is far at the top."""
    rows = np.linspace(0.0, 1.0, height)[:, None] * np.ones((1, width))
    return 1.0 - rows if far_at_top else rows


def layered_depth(height, width, cuts=(0.4, 0.7), levels=(1.0, 0.55, 0.15)):
    """Two or three flat depth layers split by horizontal lines (cuts are fractions of the height from the top)."""
    depth = np.zeros((height, width))
    edges = [0.0, *cuts, 1.0]
    for band, level in enumerate(levels):
        top, bottom = round(edges[band] * height), round(edges[band + 1] * height)
        depth[top:bottom, :] = level
    return depth
