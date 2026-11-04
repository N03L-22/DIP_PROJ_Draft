"""Checks for restore/local.py: the local forms must reduce to the global method in the trivial cases and behave sensibly otherwise.
Run from the project root:  .venv/bin/python code/tests/test_local.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from restore import local  # noqa: E402
from restore.baselines import contrast_stretch  # noqa: E402

checks = []


def check(name, condition):
    checks.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)


rng = np.random.default_rng(1)
picture = rng.random((60, 80, 3)) * 0.5 + 0.2
identity = lambda image: image  # noqa: E731
brighten = lambda image: np.clip(image * 1.5, 0, 1)  # noqa: E731

check("tiled with grid 1 equals the global method", np.allclose(local.tiled(contrast_stretch, picture, 1), contrast_stretch(picture), atol=1e-9))
check("tiled identity returns the picture (blending adds no seams)", np.allclose(local.tiled(identity, picture, 4), picture, atol=1e-9))
check("tiled keeps the picture size and range", local.tiled(brighten, picture, 3).shape == picture.shape and local.tiled(brighten, picture, 3).max() <= 1.0)
check("segmented with one region equals the global method", np.allclose(local.segmented(contrast_stretch, picture, np.zeros((60, 80), int)), contrast_stretch(picture), atol=1e-6))
smooth = np.dstack([np.tile(np.linspace(0.2, 0.9, 80), (60, 1))] * 3)  # regions of a smooth picture are clean blobs (noise would give speckle)
# near a region border the blend mixes in nearest-pixel copies, so the identity is only approximate there
check("segmented identity returns the picture closely", np.abs(local.segmented(identity, smooth, local.quantile_labels(smooth.mean(axis=2), 3)) - smooth).mean() < 0.02)
labels = local.quantile_labels(picture.mean(axis=2), 3)
check("quantile regions are about equal in size", all(abs((labels == k).mean() - 1 / 3) < 0.05 for k in range(3)))
# a brightness ramp: each tile stretches its own part of the ramp, so the result differs from one global stretch
ramp = smooth
check("tiles decide per tile: the result differs from the global stretch", np.abs(local.tiled(contrast_stretch, ramp, 2) - contrast_stretch(ramp)).max() > 0.05)
failed = [name for name, ok in checks if not ok]
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
sys.exit(1 if failed else 0)
