"""Loading picture pairs and clear scenes from the project's data folders (used by the sweep scripts)."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from common.imageio import load_image, normalise_depth

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
CURATED = DATA / "curated"


def matched_dir():
    """Folder of the matched synthetic pairs for the active preset set (environment variable DIP_PRESETS, default original)."""
    import os

    return DATA / "synthetic" / ("matched_calibrated" if os.environ.get("DIP_PRESETS") == "calibrated" else "matched")


def load_matched_pairs(medium, level, count, max_side, start=0):
    """`count` matched synthetic pairs of one medium and severity, beginning at scene number `start`: a list of (degraded, clear, id).

    Pictures are float RGB. Using start=0 for one experiment and start=6 for another gives two sets of different scenes.
    """
    with open(matched_dir() / "manifest_matched.csv") as handle:
        rows = [r for r in csv.DictReader(handle) if r["medium"] == medium and r["level"] == level][start:start + count]
    return [(load_image(DATA / r["degraded_path"], max_side), load_image(DATA / r["reference_path"], max_side), r["id"]) for r in rows]


def load_clear_scenes(count, max_side, only=None):
    """`count` clear NYU scenes with their depth maps (0 near .. 1 far), spread across the file: a list of (clear, depth_map, id).

    Give `only` (a position 0..count-1) to load just that one scene; the list then holds a single entry.
    """
    with open(CURATED / "manifest_depth.csv") as handle:
        rows = list(csv.DictReader(handle))
    scenes = []
    for position, index in enumerate(np.linspace(0, len(rows) - 1, count).round().astype(int)):
        if only is not None and position != only:
            continue
        row = rows[index]
        clear = load_image(CURATED / row["degraded_path"], max_side)  # in the depth manifest the colour picture is "degraded_path"
        depth_mm = Image.open(CURATED / row["reference_path"])
        depth_mm = depth_mm.resize(clear.shape[1::-1], Image.NEAREST)  # same size as the (possibly shrunk) colour picture
        scenes.append((clear, normalise_depth(np.asarray(depth_mm, dtype=np.float64)), row["id"]))
    return scenes
