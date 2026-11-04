"""Export a spread of RGB + depth pairs from the NYU Depth v2 file, without loading the whole 3 GB.

The file is HDF5, so single images can be read straight from disk. About 150 scenes are picked evenly
across the file (it is ordered by scene, so this gives many different rooms). For each pick we write:
  <id>_rgb.png          the colour picture
  <id>_depth_mm.png     depth in millimetres, 16-bit (exact)
  <id>_depth_near_white.png   an 8-bit picture, white = near and black = far, ready to drop into the demo page
A picture is skipped if its depth range is too small to be interesting (a flat wall).
Run from the project root:  .venv/bin/python code/data_tools/export_nyu.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.imageio import normalise_depth  # noqa: E402
from data_tools.curation import CURATED, DATA, contact_sheet, write_manifest  # noqa: E402

NYU_FILE = DATA / "depth" / "NYUv2" / "nyu_depth_v2_labeled.mat"
OUT = CURATED / "depth" / "nyu"
HOW_MANY = 150
MIN_DEPTH_RANGE_M = 1.5  # skip pictures whose 1st-to-99th percentile depth range is smaller than this


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, sheet = [], []
    with h5py.File(NYU_FILE, "r") as nyu:
        total = nyu["images"].shape[0]
        # step 1: look at about twice as many pictures as we need, spread over the whole file, and keep the ones with enough depth range
        looked_at = np.linspace(0, total - 1, HOW_MANY * 2).round().astype(int)
        ranges = {}
        for index in dict.fromkeys(looked_at):
            low, high = np.percentile(np.asarray(nyu["depths"][index]), [1, 99])
            ranges[index] = high - low
        eligible = [i for i in ranges if ranges[i] >= MIN_DEPTH_RANGE_M]
        # step 2: from the eligible ones pick HOW_MANY evenly, so the picks still cover the whole file (many different rooms)
        chosen = [eligible[i] for i in dict.fromkeys(np.linspace(0, len(eligible) - 1, min(HOW_MANY, len(eligible))).round().astype(int))]
        for index in chosen:
            # the file stores arrays transposed (MATLAB column order): images are (3, width, height), depth (width, height)
            rgb = np.asarray(nyu["images"][index]).transpose(2, 1, 0)       # -> (height, width, 3), uint8
            depth_m = np.asarray(nyu["depths"][index]).T.astype(np.float64)  # -> (height, width), metres
            name = f"nyu_{index:04d}"
            Image.fromarray(rgb).save(OUT / f"{name}_rgb.png")
            Image.fromarray(np.round(depth_m * 1000).astype(np.uint16)).save(OUT / f"{name}_depth_mm.png")
            near_white = 255 * (1.0 - normalise_depth(depth_m, bigger_is_farther=True))
            Image.fromarray(np.round(near_white).astype(np.uint8)).save(OUT / f"{name}_depth_near_white.png")
            manifest.append({"medium": "depth", "source": "NYU-Depth-v2", "id": name, "level": "", "depth_range_1_99_m": f"{ranges[index]:.2f}",
                             "degraded_path": (OUT / f"{name}_rgb.png").relative_to(CURATED), "reference_path": (OUT / f"{name}_depth_mm.png").relative_to(CURATED),
                             "origin": f"{NYU_FILE.name}:index {index}"})
            sheet.append((Image.fromarray(rgb), f"{name} {ranges[index]:.1f} m"))
    kept = len(manifest)
    write_manifest("depth", manifest)
    contact_sheet("depth_nyu", sheet, columns=10)
    print(f"exported {kept} pictures to {OUT.relative_to(CURATED.parent.parent)}")


if __name__ == "__main__":
    main()
