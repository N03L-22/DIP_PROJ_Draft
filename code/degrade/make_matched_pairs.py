"""STAGE 2 - make matched synthetic pairs: a clear picture and its degraded copy for every medium and severity.

Because the clear picture is the reference, these pairs have an exact ground truth, so restoration scores against
them are exact. The scenes are NYU indoor pictures, which come with measured depth (no learned depth model is used),
so haze and water thicken with real distance.

Output:  data/synthetic/matched/<medium>/<level>/<scene>_<medium>_<level>.png                  (original presets)
         data/synthetic/matched_calibrated/...  (settings fitted to real pictures; haze and glass differ)
         <folder>/manifest_matched.csv   (the degraded file, its clear reference, and the settings used)
Run from the project root:  .venv/bin/python code/degrade/make_matched_pairs.py --presets calibrated   (or original)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.imageio import load_image, normalise_depth, save_image  # noqa: E402
from degrade.presets import LEVELS, PRESET_SETS  # noqa: E402
from degrade.stack import MODELS  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
CURATED = PROJECT / "data" / "curated"
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--presets", choices=sorted(PRESET_SETS), default="calibrated")
PRESETS = PRESET_SETS[parser.parse_args().presets]
OUT = PROJECT / "data" / "synthetic" / ("matched" if parser.parse_args().presets == "original" else "matched_" + parser.parse_args().presets)
SCENES = 24  # how many NYU pictures to use (24 scenes x 4 media x 3 levels = 288 pairs)

# --- 1. choose scenes: evenly spaced through the curated NYU list, so the rooms differ
with open(CURATED / "manifest_depth.csv") as handle:
    nyu_rows = list(csv.DictReader(handle))
chosen = [nyu_rows[i] for i in np.linspace(0, len(nyu_rows) - 1, SCENES).round().astype(int)]

manifest = []
for scene in chosen:
    clear_path = CURATED / scene["degraded_path"]  # in the depth manifest, "degraded_path" is the colour picture
    clear = load_image(clear_path)

    # --- 2. depth: the NYU depth file is in millimetres; stretch it to 0 (near) .. 1 (far)
    depth_mm = np.asarray(Image.open(CURATED / scene["reference_path"]), dtype=np.float64)
    depth_map = normalise_depth(depth_mm, bigger_is_farther=True)

    # --- 3. degrade with every medium at every severity and save the result
    for medium, model in MODELS.items():
        for level in LEVELS:
            settings = PRESETS[medium][level]
            degraded = np.clip(model.apply(clear, depth_map, settings), 0.0, 1.0)
            out_file = OUT / medium / level / f"{scene['id']}_{medium}_{level}.png"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            save_image(out_file, degraded)
            manifest.append({"medium": medium, "level": level, "id": scene["id"],
                             "degraded_path": out_file.relative_to(PROJECT / "data"),
                             "reference_path": clear_path.relative_to(PROJECT / "data"),
                             "settings": json.dumps(settings)})

with open(OUT / "manifest_matched.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
    writer.writeheader()
    writer.writerows(manifest)
print(f"wrote {len(manifest)} matched pairs to {OUT.relative_to(PROJECT)}")
