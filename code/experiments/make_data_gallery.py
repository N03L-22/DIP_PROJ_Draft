"""One picture of what each data source looks like (spoiled picture | clear picture), for the report. UIEB photos are left out (their licence forbids redistribution;
they appear in the ocean figure). Output: results/figures/data_gallery.jpg
Run from the project root:  .venv/bin/python code/experiments/make_data_gallery.py
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("DIP_PRESETS", "calibrated")
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.imageio import load_image  # noqa: E402
from degrade import glass, haze, plastic, water  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, PRESETS  # noqa: E402
from experiments.validate_models import read_pairs  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
CURATED = PROJECT / "data" / "curated"
SIDE = 230


def tile(caption, picture):
    image = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((SIDE, SIDE))
    canvas = Image.new("RGB", (SIDE, SIDE * 3 // 4 + 16), (30, 30, 30))
    canvas.paste(image, (0, 16))
    ImageDraw.Draw(canvas).text((3, 2), caption, fill=(225, 225, 225))
    return canvas


def main():
    haze_pairs = read_pairs("haze")
    pick = {s: next(p for p in haze_pairs if p["id"].startswith(s)) for s in ("O-HAZE", "I-HAZE", "Dense-Haze", "NH-Haze")}
    pair = lambda label, p: [tile(f"{label} hazy (REAL)", load_image(p["degraded"], SIDE)), tile("clear", load_image(p["reference"], SIDE))]  # noqa: E731
    rain = read_pairs("glass")[3]
    with open(CURATED / "manifest_water_tub.csv") as handle:
        tub = next(r for r in csv.DictReader(handle) if r["id"] == "Exp20_Cam2_S04")
    reside = PROJECT / "data" / "existing" / "RESIDE-SOTS" / "SOTS" / "outdoor"
    hazy_file = sorted((reside / "hazy").glob("*.jpg"))[5]
    gt_file = reside / "gt" / (hazy_file.stem.split("_")[0] + ".png")
    rows = [pair("O-HAZE", pick["O-HAZE"]) + pair("I-HAZE", pick["I-HAZE"]),
            pair("Dense-Haze", pick["Dense-Haze"]) + pair("NH-Haze", pick["NH-Haze"]),
            [tile("raindrops (REAL)", load_image(rain["degraded"], SIDE)), tile("clean pane", load_image(rain["reference"], SIDE)),
             tile("tank water, 8 NTU (REAL)", load_image(CURATED / tub["degraded_path"], SIDE)), tile("least turbid view", load_image(CURATED / tub["reference_path"], SIDE))],
            [tile("RESIDE outdoor haze (SYNTH.)", load_image(hazy_file, SIDE)), tile("clear", load_image(gt_file, SIDE))]]
    clear, depth, _ = load_clear_scenes(12, SIDE, only=3)[0]
    rows.append([tile("NYU indoor scene (clear)", clear),
                 tile("+ haze (SYNTHETIC)", haze.apply(clear, depth, CALIBRATED_PRESETS["haze"]["medium"])),
                 tile("+ water (SYNTHETIC)", water.apply(clear, depth, PRESETS["water"]["medium"])),
                 tile("+ raindrops (SYNTHETIC)", glass.apply(clear, None, CALIBRATED_PRESETS["glass"]["medium"])),
                 tile("+ plastic (SYNTHETIC)", plastic.apply(clear, depth, PRESETS["plastic"]["medium"]))])
    width = max(len(r) for r in rows) * SIDE
    sheet = Image.new("RGB", (width, sum(r[0].height for r in rows)), (30, 30, 30))
    y = 0
    for row in rows:
        for k, t in enumerate(row):
            sheet.paste(t, (k * SIDE, y))
        y += row[0].height
    sheet.save(PROJECT / "results" / "figures" / "data_gallery.jpg", quality=90)
    print("wrote data_gallery.jpg")


if __name__ == "__main__":
    main()
