"""Before / after pictures for REAL underwater photographs, for the report.

report_ba_tub.jpg    PAIRED real tank photos (TUB): turbid picture, several repairs, and the least-turbid picture of the same view. One row per turbidity band.
report_ba_uieb.jpg   REAL ocean and reef photos (UIEB, no clean reference): original and several repairs. One row per picture.
Run from the project root:  .venv/bin/python code/experiments/make_water_figures.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.imageio import load_image  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
CURATED = PROJECT / "data" / "curated"
FIGURES = PROJECT / "results" / "figures"
SIDE = 260
TUB_METHODS = ["dcp_guided", "clahe", "dcp_clahe", "fusion_dcp", "red_comp", "gamma_auto"]
UIEB_METHODS = ["red_comp", "von_kries", "udcp", "fusion", "clahe", "wb_dcp"]


def tile(caption, picture):
    image = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((SIDE, SIDE))
    framed = Image.new("RGB", (SIDE, SIDE * 3 // 4 + 16), (30, 30, 30))
    framed.paste(image, (0, 16))
    ImageDraw.Draw(framed).text((3, 2), caption, fill=(225, 225, 225))
    return framed


def sheet(rows, path):
    width = max(sum(t.width for t in r) for r in rows)
    out = Image.new("RGB", (width, sum(max(t.height for t in r) for r in rows)), (30, 30, 30))
    y = 0
    for row in rows:
        x = 0
        for t in row:
            out.paste(t, (x, y))
            x += t.width
        y += max(t.height for t in row)
    out.save(path, quality=90)


def main():
    with open(CURATED / "manifest_water_tub.csv") as handle:
        pairs = list(csv.DictReader(handle))
    rows = []
    for level, wanted in (("mild", "Exp10_Cam1_S02"), ("medium", "Exp20_Cam2_S04"), ("heavy", "Exp20_Cam2_S07")):
        pair = next(p for p in pairs if p["id"] == wanted)
        degraded, reference = load_image(CURATED / pair["degraded_path"], SIDE), load_image(CURATED / pair["reference_path"], SIDE)
        row = [tile(f"turbid {pair['ntu']} NTU (REAL)", degraded)]
        row += [tile(name, build_method({"name": name})(degraded)) for name in TUB_METHODS]
        row.append(tile("least turbid (reference)", reference))
        rows.append(row)
    sheet(rows, FIGURES / "report_ba_tub.jpg")

    with open(CURATED / "manifest_water.csv") as handle:
        uieb = list(csv.DictReader(handle))
    rows = []
    for index in np.linspace(0, len(uieb) - 1, 5).round().astype(int):
        picture = load_image(CURATED / uieb[index]["degraded_path"], SIDE)
        row = [tile(f"original ({uieb[index]['level']}, REAL)", picture)]
        row += [tile(name, build_method({"name": name})(picture)) for name in UIEB_METHODS]
        rows.append(row)
    sheet(rows, FIGURES / "report_ba_uieb.jpg")
    print("wrote report_ba_tub.jpg and report_ba_uieb.jpg")


if __name__ == "__main__":
    main()
