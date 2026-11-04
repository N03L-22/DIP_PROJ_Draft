"""Stage-by-stage pictures of the restoration pipelines, for the report: what each step of a pipeline does to one picture, with the PSNR after each step
(PSNR is shown wherever the clean reference is known; the real ocean photos have none).

pipeline_haze.jpg        REAL photo through the DCP pipeline: dark channel, transmission t, DCP, + CLAHE, fusion, local-glow subtraction
pipeline_haze_chain.jpg  the long chain  hist_eq > clahe > dcp_guided  on SYNTHETIC haze and REAL haze, and DCP > CLAHE on real haze
pipeline_water.jpg       the chain  red_comp > von_kries > unsharp  on SYNTHETIC open-sea water, SYNTHETIC milky water, REAL tank water and a REAL ocean photo
pipeline_plastic.jpg     richardson_lucy > unsharp > dcp_guided on SYNTHETIC plastic, next to the plain stretch
pipeline_glass.jpg       REAL raindrops: the glass-specific methods, and a chain that gets worse step by step
pipeline_stack.jpg       a SYNTHETIC water + glass + plastic stack through the searched chain  wiener > stretch > von_kries
Run from the project root:  .venv/bin/python code/experiments/make_pipeline_figures.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.imageio import load_image  # noqa: E402
from degrade import glass, milky, plastic, water  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, PRESETS  # noqa: E402
from experiments.validate_models import read_pairs  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.dcp import dcp_restore  # noqa: E402
from restore.frequency import blind_veil_inversion  # noqa: E402
from restore.methods import build_method  # noqa: E402
from degrade import haze as haze_model  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
FIGURES = PROJECT / "results" / "figures"
CURATED = PROJECT / "data" / "curated"
SIDE = 260


def gray(map2d):
    return np.dstack([np.clip(map2d, 0, 1)] * 3)


def label(caption, picture, reference):
    """Caption with the PSNR against the clean reference when there is one."""
    if reference is None:
        return caption, picture
    return f"{caption}  {full_reference_scores(np.clip(picture, 0, 1), reference)['psnr']:.1f} dB", picture


def run_method(name, picture):
    return np.clip(build_method({"name": name})(picture), 0, 1)


def chain(names, picture, reference=None, input_caption="input"):
    """[(caption, picture)] for the input and the picture after each step of the chain (cumulative)."""
    shown, current = [label(input_caption, picture, reference)], picture
    for name in names:
        current = run_method(name, current)
        shown.append(label("+ " + name if shown[1:] else name, current, reference))
    return shown


def frame(caption, picture):
    image = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((SIDE, SIDE))
    canvas = Image.new("RGB", (SIDE, SIDE * 3 // 4 + 16), (30, 30, 30))
    canvas.paste(image, (0, 16))
    ImageDraw.Draw(canvas).text((3, 2), caption, fill=(225, 225, 225))
    return canvas


def sheet(rows, path):
    framed = [[frame(c, p) for c, p in row] for row in rows]
    out = Image.new("RGB", (max(len(r) for r in framed) * SIDE, sum(r[0].height for r in framed)), (30, 30, 30))
    y = 0
    for row in framed:
        for k, canvas in enumerate(row):
            out.paste(canvas, (k * SIDE, y))
        y += row[0].height
    out.save(path, quality=92)


def main():
    scene, depth, _ = load_clear_scenes(12, SIDE, only=3)[0]
    # ---- haze: the DCP pipeline step by step on a real photo
    pair = read_pairs("haze")[4]
    hazy, clear = load_image(pair["degraded"], SIDE), load_image(pair["reference"], SIDE)
    steps = dcp_restore(hazy, refine=True)
    dcp = steps["restored"]
    sheet([[("hazy (REAL)", hazy), ("1 dark channel", gray(steps["dark_channel"])), ("2 transmission t (refined)", gray(steps["transmission"])), label("3 DCP result", dcp, clear)],
           [label("4 DCP + CLAHE", run_method("clahe", dcp), clear), label("fusion_dcp", run_method("fusion_dcp", hazy), clear),
            label("local-glow subtraction", np.clip(blind_veil_inversion(hazy, local_sigma=0.1), 0, 1), clear), ("clear (reference)", clear)]], FIGURES / "pipeline_haze.jpg")

    # ---- haze: the long chain, synthetic and real
    syn_haze = np.clip(haze_model.apply(scene, depth, CALIBRATED_PRESETS["haze"]["medium"]), 0, 1)
    real_pairs = read_pairs("haze")
    real2 = real_pairs[len(real_pairs) // 2]
    real2_hazy, real2_clear = load_image(real2["degraded"], SIDE), load_image(real2["reference"], SIDE)
    long_chain = ["hist_eq", "clahe", "dcp_guided"]
    sheet([chain(long_chain, syn_haze, scene, "haze (SYNTHETIC)") + [("clear", scene)],
           chain(long_chain, hazy, clear, "haze (REAL)") + [("clear", clear)],
           chain(["dcp_guided", "clahe"], real2_hazy, real2_clear, "haze (REAL)") + [("clear", real2_clear)]], FIGURES / "pipeline_haze_chain.jpg")

    # ---- water: four kinds of picture through the same chain
    names = ["red_comp", "von_kries", "unsharp"]
    open_sea = np.clip(water.apply(scene, depth, PRESETS["water"]["medium"]), 0, 1)
    milk = np.clip(milky.apply(scene, depth, {"beta": 1.7, "blur": 0.004, "airlight": [0.553, 0.519, 0.407]}), 0, 1)
    with open(CURATED / "manifest_water_tub.csv") as handle:
        tub = next(r for r in csv.DictReader(handle) if r["id"] == "Exp20_Cam2_S04")
    tub_turbid, tub_ref = load_image(CURATED / tub["degraded_path"], SIDE), load_image(CURATED / tub["reference_path"], SIDE)
    with open(CURATED / "manifest_water.csv") as handle:
        uieb = list(csv.DictReader(handle))
    ocean = load_image(CURATED / uieb[len(uieb) // 2]["degraded_path"], SIDE)
    sheet([chain(names, open_sea, scene, "open-sea water (SYNTHETIC)") + [("clear", scene)],
           chain(names, milk, scene, "milky water (SYNTHETIC)") + [("clear", scene)],
           chain(names, tub_turbid, tub_ref, "tank water (REAL)") + [("least turbid", tub_ref)],
           chain(names, ocean, None, "ocean (REAL, no reference)")], FIGURES / "pipeline_water.jpg")

    # ---- plastic
    plastic_img = np.clip(plastic.apply(scene, depth, {"transmission": 0.70, "blur": 0.006}), 0, 1)
    sheet([chain(["richardson_lucy", "unsharp", "dcp_guided"], plastic_img, scene, "plastic (SYNTHETIC)") + [label("plain stretch", run_method("stretch", plastic_img), scene), ("clear", scene)]],
          FIGURES / "pipeline_plastic.jpg")

    # ---- glass: real raindrops
    drop = read_pairs("glass")[3]
    wet, dry = load_image(drop["degraded"], SIDE), load_image(drop["reference"], SIDE)
    sheet([[label("raindrops (REAL)", wet, dry)] + [label(n, run_method(n, wet), dry) for n in ("dcp_near_global", "dcp_local_airlight", "flat_field", "local_stats")] + [("clean pane", dry)],
           chain(["flat_field", "unsharp", "stretch"], wet, dry, "raindrops (REAL)") + [("clean pane", dry)]], FIGURES / "pipeline_glass.jpg")

    # ---- stack: water behind glass behind plastic, searched chain
    stacked = water.apply(scene, depth, PRESETS["water"]["medium"])
    stacked = glass.apply(stacked, None, CALIBRATED_PRESETS["glass"]["medium"])
    stacked = np.clip(plastic.apply(stacked, depth, PRESETS["plastic"]["medium"]), 0, 1)
    sheet([chain(["wiener", "stretch", "von_kries"], stacked, scene, "water+glass+plastic (SYNTH.)") + [("clear", scene)]], FIGURES / "pipeline_stack.jpg")
    print("wrote pipeline_haze, _haze_chain, _water, _plastic, _glass, _stack")


if __name__ == "__main__":
    main()
