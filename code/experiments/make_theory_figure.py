"""Pictures of the best chain next to the theoretical restorations, for the best, a typical and the worst case of each medium (SYNTHETIC), for the report.

For every synthetic held-out picture of run_theory_chains.py the best of all chains tried (earlier searches and theory-derived) is compared with the theoretical ceiling:
the TRUE INVERSE (haze, water, plastic) or the oracle LUT (glass, which has no inverse). The case with the smallest gap, the median gap and the largest gap are drawn:
  degraded | best chain | oracle LUT | theoretical restoration | clear      (PSNR in each caption), with the diagnostics that explain the gap.
Output: results/figures/theory_<medium>.jpg   Run from the project root:  .venv/bin/python code/experiments/make_theory_figure.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS  # noqa: E402
from experiments.analyse_chain_limits import MODELS, oracle_lut, psnr, quantise, transmission, true_inverse  # noqa: E402
from experiments.run_theory_chains import spec  # noqa: E402
from restore.baselines import wiener_deblur  # noqa: E402
from restore.methods import build_sequence  # noqa: E402
from degrade import plastic  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE = 220
META = {"world", "medium", "level", "picture", "oracle_affine", "oracle_lut", "true_inverse", "none"}


def theory_restoration(medium, degraded, clear, depth, params):
    """The theoretical restoration: the true inverse (plastic: affine inverse, then the Wiener filter with the true blur and the best noise setting); glass: the oracle LUT."""
    if medium == "glass":
        return oracle_lut(degraded, clear), "oracle LUT (no inverse exists)"
    inverse = true_inverse(medium, degraded, depth, params)
    if medium == "plastic":
        sigma = {**plastic.DEFAULTS, **params}["blur"]
        inverse = max((wiener_deblur(inverse, sigma, k) for k in (1e-5, 1e-4, 1e-3, 1e-2)), key=lambda p: psnr(p, clear))
    return inverse, "true inverse"


def tile(caption, picture):
    image = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((SIDE, SIDE))
    canvas = Image.new("RGB", (SIDE, SIDE * 3 // 4 + 30), (30, 30, 30))
    canvas.paste(image, (0, 30))
    for k, line in enumerate(caption.split("\n")[:2]):
        ImageDraw.Draw(canvas).text((3, 2 + 12 * k), line, fill=(225, 225, 225))
    return canvas


def main():
    for medium in ("haze", "water", "plastic", "glass"):
        with open(TABLES / f"theory_chains_synthetic_{medium}.csv") as handle:
            rows = list(csv.DictReader(handle))
        chains = [k for k in rows[0] if k not in META]
        records = []
        for r in rows:
            best = max(chains, key=lambda c: float(r[c]))
            index = int(r["picture"].replace("scene", ""))
            clear, depth, _ = load_clear_scenes(24, 256, only=index)[0]
            params = CALIBRATED_PRESETS[medium][r["level"]]
            degraded = quantise(MODELS[medium].apply(clear, depth if medium != "glass" else None, params))
            theory, label = theory_restoration(medium, degraded, clear, depth, params)
            theory_psnr = psnr(theory, clear)
            records.append({"gap": theory_psnr - float(r[best]), "row": r, "best": best, "index": index, "level": r["level"], "theory": theory, "label": label, "theory_psnr": theory_psnr,
                            "degraded": degraded, "clear": clear, "depth": depth, "params": params})
        records.sort(key=lambda x: x["gap"])
        cases = [("smallest gap", records[0]), ("typical gap", records[len(records) // 2]), ("largest gap", records[-1])]
        sheet_rows = []
        for title, rec in cases:
            t = transmission(medium, rec["depth"], rec["params"])
            note = "" if t is None else f"t<0.1 on {100 * np.mean(t[:, :, 0] < 0.1):.0f}% of pixels"
            r = rec["row"]
            lut = oracle_lut(rec["degraded"], rec["clear"])
            best_picture = build_sequence(spec(rec["best"], rec["level"]))(rec["degraded"])
            sheet_rows.append([tile(f"{title}, {rec['level']}\n{note}", rec["degraded"]),
                               tile(f"best chain {psnr(best_picture, rec['clear']):.1f} dB\n{rec['best']}"[:80], best_picture),
                               tile(f"oracle LUT {psnr(lut, rec['clear']):.1f} dB\n", lut),
                               tile(f"{rec['label']} {rec['theory_psnr']:.1f} dB\ngap {rec['gap']:.1f} dB", rec["theory"]),
                               tile("clear\n", rec["clear"])])
            print(f"{medium:8} {title:22} {rec['level']:7} best {rec['best']:34} {float(r[rec['best']]):5.1f} dB  theory {rec['theory_psnr']:5.1f} dB  gap {rec['gap']:5.1f}   {note}")
        sheet = Image.new("RGB", (5 * SIDE, sum(row[0].height for row in sheet_rows)), (30, 30, 30))
        y = 0
        for row in sheet_rows:
            for k, t_ in enumerate(row):
                sheet.paste(t_, (k * SIDE, y))
            y += row[0].height
        sheet.save(FIGURES / f"theory_{medium}.jpg", quality=92)


if __name__ == "__main__":
    main()
