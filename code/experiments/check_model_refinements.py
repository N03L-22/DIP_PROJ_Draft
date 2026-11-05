"""STAGE 2h - which parts of the image-formation equation do the REAL pictures need? Fit nested models to each real (clear, degraded) pair.

For every real pair we find, by least squares, the best version of each model that turns the clear picture J into the degraded picture I, and score how close the
fitted picture is to the real degraded one (PSNR, higher = the model explains the real picture better). The models, each adding one idea:
   M1  I = a J + b                          one transmission and one glow for all colours (our haze model with a flat t)
   M2  I_c = a_c J_c + b_c                  a separate transmission and glow per colour channel
   M3  M2 with a blurred scene              forward scattering: J is first blurred (best width chosen per pair)
   M4  M3 with a different affine map in each cell of a 4 x 4 grid    the damage varies across the picture (depth, uneven light): the spatial ceiling
The step from one model to the next is the value of that refinement. Also reported: the mean transmission a_c and glow A_c = b_c / (1 - a_c) per colour from M2.
Output: results/tables/model_refinement.csv  model_refinement_summary.csv
Run from the project root:  .venv/bin/python code/experiments/check_model_refinements.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.filters import gaussian_blur_fft  # noqa: E402
from common.imageio import load_image  # noqa: E402
from experiments.validate_models import read_pairs  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES = PROJECT / "results" / "tables"
CURATED = PROJECT / "data" / "curated"
SIDE = 256
SIGMAS = [0.0, 0.004, 0.008, 0.015, 0.03]  # blur widths, share of the longer side


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def affine_fit(j, i):
    """Least-squares a, b with i = a j + b over all values of the arrays given."""
    a, b = np.polyfit(j.ravel(), i.ravel(), 1)
    return a, b


def fit_models(clear, degraded):
    out = {}
    a, b = affine_fit(clear, degraded)
    out["M1"] = psnr(np.clip(a * clear + b, 0, 1), degraded)
    channels = [affine_fit(clear[:, :, c], degraded[:, :, c]) for c in range(3)]
    fitted = np.dstack([a_c * clear[:, :, c] + b_c for c, (a_c, b_c) in enumerate(channels)])
    out["M2"] = psnr(np.clip(fitted, 0, 1), degraded)
    best, best_sigma, best_blurred = -1.0, 0.0, clear
    for sigma in SIGMAS:
        blurred = clear if sigma == 0 else np.dstack([gaussian_blur_fft(clear[:, :, c], sigma * SIDE) for c in range(3)])
        fit = np.dstack([a_c * blurred[:, :, c] + b_c for c, (a_c, b_c) in enumerate(affine_fit(blurred[:, :, c], degraded[:, :, c]) for c in range(3))])
        score = psnr(np.clip(fit, 0, 1), degraded)
        if score > best:
            best, best_sigma, best_blurred = score, sigma, blurred
    out["M3"], out["sigma"] = best, best_sigma
    fitted = np.empty_like(degraded)
    h, w = degraded.shape[:2]
    for r in range(4):
        for q in range(4):
            ys, xs = slice(r * h // 4, (r + 1) * h // 4), slice(q * w // 4, (q + 1) * w // 4)
            for c in range(3):
                a_c, b_c = affine_fit(best_blurred[ys, xs, c], degraded[ys, xs, c])
                fitted[ys, xs, c] = a_c * best_blurred[ys, xs, c] + b_c
    out["M4"] = psnr(np.clip(fitted, 0, 1), degraded)
    out["t_r"], out["t_g"], out["t_b"] = (channels[0][0], channels[1][0], channels[2][0])
    out["A_r"], out["A_g"], out["A_b"] = [(b_c / (1 - a_c)) if abs(1 - a_c) > 0.05 else float("nan") for a_c, b_c in channels]
    return out


def main():
    sets = {"real haze (185)": [(p["degraded"], p["reference"]) for p in read_pairs("haze")],
            "real raindrops (58)": [(p["degraded"], p["reference"]) for p in read_pairs("glass")]}
    with open(CURATED / "manifest_water_tub.csv") as handle:
        tub = list(csv.DictReader(handle))
    sets["real tank water (120)"] = [(str(CURATED / r["degraded_path"]), str(CURATED / r["reference_path"])) for r in tub[::10]][:120]
    rows, summary = [], []
    for name, pairs in sets.items():
        results = [fit_models(load_image(ref, SIDE), load_image(deg, SIDE)) for deg, ref in pairs]
        rows += [{"set": name, **r} for r in results]
        mean = lambda k: float(np.nanmean([r[k] for r in results]))  # noqa: E731
        summary.append({"set": name, "pairs": len(results), **{k: round(mean(k), 2) for k in ("M1", "M2", "M3", "M4")}, "mean_blur_sigma": round(mean("sigma"), 4),
                        "t_rgb": "/".join(f"{mean(k):.2f}" for k in ("t_r", "t_g", "t_b")), "A_rgb": "/".join(f"{mean(k):.2f}" for k in ("A_r", "A_g", "A_b"))})
        print(summary[-1], flush=True)
    with open(TABLES / "model_refinement.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(TABLES / "model_refinement_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
