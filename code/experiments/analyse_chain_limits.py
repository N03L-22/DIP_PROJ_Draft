"""STAGE 4g - how long a chain can possibly help? A mathematical check before searching chains of 4 and 5 steps.

Theory (SYNTHETIC pictures, where the damage is known exactly):
  * Haze, water, glass film and plastic each act on a picture as  I = (blur) then (per-colour affine map: gain t, offset A(1 - t)).
    Affine maps compose into ONE affine map, so one step (a stretch or white balance) undoes any number of such layers; a blur adds one more step.
    So the damage is invertible with 1 to 2 steps. Where t depends on position (haze, water: t = exp(-beta d)) the map differs per pixel.
  * Any chain of GLOBAL, per-colour pointwise steps (stretch, gamma, histogram equalisation, white balance, log ...) is one global per-colour function
    of the pixel value. The best such function (the ORACLE LUT: for each input value, the mean clear value of those pixels) is therefore a ceiling
    for every chain of pointwise global steps of ANY length. Longer chains of such steps cannot beat it.
  * Inverting with the TRUE parameters is exact except for 8-bit rounding, which is multiplied by 1/t. Predicted PSNR of that inverse:
        PSNR = -10 log10( mean(1/t^2) / (12 * 255^2) )     (rounding error of 1/sqrt(12) grey levels, scaled by 1/t)
    This is the ceiling for ANY chain on invertible damage, however long.
We measure, per medium on held-out pictures: nothing, best chain of 1, 2, 3 steps (chosen on the search pictures), oracle affine, oracle LUT,
true inverse and the predicted inverse, and the gain of each extra step with its paired standard error. The gains shrink or the ceilings are reached:
that decides whether chains of 4 and 5 can be worth searching.
Output:  results/tables/chain_limits.csv  chain_limits_summary.csv   results/figures/chain_limits.png
Run from the project root:  .venv/bin/python code/experiments/analyse_chain_limits.py
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from degrade import glass, haze, plastic, water  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, LEVELS  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.baselines import wiener_deblur  # noqa: E402
from restore.methods import build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE, FIRST_HELD_OUT, HELD_OUT = 256, 6, 6  # the verify pictures of Stage 4d: scenes 6 to 11 of 24
MODELS = {"haze": haze, "water": water, "glass": glass, "plastic": plastic}


def psnr(picture, clear):
    return full_reference_scores(np.clip(picture, 0, 1), clear)["psnr"]


def quantise(picture):
    """The matched pictures are saved as 8-bit files, so this rounding is part of what every method sees."""
    return np.round(np.clip(picture, 0, 1) * 255.0) / 255.0


def transmission(medium, depth, params):
    """The per-pixel, per-colour t of the true model (None for media without one)."""
    if medium == "haze":
        settings = {**haze.DEFAULTS, **params}
        return np.broadcast_to(haze.transmission(depth, settings), depth.shape + (3,))
    if medium == "water":
        return water.transmission(depth, {**water.DEFAULTS, **params})
    if medium == "plastic":
        settings = {**plastic.DEFAULTS, **params}
        return np.broadcast_to(np.array([settings["tint_red"], settings["tint_green"], settings["tint_blue"]]) * settings["transmission"], depth.shape + (3,))
    return None


def true_inverse(medium, degraded, depth, params):
    """Invert the model with the TRUE parameters on the 8-bit picture (None where the damage is not invertible: raindrops)."""
    t = transmission(medium, depth, params)
    if medium == "haze":
        return (degraded - haze.DEFAULTS["airlight"] * 0 - {**haze.DEFAULTS, **params}["airlight"]) / t + {**haze.DEFAULTS, **params}["airlight"]
    if medium == "water":
        settings = {**water.DEFAULTS, **params}
        veil = np.array([settings["veil_red"], settings["veil_green"], settings["veil_blue"]])[None, None, :]
        return (degraded - veil) / t + veil
    if medium == "plastic":
        settings = {**plastic.DEFAULTS, **params}
        flat = np.clip((degraded - settings["airlight"] * (1 - settings["transmission"])) / t, 0, 1)  # undo the affine part, then the blur
        return flat  # the blur is added by the caller (it needs a noise setting chosen with the clear picture: an oracle)
    return None


def oracle_lut(degraded, clear, bins=256):
    """Best global per-colour function of the pixel value: for each input bin, the mean clear value of the pixels in it."""
    out = np.empty_like(degraded)
    for c in range(3):
        index = np.clip((degraded[:, :, c] * (bins - 1)).round().astype(int), 0, bins - 1)
        sums, counts = np.bincount(index.ravel(), clear[:, :, c].ravel(), bins), np.bincount(index.ravel(), minlength=bins)
        table = np.where(counts > 0, sums / np.maximum(counts, 1), np.arange(bins) / (bins - 1))
        out[:, :, c] = table[index]
    return out


def oracle_affine(degraded, clear):
    out = np.empty_like(degraded)
    for c in range(3):
        a, b = np.polyfit(degraded[:, :, c].ravel(), clear[:, :, c].ravel(), 1)
        out[:, :, c] = a * degraded[:, :, c] + b
    return out


def best_chains():
    """The best chain of each length 1 to 3 for each medium, chosen on the SEARCH pictures of Stage 4d (never on the pictures judged here)."""
    chosen = {}
    for medium in MODELS:
        by = {}
        with open(TABLES / f"pipelines_{medium}_search.csv") as handle:
            for r in csv.DictReader(handle):
                by.setdefault((int(r["length"]), r["pipeline"]), []).append(float(r["psnr"]))
        chosen[medium] = {n: max((k for k in by if k[0] == n), key=lambda k: np.mean(by[k]))[1] for n in (1, 2, 3)}
    return chosen


def main():
    chains = best_chains()
    rows = []
    for medium, model in MODELS.items():
        runners = {n: build_sequence(chains[medium][n].split(" > ")) for n in (1, 2, 3)}
        for index in range(FIRST_HELD_OUT, FIRST_HELD_OUT + HELD_OUT):
            clear, depth, scene = load_clear_scenes(24, SIDE, only=index)[0]
            for level in LEVELS:
                params = CALIBRATED_PRESETS[medium][level]
                degraded = quantise(model.apply(clear, depth if medium != "glass" else None, params))
                row = {"medium": medium, "level": level, "scene": scene, "none": psnr(degraded, clear)}
                for n, run in runners.items():
                    row[f"chain{n}"] = psnr(run(degraded), clear)
                row["oracle_affine"], row["oracle_lut"] = psnr(oracle_affine(degraded, clear), clear), psnr(oracle_lut(degraded, clear), clear)
                inverse = true_inverse(medium, degraded, depth, params)
                if inverse is None:
                    row["true_inverse"] = row["predicted_inverse"] = float("nan")
                else:
                    if medium == "plastic":
                        # the blur is undone with the true blur size; the noise setting is the best of a small grid (an oracle choice)
                        sigma = {**plastic.DEFAULTS, **params}["blur"]
                        inverse = max((wiener_deblur(inverse, sigma, k) for k in (1e-5, 1e-4, 1e-3, 1e-2)), key=lambda p: psnr(p, clear))
                    row["true_inverse"] = psnr(inverse, clear)
                    t = transmission(medium, depth, params)
                    row["predicted_inverse"] = float(-10 * np.log10(np.mean(1.0 / t ** 2) / (12 * 255.0 ** 2)))
                rows.append(row)
        print(f"{medium}: chains {chains[medium]}", flush=True)
    with open(TABLES / "chain_limits.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    print("\nHeld-out PSNR (dB), mean of 18 pictures; gain of each extra step with its paired standard error")
    print(f"{'medium':8}{'none':>7}{'1 step':>8}{'2':>7}{'3':>7}{'d2':>14}{'d3':>14}{'oracleLUT':>11}{'oracleAff':>10}{'inverse':>9}{'predicted':>10}")
    for medium in MODELS:
        own = [r for r in rows if r["medium"] == medium]
        mean = lambda key: float(np.nanmean([r[key] for r in own]))  # noqa: E731
        d2 = np.array([r["chain2"] - r["chain1"] for r in own])
        d3 = np.array([r["chain3"] - r["chain2"] for r in own])
        se = lambda d: float(d.std(ddof=1) / np.sqrt(len(d)))  # noqa: E731
        summary.append({"medium": medium, "chains": " | ".join(chains[medium][n] for n in (1, 2, 3)), "none": mean("none"), "chain1": mean("chain1"), "chain2": mean("chain2"),
                        "chain3": mean("chain3"), "gain_2_over_1": float(d2.mean()), "se_2_over_1": se(d2), "gain_3_over_2": float(d3.mean()), "se_3_over_2": se(d3),
                        "oracle_lut": mean("oracle_lut"), "oracle_affine": mean("oracle_affine"), "true_inverse": mean("true_inverse"), "predicted_inverse": mean("predicted_inverse")})
        print(f"{medium:8}{mean('none'):7.2f}{mean('chain1'):8.2f}{mean('chain2'):7.2f}{mean('chain3'):7.2f}{d2.mean():+8.2f}+-{se(d2):.2f}{d3.mean():+8.2f}+-{se(d3):.2f}"
              f"{mean('oracle_lut'):11.2f}{mean('oracle_affine'):10.2f}{mean('true_inverse'):9.2f}{mean('predicted_inverse'):10.2f}")
    with open(TABLES / "chain_limits_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    figure, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for axis, s in zip(axes, summary):
        labels = ["none", "1 step", "2 steps", "3 steps", "oracle\nLUT", "true\ninverse"]
        values = [s["none"], s["chain1"], s["chain2"], s["chain3"], s["oracle_lut"], s["true_inverse"]]
        axis.bar(range(6), [0 if v != v else v for v in values], color=["#999999", "#3d6299", "#3d6299", "#3d6299", "#d98c2f", "#4c9a52"])
        axis.set_xticks(range(6), labels, fontsize=8)
        axis.set_title(s["medium"])
        axis.set_ylabel("held-out PSNR (dB)")
        axis.set_ylim(min(v for v in values if v == v) - 2, min(max(v for v in values if v == v), 60) + 2)
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle("Chains of 1 to 3 steps against the mathematical ceilings (SYNTHETIC; no bar for the true inverse of raindrops)", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "chain_limits.png", dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    main()
