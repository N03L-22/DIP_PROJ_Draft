"""STAGE 2c - does our synthetic degradation look like the real thing? (The proposal's 'validate the model against real captures'.)

We have real before/after pairs for haze (185 pairs, four datasets) and for raindrops on glass (58 pairs). Two checks:

  A. FIT (haze): try to reproduce each real hazy picture from its real clear picture with our model I = J t + A (1 - t).
     Three predictions are scored by PSNR against the real hazy picture:
        no model       the clear picture itself (what you get if you ignore the haze)
        affine         one number a and b per colour:  hazy = a clear + b   (t constant over the picture, a free airlight per colour)
        ramp model     our model with t = exp(-tau * ramp), ramp running from near (bottom) to far (top), one airlight A
     The ramp model is what the synthetic generator can do without measured depth. Then we CALIBRATE: take the middle values of
     tau and A from half of the pairs of a dataset and test them, fixed, on the other half (held out).
  B. SIGNATURE (haze and glass): five simple numbers that describe how a picture changed (dark-channel shift, contrast ratio,
     brightness shift, share of pixels changed by more than 0.1, sharpness ratio), measured on the real pairs and on our synthetic
     pairs at mild, medium and heavy. If the synthetic numbers fall in the real range, the model is plausible; if not, it is not.
One job = one picture pair; parallel and resumable.

Output:  results/tables/model_fit_haze.csv  model_signature.csv  model_validation_summary.csv
         results/figures/model_fit_haze.png  model_signature.png  model_fit_examples.jpg
Run from the project root:  .venv/bin/python code/experiments/validate_models.py
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.filters import dark_channel  # noqa: E402
from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from common.stats import luminance  # noqa: E402
from degrade.presets import LEVELS, PRESETS  # noqa: E402
from degrade.stack import MODELS  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
CURATED = PROJECT / "data" / "curated"
SIDE = 256
TAU_GRID = np.arange(0.0, 8.01, 0.05)  # tau = beta x scene depth: the optical thickness at the far end of the picture
SYNTHETIC_SCENES = 12
SOURCES = ("O-HAZE", "I-HAZE", "Dense-Haze", "NH-Haze")


def signature(degraded, clear):
    """Five numbers describing how `degraded` differs from `clear`: the same numbers are measured on real and synthetic pairs."""
    grey_d, grey_c = luminance(degraded), luminance(clear)
    gradient = lambda g: np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()  # total edge strength
    return {"dark_shift": float(dark_channel(degraded).mean() - dark_channel(clear).mean()),
            "contrast_ratio": float(grey_d.std() / max(grey_c.std(), 1e-6)),
            "brightness_shift": float(grey_d.mean() - grey_c.mean()),
            "changed_share": float((np.abs(grey_d - grey_c) > 0.1).mean()),
            "sharpness_ratio": float(gradient(grey_d) / max(gradient(grey_c), 1e-6))}


def ramp_model(clear, tau, airlight):
    """Our haze model with a planar ramp for depth: t = exp(-tau * r(y)), r = 1 at the top (far) down to 0 at the bottom (near)."""
    ramp = np.linspace(1.0, 0.0, clear.shape[0])[:, None, None]
    t = np.exp(-tau * ramp)
    return np.clip(clear * t + airlight * (1.0 - t), 0, 1)


def fit_ramp_model(clear, hazy):
    """Find tau and A that make the ramp model closest to the real hazy picture (least squares).

    For each tau on a grid the best airlight has a closed form: hazy - clear t = A (1 - t) + error, so A is a simple ratio of sums.
    """
    ramp = np.linspace(1.0, 0.0, clear.shape[0])[:, None, None]
    best = (np.inf, 0.0, 0.0)
    for tau in TAU_GRID:
        t = np.exp(-tau * ramp)
        weight = 1.0 - t
        if np.sum(weight ** 2) < 1e-9:
            airlight = 0.8  # tau = 0 leaves the picture unchanged; the airlight does not matter
        else:
            airlight = float(np.clip(np.sum((hazy - clear * t) * weight) / (3 * np.sum(weight[:, :, 0] ** 2) * clear.shape[1]), 0.0, 1.0))
        error = float(np.mean((np.clip(clear * t + airlight * weight, 0, 1) - hazy) ** 2))
        if error < best[0]:
            best = (error, float(tau), airlight)
    return best[1], best[2]


def affine_fit(clear, hazy):
    """Per colour, the straight line hazy = a clear + b that fits best. a is the transmission, b = A (1 - a) the airlight part."""
    coefficients = [np.polyfit(clear[:, :, c].ravel(), hazy[:, :, c].ravel(), 1) for c in range(3)]
    prediction = np.stack([np.clip(a * clear[:, :, c] + b, 0, 1) for c, (a, b) in enumerate(coefficients)], axis=-1)
    return prediction, [float(a) for a, _ in coefficients], [float(b) for _, b in coefficients]


def work(job):
    """One job (in a worker). Kinds: 'fit' (haze pair), 'eval' (haze pair with fixed calibrated parameters), 'real_signature', 'synthetic_signature'."""
    kind = job["kind"]
    if kind == "synthetic_signature":
        clear, depth, scene = load_clear_scenes(SYNTHETIC_SCENES, SIDE, only=job["scene_index"])[0]
        degraded = np.clip(MODELS[job["medium"]].apply(clear, depth, PRESETS[job["medium"]][job["level"]]), 0, 1)
        return [{"world": "synthetic", "medium": job["medium"], "group": job["level"], "picture": scene, **signature(degraded, clear)}]
    hazy, clear = load_image(job["degraded"], SIDE), load_image(job["reference"], SIDE)
    if kind == "real_signature":
        return [{"world": "real", "medium": job["medium"], "group": job["source"], "picture": job["id"], **signature(hazy, clear)}]
    if kind == "eval":  # calibrated parameters from other pairs, applied unchanged
        return [{"source": job["source"], "picture": job["id"], "psnr_no_model": full_reference_scores(clear, hazy)["psnr"],
                 "psnr_calibrated": full_reference_scores(ramp_model(clear, job["tau"], job["airlight"]), hazy)["psnr"]}]
    tau, airlight = fit_ramp_model(clear, hazy)  # kind == "fit"
    affine, a_values, b_values = affine_fit(clear, hazy)
    return [{"source": job["source"], "picture": job["id"], "psnr_no_model": full_reference_scores(clear, hazy)["psnr"],
             "psnr_affine": full_reference_scores(affine, hazy)["psnr"], "psnr_ramp": full_reference_scores(ramp_model(clear, tau, airlight), hazy)["psnr"],
             "tau": tau, "airlight": airlight, "affine_t": float(np.mean(a_values))}]


def read_pairs(medium):
    """Every real before/after pair of one medium from the curated manifest: dicts with the two file paths."""
    with open(CURATED / f"manifest_{medium}.csv") as handle:
        return [{"source": r["source"], "id": f"{r['source']}_{r['id']}", "degraded": str(CURATED / r["degraded_path"]), "reference": str(CURATED / r["reference_path"])}
                for r in csv.DictReader(handle)]


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(workers=None, resume=True):
    tables, figures = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
    haze_pairs, glass_pairs = read_pairs("haze"), read_pairs("glass")
    stamp = fingerprint(SIDE, TAU_GRID.tolist(), len(haze_pairs), len(glass_pairs))

    # --- A. fit the haze model to every real pair
    fit_jobs = [{"key": p["id"], "kind": "fit", **p} for p in haze_pairs]
    fits = run_stage("validate_fit", stamp, fit_jobs, work, workers=workers, memory_per_worker_gb=0.6, resume=resume)
    # calibrate on every second pair of each dataset (median tau and A), then test those fixed values on the other pairs
    calibration = {}
    for source in SOURCES:
        own = [f for i, f in enumerate([f for f in fits if f["source"] == source]) if i % 2 == 0]
        calibration[source] = (float(np.median([f["tau"] for f in own])), float(np.median([f["airlight"] for f in own])))
    eval_jobs = []
    for source in SOURCES:
        for i, p in enumerate([p for p in haze_pairs if p["source"] == source]):
            if i % 2 == 1:
                eval_jobs.append({"key": p["id"], "kind": "eval", "tau": calibration[source][0], "airlight": calibration[source][1], **p})
    evals = run_stage("validate_eval", fingerprint(stamp, calibration), eval_jobs, work, workers=workers, memory_per_worker_gb=0.6, resume=resume)
    save_csv(fits, tables / "model_fit_haze.csv")

    # --- B. signature numbers on real pairs and synthetic pairs
    signature_jobs = [{"key": f"real|haze|{p['id']}", "kind": "real_signature", "medium": "haze", **p} for p in haze_pairs]
    signature_jobs += [{"key": f"real|glass|{p['id']}", "kind": "real_signature", "medium": "glass", **p} for p in glass_pairs]
    signature_jobs += [{"key": f"syn|{m}|{lv}|{i}", "kind": "synthetic_signature", "medium": m, "level": lv, "scene_index": i}
                       for m in ("haze", "glass") for lv in LEVELS for i in range(SYNTHETIC_SCENES)]
    signatures = run_stage("validate_signature", stamp, signature_jobs, work, workers=workers, memory_per_worker_gb=0.6, resume=resume)
    save_csv(signatures, tables / "model_signature.csv")

    # --- summary table and numbers
    summary = []
    for source in SOURCES:
        own, held = [f for f in fits if f["source"] == source], [e for e in evals if e["source"] == source]
        summary.append({"source": source, "pairs": len(own), "psnr_no_model": np.mean([f["psnr_no_model"] for f in own]), "psnr_affine": np.mean([f["psnr_affine"] for f in own]),
                        "psnr_ramp_fit": np.mean([f["psnr_ramp"] for f in own]), "median_tau": np.median([f["tau"] for f in own]), "median_airlight": np.median([f["airlight"] for f in own]),
                        "heldout_no_model": np.mean([e["psnr_no_model"] for e in held]), "heldout_calibrated": np.mean([e["psnr_calibrated"] for e in held]),
                        "calibrated_tau": calibration[source][0], "calibrated_airlight": calibration[source][1]})
    save_csv(summary, tables / "model_validation_summary.csv")
    print("\nCan our haze model reproduce real hazy pictures? (PSNR against the real hazy picture, dB)")
    print(f"  {'dataset':11}{'no model':>10}{'affine':>9}{'ramp fit':>10}{'held-out: none':>16}{'calibrated':>12}{'  tau':>7}{'  A':>6}")
    for s in summary:
        print(f"  {s['source']:11}{s['psnr_no_model']:10.2f}{s['psnr_affine']:9.2f}{s['psnr_ramp_fit']:10.2f}{s['heldout_no_model']:16.2f}{s['heldout_calibrated']:12.2f}{s['median_tau']:7.2f}{s['median_airlight']:6.2f}")
    print("  our synthetic presets: tau (= beta x 2.0) is 1.0 mild, 2.4 medium, 4.8 heavy; airlight 0.92")
    metrics = ("dark_shift", "contrast_ratio", "brightness_shift", "changed_share", "sharpness_ratio")
    print("\nSignature of the change (mean over pictures): real range against synthetic mild | medium | heavy")
    for medium in ("haze", "glass"):
        real_rows = [r for r in signatures if r["world"] == "real" and r["medium"] == medium]
        print(f"  {medium}:")
        for m in metrics:
            syn = [np.mean([r[m] for r in signatures if r["world"] == "synthetic" and r["medium"] == medium and r["group"] == lv]) for lv in LEVELS]
            print(f"    {m:17} real {np.mean([r[m] for r in real_rows]):+.3f} (middle half {np.percentile([r[m] for r in real_rows], 25):+.3f} to {np.percentile([r[m] for r in real_rows], 75):+.3f})   synthetic {syn[0]:+.3f} | {syn[1]:+.3f} | {syn[2]:+.3f}")

    # --- figure 1: fit quality and fitted optical thickness
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(len(SOURCES))
    for offset, (key, label, colour) in enumerate((("psnr_no_model", "no model (clear picture)", "#888888"), ("psnr_affine", "affine fit", "#2e7d4f"),
                                                    ("psnr_ramp_fit", "our model, ramp depth, fitted", "#3d6299"), ("heldout_calibrated", "our model, calibrated on other pairs", "#b05d0c"))):
        left.bar(x + (offset - 1.5) * 0.2, [s[key] for s in summary], 0.2, label=label, color=colour)
    left.set_xticks(x)
    left.set_xticklabels(SOURCES)
    left.set_ylabel("PSNR against the real hazy picture (dB)")
    left.set_title("How closely can our haze model reproduce real haze?")
    left.legend(fontsize=7)
    left.grid(axis="y", alpha=0.25)
    for source, colour in zip(SOURCES, ("#3d6299", "#2e7d4f", "#b05d0c", "#6a3d99")):
        right.hist([f["tau"] for f in fits if f["source"] == source], bins=np.arange(0, 8.5, 0.5), alpha=0.55, label=source, color=colour)
    for tau, label in ((1.0, "mild"), (2.4, "medium"), (4.8, "heavy")):
        right.axvline(tau, color="black", linestyle="--", linewidth=1)
        right.text(tau, right.get_ylim()[1] * 0.92, f" {label}", fontsize=8)
    right.set_xlabel("fitted optical thickness tau at the far end (dashed: our synthetic presets)")
    right.set_ylabel("number of real pairs")
    right.set_title("How thick is real fog, in the model's units?")
    right.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "model_fit_haze.png", dpi=105)
    plt.close(figure)

    # --- figure 2: signature numbers, real against synthetic
    figure, axes = plt.subplots(2, len(metrics), figsize=(16, 6))
    for row, medium in enumerate(("haze", "glass")):
        real_rows = [r for r in signatures if r["world"] == "real" and r["medium"] == medium]
        for column, m in enumerate(metrics):
            axis = axes[row][column]
            groups = [("real", [r[m] for r in real_rows], "#b05d0c")] + [(lv, [r[m] for r in signatures if r["world"] == "synthetic" and r["medium"] == medium and r["group"] == lv], "#3d6299") for lv in LEVELS]
            axis.boxplot([g[1] for g in groups], showfliers=False, patch_artist=True, boxprops={"facecolor": "#dddddd"}, medianprops={"color": "black"})
            for index, g in enumerate(groups, start=1):
                axis.scatter([index] * 1, [np.mean(g[1])], color=g[2], zorder=3)
            axis.set_xticklabels(["real"] + [lv[:3] for lv in LEVELS], fontsize=8)
            axis.set_title(f"{medium}: {m}", fontsize=9)
            axis.grid(alpha=0.25)
    figure.suptitle("Does the synthetic change look like the real change? Boxes: middle half of the pictures; dots: means (orange real, blue synthetic)")
    figure.tight_layout()
    figure.savefig(figures / "model_signature.png", dpi=105)
    plt.close(figure)

    # --- figure 3: examples, one real pair per dataset: clear | real hazy | affine fit | ramp model fit
    tiles_per_row = []
    for source in SOURCES:
        pair = next(p for p in haze_pairs if p["source"] == source)
        clear, hazy = load_image(pair["reference"], 256), load_image(pair["degraded"], 256)
        fit = next(f for f in fits if f["picture"] == pair["id"])
        tiles_per_row.append([("clear (real)", clear), ("hazy (real)", hazy), (f"ramp model tau={fit['tau']:.1f} A={fit['airlight']:.2f}: {fit['psnr_ramp']:.1f} dB", ramp_model(clear, fit["tau"], fit["airlight"])),
                              (f"affine fit: {fit['psnr_affine']:.1f} dB", affine_fit(clear, hazy)[0])])
    rows_images = []
    for tiles in tiles_per_row:
        framed = []
        for caption, picture in tiles:
            tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            tile.thumbnail((230, 230))
            canvas = Image.new("RGB", (tile.width, tile.height + 18), (30, 30, 30))
            canvas.paste(tile, (0, 18))
            ImageDraw.Draw(canvas).text((3, 3), caption[:40], fill=(225, 225, 225))
            framed.append(canvas)
        rows_images.append(framed)
    sheet = Image.new("RGB", (max(sum(t.width for t in r) for r in rows_images), sum(max(t.height for t in r) for r in rows_images)), (30, 30, 30))
    y = 0
    for framed in rows_images:
        xpos = 0
        for tile in framed:
            sheet.paste(tile, (xpos, y))
            xpos += tile.width
        y += max(t.height for t in framed)
    sheet.save(figures / "model_fit_examples.jpg", quality=92)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
