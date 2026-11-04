"""STAGE 7 - the frequency domain: do the equations we derived predict the spectra, and does subtracting the haze work?

PART 1  THEORY CHECK. docs/model-equations.md derives, for every medium, how it changes a picture's spectrum. The key quantity is
        the spectral ratio  R(f) = |spectrum of degraded| / |spectrum of clear|,  averaged over rings of equal frequency f. Predictions:
          haze with constant t            R(f) = t at every f > 0 (the veil only adds to f = 0)
          plastic (blur + attenuation)    R(f) = t * tint * exp(-2 pi^2 sigma^2 f^2)
          water                           R_c(f) = t_c for each colour c
          glass droplets (masked blend)   R(f) = 1 - <w> (1 - 0.9 g(f)),  <w> = average blend weight, g = blur transfer function
          two media stacked               R(f) = product of the single-layer ratios
        We measure R(f) on synthetic pictures made by our code (a correct derivation and a correct program must agree) and on real
        pictures (real haze pairs, real raindrop pairs), where it tells us how real haze and rain behave.
PART 2  SUBTRACTION. Haze removal by frequency-domain methods that follow from the theory (subtract the slow veil; invert the model
        with t and the veil estimated from priors; a radial filter measured from half of the real pairs as an upper bound for any
        space-invariant isotropic linear filter), against DCP, on held-out real haze pairs and on their calibrated synthetic twins.
One job = one picture (pair); parallel and resumable.

Output:  results/tables/theory_checks.csv  theory_real_haze.csv  frequency_subtraction.csv  frequency_subtraction_summary.csv  theory_curves.json
         results/figures/theory_spectral_ratio.png  frequency_filter.png  frequency_subtraction.png
Run from the project root (after validate_models.py and tune_models.py):  .venv/bin/python code/experiments/run_frequency_analysis.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from common.spectra import amplitude_ratio, frequency_radius, padded_spectrum, ring_edges, ring_means  # noqa: E402
from common.stats import luminance  # noqa: E402
from degrade import glass, haze, plastic, water  # noqa: E402
from degrade.stack import apply_stack  # noqa: E402
from experiments.validate_models import SOURCES, ramp_model, read_pairs  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.frequency import apply_radial_filter  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE, SCENES = 256, 12
EDGES = ring_edges(36, 0.5)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
FIT_RANGE = (CENTRES > 0.03) & (CENTRES < 0.45)  # rings used for errors and fits: not the zero-frequency ring, not the very corners


def channel_ratio(degraded, clear, channel):
    """Spectral ratio of one colour channel (0, 1, 2) or of the luminance (None)."""
    pick = (lambda image: luminance(image)) if channel is None else (lambda image: image[:, :, channel])
    return amplitude_ratio(pick(degraded), pick(clear), EDGES)


def synthetic_case(case, clear, depth, calibrated_glass):
    """Degrade one clear scene for a theory case. Returns (degraded picture, predicted ratio curves {name: curve over CENTRES})."""
    flat = np.full_like(depth, 0.5)  # constant depth, so t is constant and the prediction is exact
    if case == "haze_flat":
        degraded = np.clip(haze.apply(clear, flat, {}), 0, 1)
        t = float(np.exp(-haze.DEFAULTS["beta"] * haze.DEFAULTS["scene_depth"] * 0.5))
        return degraded, {"luminance": np.full(CENTRES.shape, t)}
    if case == "haze_depth":
        degraded = np.clip(haze.apply(clear, depth, {}), 0, 1)
        t_map = haze.transmission(depth, haze.DEFAULTS)[:, :, 0]
        grey = luminance(clear)
        detail = np.gradient(grey, axis=0) ** 2 + np.gradient(grey, axis=1) ** 2  # where the picture has detail (edges, texture)
        # first guess: the average t over the picture; refinement: detail lives where it lives, so weight t^2 by the detail energy
        return degraded, {"luminance": np.full(CENTRES.shape, float(t_map.mean())),
                          "luminance_refined": np.full(CENTRES.shape, float(np.sqrt((t_map ** 2 * detail).sum() / detail.sum())))}
    if case == "plastic":
        s = plastic.DEFAULTS
        sigma = s["blur"] * max(clear.shape[:2])
        tint = float(luminance(np.array([[[s["tint_red"], s["tint_green"], s["tint_blue"]]]]))[0, 0])
        return np.clip(plastic.apply(clear, None, {}), 0, 1), {"luminance": s["transmission"] * tint * np.exp(-2 * (np.pi * sigma * CENTRES) ** 2)}
    if case == "water_flat":
        degraded = np.clip(water.apply(clear, flat, {}), 0, 1)
        t = water.transmission(flat, water.DEFAULTS)[0, 0]
        return degraded, {f"channel {c}": np.full(CENTRES.shape, float(t[c])) for c in range(3)}
    if case == "glass_drops":
        settings = {**glass.DEFAULTS, "veil": 0.0, "smudge": 0.0, **calibrated_glass}
        weight = float((settings["droplet_strength"] * glass.droplet_mask(clear.shape[0], clear.shape[1], settings)).mean())
        sigma = settings["droplet_blur"] * max(clear.shape[:2])
        blend_weight = settings["droplet_strength"] * glass.droplet_mask(clear.shape[0], clear.shape[1], settings)
        lost = 1.0 - 0.9 * np.exp(-2 * (np.pi * sigma * CENTRES) ** 2)  # fraction of a frequency removed where a droplet sits
        # first guess: only the average blend weight matters. Refinement: the droplets' random pattern also scatters energy from
        # other frequencies into this one (an extra, incoherent term that grows with the variance of the blend weight)
        return np.clip(glass.apply(clear, None, {"veil": 0.0, "smudge": 0.0, **calibrated_glass}), 0, 1), {
            "luminance": 1.0 - weight * lost, "luminance_refined": np.sqrt((1.0 - weight * lost) ** 2 + blend_weight.var() * lost ** 2)}
    # stack: water behind plastic, flat depth; per channel the prediction is the product of the layers' ratios
    layers = [{"type": "water"}, {"type": "plastic"}]
    t_water = water.transmission(flat, water.DEFAULTS)[0, 0]
    s = plastic.DEFAULTS
    sigma = s["blur"] * max(clear.shape[:2])
    tint = [s["tint_red"], s["tint_green"], s["tint_blue"]]
    return apply_stack(clear, flat, layers), {f"channel {c}": float(t_water[c]) * tint[c] * s["transmission"] * np.exp(-2 * (np.pi * sigma * CENTRES) ** 2) for c in range(3)}


def work(job):
    """One job (in a worker). Kinds: theory_synthetic, theory_real, subtract."""
    kind = job["kind"]
    if kind == "theory_synthetic":
        clear, depth, scene = load_clear_scenes(SCENES, SIDE, only=job["scene_index"])[0]
        degraded, predictions = synthetic_case(job["case"], clear, depth, job["glass"])
        channels = [None] if all(k.startswith("luminance") for k in predictions) else [0, 1, 2]
        measured = {("luminance" if c is None else f"channel {c}"): channel_ratio(degraded, clear, c).tolist() for c in channels}
        return [{"case": job["case"], "scene": scene, "measured": measured, "predicted": {k: v.tolist() for k, v in predictions.items()}}]
    if kind == "theory_real":
        clear, degraded = load_image(job["reference"], SIDE), load_image(job["degraded"], SIDE)
        return [{"medium": job["medium"], "source": job["source"], "picture": job["id"], "ratio": channel_ratio(degraded, clear, None).tolist()}]
    # kind == "subtract": restore the real picture and its synthetic twin with every method
    clear, real = load_image(job["reference"], SIDE), load_image(job["degraded"], SIDE)
    twin = ramp_model(clear, job["tau"], job["airlight"])
    rows = []
    for spec in job["methods"]:
        if spec["name"] == "oracle_radial":
            method = lambda image: apply_radial_filter(image, EDGES, job["oracle_gains"])
        else:
            method = build_method(spec)
        for world, picture in (("real", real), ("twin", twin)):
            scores = full_reference_scores(method(picture), clear)
            rows.append({"source": job["source"], "picture": job["id"], "world": world, "method": spec["label"], "psnr": scores["psnr"], "ssim": scores["ssim"]})
    return rows


def learn_radial_filters(pairs_by_source):
    """For each dataset, the best isotropic linear filter measured on its CALIBRATION half: gain per ring = sum Re(clear x conj(hazy)) / sum |hazy|^2.

    This is a least-squares fit of 36 numbers (one per ring); it is NOT proposed as a restoration method, only used to learn the most
    that any space-invariant isotropic linear frequency filter could achieve, and what shape it has.
    """
    filters = {}
    for source, pairs in pairs_by_source.items():
        numerator, denominator = np.zeros(len(CENTRES)), np.zeros(len(CENTRES))
        for i, pair in enumerate(pairs):
            if i % 2 == 0:
                spectrum_h, _ = padded_spectrum(luminance(load_image(pair["degraded"], SIDE)))
                spectrum_c, _ = padded_spectrum(luminance(load_image(pair["reference"], SIDE)))
                radius = frequency_radius(spectrum_h.shape)
                numerator += np.nan_to_num(ring_means((spectrum_c * np.conj(spectrum_h)).real, radius, EDGES))
                denominator += np.nan_to_num(ring_means(np.abs(spectrum_h) ** 2, radius, EDGES))
        filters[source] = (numerator / np.maximum(denominator, 1e-12)).tolist()
    return filters


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), restval="")
        writer.writeheader()
        writer.writerows(rows)


def run(workers=None, resume=True):
    with open(TABLES / "model_validation_summary.csv") as handle:
        calibration = {r["source"]: (float(r["calibrated_tau"]), float(r["calibrated_airlight"])) for r in csv.DictReader(handle)}
    with open(TABLES / "model_fit_haze.csv") as handle:
        affine_t = {r["picture"]: float(r["affine_t"]) for r in csv.DictReader(handle)}
    calibrated_glass = {k: v for k, v in json.loads((PROJECT / "configs" / "calibrated_presets.json").read_text())["glass"]["medium"].items() if k != "structure"}
    haze_pairs, glass_pairs = read_pairs("haze"), read_pairs("glass")
    by_source = {s: [p for p in haze_pairs if p["source"] == s] for s in SOURCES}
    filters = learn_radial_filters(by_source)
    stamp = fingerprint(SIDE, SCENES, calibrated_glass, filters, calibration)

    # --- PART 1: theory check jobs
    cases = ("haze_flat", "haze_depth", "plastic", "water_flat", "glass_drops", "stack_water_plastic")
    jobs = [{"key": f"syn|{c}|{i}", "kind": "theory_synthetic", "case": c, "scene_index": i, "glass": calibrated_glass} for c in cases for i in range(SCENES)]
    jobs += [{"key": f"real|haze|{p['id']}", "kind": "theory_real", "medium": "haze", **p} for p in haze_pairs]
    jobs += [{"key": f"real|glass|{p['id']}", "kind": "theory_real", "medium": "glass", **p} for p in glass_pairs]

    # --- PART 2: subtraction jobs on the held-out (odd) half of each dataset's pairs
    methods = [{"label": "no restoration", "name": "none"},
               {"label": "subtract low frequencies (sigma 0.05)", "name": "lowfreq_subtract", "params": {"sigma": 0.05}},
               {"label": "subtract low frequencies (sigma 0.10)", "name": "lowfreq_subtract", "params": {"sigma": 0.10}},
               {"label": "subtract low frequencies (sigma 0.20)", "name": "lowfreq_subtract", "params": {"sigma": 0.20}},
               {"label": "invert model, constant veil", "name": "blind_veil"},
               {"label": "invert model, local veil", "name": "blind_veil", "params": {"local_sigma": 0.15}},
               {"label": "homomorphic", "name": "homomorphic"},
               {"label": "frequency Laplacian", "name": "freq_laplacian"},
               {"label": "oracle radial filter (upper bound)", "name": "oracle_radial"},
               {"label": "DCP (guided)", "name": "dcp_guided"},
               {"label": "DCP then CLAHE", "name": "dcp_clahe"}]
    for source in SOURCES:
        for i, pair in enumerate(by_source[source]):
            if i % 2 == 1:
                jobs.append({"key": f"sub|{pair['id']}", "kind": "subtract", "tau": calibration[source][0], "airlight": calibration[source][1],
                             "methods": methods, "oracle_gains": filters[source], **pair})
    rows = run_stage("frequency_analysis", stamp, jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)

    # --- PART 1 results: measured against predicted spectral ratio
    synthetic = [r for r in rows if "case" in r]
    checks, curves = [], {}
    for case in cases:
        own = [r for r in synthetic if r["case"] == case]
        for name in own[0]["predicted"]:
            measured = np.mean([r["measured"]["luminance" if name.startswith("luminance") else name] for r in own], axis=0)
            predicted = np.mean([r["predicted"][name] for r in own], axis=0)
            error = float(np.sqrt(np.mean((measured[FIT_RANGE] - predicted[FIT_RANGE]) ** 2)))
            relative = float(np.mean(np.abs(measured[FIT_RANGE] / predicted[FIT_RANGE] - 1)))
            curves[f"{case}|{name}"] = {"measured": measured.tolist(), "predicted": predicted.tolist()}
            checks.append({"case": case, "curve": name, "rms_error": round(error, 4), "mean_relative_error": round(relative, 4),
                           "measured_mid": round(float(np.median(measured[FIT_RANGE])), 3), "predicted_mid": round(float(np.median(predicted[FIT_RANGE])), 3)})
    real = [r for r in rows if "ratio" in r]
    curves["real_haze"] = {s: np.nanmean([r["ratio"] for r in real if r["medium"] == "haze" and r["source"] == s], axis=0).tolist() for s in SOURCES}
    curves["real_glass"] = np.nanmean([r["ratio"] for r in real if r["medium"] == "glass"], axis=0).tolist()
    (TABLES / "theory_curves.json").write_text(json.dumps({"centres": CENTRES.tolist(), "curves": curves, "filters": filters}))
    # real haze: the model says R(f) = t, so the measured middle of R should equal the affine-fit t of the same pair
    real_haze = [r for r in real if r["medium"] == "haze"]
    pairs_t = [(affine_t[r["picture"]], float(np.nanmedian(np.array(r["ratio"])[FIT_RANGE]))) for r in real_haze if r["picture"] in affine_t]
    correlation = float(np.corrcoef([a for a, _ in pairs_t], [b for _, b in pairs_t])[0, 1])
    save_csv([{"picture": r["picture"], "source": r["source"], "affine_t": affine_t.get(r["picture"], ""), "spectral_ratio_mid": float(np.nanmedian(np.array(r["ratio"])[FIT_RANGE]))} for r in real_haze], TABLES / "theory_real_haze.csv")
    # real raindrops: fit <w> and sigma of the droplet formula to the measured curve (least squares over a small grid)
    real_glass_curve = np.array(curves["real_glass"])
    best = min(((float(np.mean((1 - w * (1 - 0.9 * np.exp(-2 * (np.pi * s * CENTRES) ** 2)) - real_glass_curve)[FIT_RANGE] ** 2)), w, s)
                for w in np.arange(0.0, 0.61, 0.01) for s in np.arange(0.5, 12.01, 0.25)))
    checks.append({"case": "real raindrops (formula fitted to the data)", "curve": "luminance", "rms_error": round(float(np.sqrt(best[0])), 4), "mean_relative_error": "",
                   "measured_mid": round(float(np.median(real_glass_curve[FIT_RANGE])), 3), "predicted_mid": f"fitted weight {best[1]:.2f}, blur sigma {best[2]:.1f} px"})
    checks.append({"case": "real haze: spectral ratio against affine-fit t, 185 pairs", "curve": "correlation", "rms_error": "", "mean_relative_error": "",
                   "measured_mid": round(correlation, 3), "predicted_mid": ""})
    save_csv(checks, TABLES / "theory_checks.csv")
    print("\nTheory check: measured against predicted spectral ratio R(f) on synthetic pictures")
    for c in checks:
        print(f"  {c['case']:42}{c['curve']:12} RMS error {c['rms_error']!s:>7}   measured {c['measured_mid']!s:>6} against predicted {c['predicted_mid']}")

    # --- PART 2 results: subtraction methods
    sub = [r for r in rows if "world" in r]
    labels = [m["label"] for m in methods]
    summary = []
    for world in ("real", "twin"):
        for label in labels:
            own = [r for r in sub if r["world"] == world and r["method"] == label]
            summary.append({"world": world, "method": label, "pictures": len(own), "psnr": round(float(np.mean([r["psnr"] for r in own])), 2), "ssim": round(float(np.mean([r["ssim"] for r in own])), 3)})
    save_csv(summary, TABLES / "frequency_subtraction_summary.csv")
    save_csv(sub, TABLES / "frequency_subtraction.csv")
    print("\nHaze removal by subtraction and filtering (held-out real pairs and their calibrated synthetic twins), mean PSNR / SSIM")
    for label in labels:
        r = next(x for x in summary if x["world"] == "real" and x["method"] == label)
        t = next(x for x in summary if x["world"] == "twin" and x["method"] == label)
        print(f"  {label:40} real {r['psnr']:6.2f} / {r['ssim']:.3f}    twin {t['psnr']:6.2f} / {t['ssim']:.3f}")

    # --- figures
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    def panel(axis, key, title):
        axis.plot(CENTRES, curves[key]["measured"], "-", linewidth=2, label="measured")
        axis.plot(CENTRES, curves[key]["predicted"], "--", linewidth=1.6, label="first formula")
        if key.replace("|luminance", "|luminance_refined") in curves:
            axis.plot(CENTRES, curves[key.replace("|luminance", "|luminance_refined")]["predicted"], "-.", linewidth=1.6, color="#2e7d4f", label="refined formula")
        axis.set_title(title, fontsize=9)
    panel(axes[0][0], "haze_flat|luminance", "haze, constant t: R(f) = t")
    panel(axes[0][1], "haze_depth|luminance", "haze with depth: mean t, then detail-weighted")
    panel(axes[0][2], "plastic|luminance", "plastic: t exp(-2 pi^2 sigma^2 f^2)")
    for c, colour in zip(range(3), ("#c0392b", "#27ae60", "#2980b9")):
        axes[0][3].plot(CENTRES, curves[f"water_flat|channel {c}"]["measured"], "-", color=colour, linewidth=2)
        axes[0][3].plot(CENTRES, curves[f"water_flat|channel {c}"]["predicted"], "--", color=colour)
    axes[0][3].set_title("water: R_c(f) = t_c (solid measured, dashed predicted)", fontsize=9)
    panel(axes[1][0], "glass_drops|luminance", "droplets: average weight, then plus variance term")
    axes[1][0].plot(CENTRES, curves["real_glass"], ":", color="#b05d0c", linewidth=2, label="real raindrops")
    for c, colour in zip(range(3), ("#c0392b", "#27ae60", "#2980b9")):
        axes[1][1].plot(CENTRES, curves[f"stack_water_plastic|channel {c}"]["measured"], "-", color=colour, linewidth=2)
        axes[1][1].plot(CENTRES, curves[f"stack_water_plastic|channel {c}"]["predicted"], "--", color=colour)
    axes[1][1].set_title("stack water + plastic: product of ratios", fontsize=9)
    for source, colour in zip(SOURCES, ("#3d6299", "#2e7d4f", "#b05d0c", "#6a3d99")):
        axes[1][2].plot(CENTRES, curves["real_haze"][source], color=colour, label=source)
    axes[1][2].set_title("real haze: measured R(f) by dataset", fontsize=9)
    axes[1][3].scatter([a for a, _ in pairs_t], [b for _, b in pairs_t], s=10, color="#3d6299")
    axes[1][3].plot([0, 1], [0, 1], "--", color="#888888")
    axes[1][3].set_xlabel("t from the spatial affine fit")
    axes[1][3].set_ylabel("t from the spectral ratio")
    axes[1][3].set_title(f"real haze: two estimates of t agree (r = {correlation:+.2f})", fontsize=9)
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        if axis not in (axes[1][3],):
            axis.set_xlabel("frequency (cycles per pixel)")
    axes[0][0].set_ylabel("spectral ratio R(f)")
    axes[1][0].legend(fontsize=7)
    axes[0][0].legend(fontsize=7)
    axes[0][1].legend(fontsize=7)
    axes[1][2].legend(fontsize=7)
    figure.suptitle("Derived spectral predictions against measurement")
    figure.tight_layout()
    figure.savefig(FIGURES / "theory_spectral_ratio.png", dpi=105)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.4))
    for source, colour in zip(SOURCES, ("#3d6299", "#2e7d4f", "#b05d0c", "#6a3d99")):
        axis.plot(CENTRES, filters[source], color=colour, label=f"{source} (learned)")
        axis.axhline(1.0 / np.median([affine_t[p["id"]] for p in by_source[source] if p["id"] in affine_t]), color=colour, linestyle=":", linewidth=1)
    axis.set_xlabel("frequency (cycles per pixel)")
    axis.set_ylabel("gain of the best isotropic linear filter")
    axis.set_title("What the best frequency filter looks like (dotted: theory, gain 1/t at every non-zero frequency)")
    axis.set_ylim(0, 8)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "frequency_filter.png", dpi=105)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(labels))
    axis.bar(x - 0.2, [next(s["psnr"] for s in summary if s["world"] == "real" and s["method"] == l) for l in labels], 0.4, label="real haze (held-out pairs)", color="#b05d0c")
    axis.bar(x + 0.2, [next(s["psnr"] for s in summary if s["world"] == "twin" and s["method"] == l) for l in labels], 0.4, label="calibrated synthetic twin", color="#3d6299")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axis.set_ylabel("PSNR against the clear picture (dB)")
    axis.set_title("Removing haze by subtraction and frequency filtering")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "frequency_subtraction.png", dpi=105)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
