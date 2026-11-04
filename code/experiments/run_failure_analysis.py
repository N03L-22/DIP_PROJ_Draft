"""STAGE 6a - WHY do methods succeed or fail? Dark-channel statistics, transmission accuracy and colour cast.

Three questions, each answered with numbers and a figure:
  1. Does the DCP assumption hold? DCP assumes that in a clear outdoor picture almost every patch has a pixel with one
     colour near 0 (dark channel near 0). We measure the dark channel of the CLEAR pictures of every dataset. Where it is not
     near 0 (indoor rooms, bright underwater scenes) the prior is violated and DCP is expected to fail.
  2. Does DCP find the right transmission? On synthetic pictures we know the true transmission t and airlight A, so we can
     compare DCP's estimates with the truth, per medium and severity.
  3. What happens to colour? The colour cast (log of red over the average of green and blue) before and after restoration.
One job = one dataset (question 1) or one scene at one medium and severity (questions 2 and 3); parallel and resumable.

Output:  results/tables/failure_prior_check.csv   failure_transmission.csv
         results/figures/failure_prior_check.png  failure_dark_channel.png  failure_transmission.png  failure_cast.png
         results/figures/failure_cases_synthetic.jpg  failure_cases_real.jpg
Run from the project root:  .venv/bin/python code/experiments/run_failure_analysis.py
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
from common.stats import colour_cast  # noqa: E402
from degrade import glass, haze, plastic, water  # noqa: E402
from degrade.presets import LEVELS, active_presets  # noqa: E402

PRESETS = active_presets()  # DIP_PRESETS=calibrated switches the whole run to the calibrated settings
from degrade.stack import MODELS  # noqa: E402
from experiments.run_experiments import list_pairs  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.dcp import dcp_restore  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
CURATED = PROJECT / "data" / "curated"
MEDIA = ("haze", "water", "glass", "plastic")
SIDE = 256            # pictures are shrunk to this longest side
SCENES = 12           # NYU scenes per medium and severity
MODEL_DEFAULTS = {"haze": haze.DEFAULTS, "water": water.DEFAULTS, "glass": glass.DEFAULTS, "plastic": plastic.DEFAULTS}


def true_transmission(medium, depth_map, settings):
    """The transmission map the model really used (None where the effect is not a simple transmission, as for glass)."""
    if medium == "haze":
        return haze.transmission(depth_map, settings)[:, :, 0]
    if medium == "water":
        return water.transmission(depth_map, settings).mean(axis=2)  # the three colours have different t; use their average
    if medium == "plastic":
        return np.full(depth_map.shape, settings["transmission"])
    return None


def true_airlight(medium, settings):
    """The brightness of the light the medium adds (average over colours)."""
    if medium == "water":
        return float(np.mean([settings["veil_red"], settings["veil_green"], settings["veil_blue"]]))
    return float(settings["airlight"])


def work(job):
    """One job (in a worker). Returns rows for the prior check, or one row of transmission and colour numbers."""
    if job["kind"] == "prior":
        rows = []
        for path in job["paths"]:
            picture = load_image(path, max_side=SIDE)
            dark = dark_channel(picture, 15)
            rows.append({"dataset": job["dataset"], "picture": Path(path).stem, "dark_channel_mean": float(dark.mean()),
                         "share_dark_below_0.1": float((dark < 0.1).mean()), "share_dark_above_0.3": float((dark > 0.3).mean())})
        return rows
    # --- transmission and colour for one scene, medium and severity
    medium, level = job["medium"], job["level"]
    clear, depth, scene = load_clear_scenes(SCENES, SIDE, only=job["scene_index"])[0]
    settings = {**MODEL_DEFAULTS[medium], **PRESETS[medium][level]}
    degraded = np.clip(MODELS[medium].apply(clear, depth, PRESETS[medium][level]), 0, 1)
    result = dcp_restore(degraded, refine=True)
    restored, estimated_t, estimated_A = result["restored"], result["transmission"], result["airlight"]
    true_t = true_transmission(medium, depth, settings)
    row = {"medium": medium, "level": level, "scene": scene,
           "dark_clear": float(dark_channel(clear).mean()), "dark_degraded": float(dark_channel(degraded).mean()),
           "dark_restored": float(dark_channel(restored).mean()),
           "cast_clear": colour_cast(clear), "cast_degraded": colour_cast(degraded), "cast_restored": colour_cast(restored),
           "airlight_true": true_airlight(medium, settings), "airlight_estimated": float(np.mean(estimated_A)),
           "psnr_none": full_reference_scores(degraded, clear)["psnr"], "psnr_dcp": full_reference_scores(restored, clear)["psnr"]}
    if true_t is not None:
        row["t_mae"] = float(np.abs(estimated_t - true_t).mean())
        row["t_corr"] = float(np.corrcoef(estimated_t.ravel(), true_t.ravel())[0, 1]) if true_t.std() > 1e-9 else float("nan")
    else:
        row["t_mae"], row["t_corr"] = float("nan"), float("nan")
    return [row]


def strip(panels, path, size=200):
    """Save a row of labelled pictures (float RGB or 2-D maps) as one image."""
    tiles = []
    for caption, picture in panels:
        picture = np.clip(picture, 0, 1)
        if picture.ndim == 2:
            picture = np.stack([picture] * 3, axis=-1)
        tile = Image.fromarray(np.round(picture * 255).astype(np.uint8))
        tile.thumbnail((size, size))
        framed = Image.new("RGB", (tile.width, tile.height + 16), (30, 30, 30))
        framed.paste(tile, (0, 16))
        ImageDraw.Draw(framed).text((3, 2), caption[:34], fill=(220, 220, 220))
        tiles.append(framed)
    canvas = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), (30, 30, 30))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def prior_jobs():
    """Question 1 jobs: the clear pictures of every dataset, 20 evenly spaced from each."""
    def pick(paths):
        return [str(paths[i]) for i in np.linspace(0, len(paths) - 1, min(20, len(paths))).round().astype(int)]
    with open(CURATED / "manifest_haze.csv") as handle:
        haze_rows = list(csv.DictReader(handle))
    datasets = {source: pick([CURATED / r["reference_path"] for r in haze_rows if r["source"] == source])
                for source in ("O-HAZE", "I-HAZE", "Dense-Haze", "NH-Haze")}
    with open(CURATED / "manifest_depth.csv") as handle:
        datasets["NYU indoor (synthetic scenes)"] = pick([CURATED / r["degraded_path"] for r in csv.DictReader(handle)])
    with open(CURATED / "manifest_glass.csv") as handle:
        datasets["raindrop clean pane"] = pick([CURATED / r["reference_path"] for r in csv.DictReader(handle)])
    with open(CURATED / "manifest_water.csv") as handle:
        datasets["UIEB underwater (not clear)"] = pick([CURATED / r["degraded_path"] for r in csv.DictReader(handle)])
    return [{"key": f"prior|{name}", "kind": "prior", "dataset": name, "paths": paths} for name, paths in datasets.items()]


def run(workers=None, resume=True):
    jobs = prior_jobs() + [{"key": f"t|{m}|{lv}|{i}", "kind": "transmission", "medium": m, "level": lv, "scene_index": i}
                           for m in MEDIA for lv in LEVELS for i in range(SCENES)]
    rows = run_stage("failure_analysis", fingerprint(SIDE, SCENES, [j["key"] for j in jobs]), jobs, work, workers=workers,
                     memory_per_worker_gb=0.8, resume=resume)
    prior = [r for r in rows if "dataset" in r]
    trans = [r for r in rows if "medium" in r]
    tables = PROJECT / "results" / "tables"
    figures = PROJECT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    for name, data in (("failure_prior_check", prior), ("failure_transmission", trans)):
        with open(tables / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    # --- figure 1: does the dark-channel prior hold? mean dark channel of the CLEAR pictures of each dataset
    names = list(dict.fromkeys(r["dataset"] for r in prior))
    means = [np.mean([r["dark_channel_mean"] for r in prior if r["dataset"] == n]) for n in names]
    spreads = [np.std([r["dark_channel_mean"] for r in prior if r["dataset"] == n]) for n in names]
    below = [np.mean([r["share_dark_below_0.1"] for r in prior if r["dataset"] == n]) for n in names]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))
    left.barh(range(len(names)), means, xerr=spreads, color=["#3d6299" if "haze" in n.lower() or "HAZE" in n else "#888888" for n in names])
    left.axvline(0.1, color="#b05d0c", linestyle="--")
    left.set_yticks(range(len(names)))
    left.set_yticklabels(names, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("mean dark channel of the clear pictures (the prior says: near 0)")
    left.set_title("How strongly the dark-channel prior holds (dashed: 0.1)")
    right.barh(range(len(names)), [100 * b for b in below], color="#3d6299")
    right.set_yticks(range(len(names)))
    right.set_yticklabels([])
    right.invert_yaxis()
    right.set_xlabel("% of pixels whose dark channel is below 0.1")
    right.set_title("Share of the picture where the prior holds")
    figure.tight_layout()
    figure.savefig(figures / "failure_prior_check.png", dpi=105)
    plt.close(figure)

    # --- figures 2 to 4: per medium and severity, averaged over scenes
    def mean_of(key, medium, level):
        return float(np.nanmean([r[key] for r in trans if r["medium"] == medium and r["level"] == level]))
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for axis, medium in zip(axes, MEDIA):
        x = np.arange(len(LEVELS))
        for offset, (key, label, colour) in enumerate((("dark_clear", "clear", "#2e7d4f"), ("dark_degraded", "degraded", "#b05d0c"), ("dark_restored", "after DCP", "#3d6299"))):
            axis.bar(x + (offset - 1) * 0.27, [mean_of(key, medium, lv) for lv in LEVELS], 0.27, label=label, color=colour)
        axis.set_xticks(x)
        axis.set_xticklabels(LEVELS)
        axis.set_title(medium)
    axes[0].set_ylabel("mean dark channel")
    axes[0].legend(fontsize=8)
    figure.suptitle("Dark channel before and after DCP: a good restoration brings it back down to the clear picture's value")
    figure.tight_layout()
    figure.savefig(figures / "failure_dark_channel.png", dpi=105)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for medium, colour in zip(("haze", "water", "plastic"), ("#3d6299", "#2e7d4f", "#b05d0c")):
        axes[0].plot(LEVELS, [mean_of("t_mae", medium, lv) for lv in LEVELS], "o-", label=medium, color=colour)
        axes[1].plot(LEVELS, [mean_of("t_corr", medium, lv) for lv in LEVELS], "o-", label=medium, color=colour)
    for medium, colour in zip(MEDIA, ("#3d6299", "#2e7d4f", "#888888", "#b05d0c")):
        axes[2].plot(LEVELS, [mean_of("airlight_estimated", medium, lv) - mean_of("airlight_true", medium, lv) for lv in LEVELS], "o-", label=medium, color=colour)
    axes[0].set_title("transmission error (mean absolute)")
    axes[1].set_title("correlation with the true transmission")
    axes[2].set_title("airlight estimate minus true airlight")
    axes[2].axhline(0, color="black", linewidth=0.8)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("How well DCP recovers what really happened (synthetic pictures, truth known)")
    figure.tight_layout()
    figure.savefig(figures / "failure_transmission.png", dpi=105)
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for axis, medium in zip(axes, MEDIA):
        x = np.arange(len(LEVELS))
        for offset, (key, label, colour) in enumerate((("cast_clear", "clear", "#2e7d4f"), ("cast_degraded", "degraded", "#b05d0c"), ("cast_restored", "after DCP", "#3d6299"))):
            axis.bar(x + (offset - 1) * 0.27, [mean_of(key, medium, lv) for lv in LEVELS], 0.27, label=label, color=colour)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(LEVELS)
        axis.set_title(medium)
    axes[0].set_ylabel("colour cast, log(R / mean(G, B))")
    axes[0].legend(fontsize=8)
    figure.suptitle("Colour cast before and after DCP (negative = blue-green, 0 = balanced)")
    figure.tight_layout()
    figure.savefig(figures / "failure_cast.png", dpi=105)
    plt.close(figure)

    # --- figures 5 and 6: the best and worst cases of DCP, with true and estimated transmission (synthetic) and on real photographs
    haze_rows = [r for r in trans if r["medium"] == "haze" and r["level"] == "medium"]
    haze_rows.sort(key=lambda r: r["psnr_dcp"] - r["psnr_none"])
    scenes = load_clear_scenes(SCENES, SIDE)
    names_by_id = {s[2]: s for s in scenes}
    panels = []
    for tag, picks in (("worst", haze_rows[:3]), ("best", haze_rows[-3:])):
        for r in picks:
            clear, depth, scene = names_by_id[r["scene"]]
            settings = {**haze.DEFAULTS, **PRESETS["haze"]["medium"]}
            degraded = np.clip(haze.apply(clear, depth, PRESETS["haze"]["medium"]), 0, 1)
            result = dcp_restore(degraded, refine=True)
            panels += [(f"{tag} {scene}: hazy", degraded), (f"DCP {r['psnr_dcp']:.1f} dB (none {r['psnr_none']:.1f})", result["restored"]),
                       ("estimated t", result["transmission"]), ("true t", haze.transmission(depth, settings)[:, :, 0]), ("clear", clear)]
    strip_rows = [panels[i:i + 5] for i in range(0, len(panels), 5)]
    for index, tiles in enumerate(strip_rows):
        strip(tiles, figures / f"_failure_row_{index}.jpg", size=170)
    rows_images = [Image.open(figures / f"_failure_row_{i}.jpg") for i in range(len(strip_rows))]
    sheet = Image.new("RGB", (max(i.width for i in rows_images), sum(i.height for i in rows_images)), (30, 30, 30))
    y = 0
    for image in rows_images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(figures / "failure_cases_synthetic.jpg", quality=90)
    for index in range(len(strip_rows)):
        (figures / f"_failure_row_{index}.jpg").unlink()

    # real photographs: the cases where DCP helped least or hurt (compared with doing nothing)
    with open(tables / "haze_real.csv") as handle:
        real = list(csv.DictReader(handle))
    by_picture = {}
    for r in real:
        by_picture.setdefault((r["level"], r["id"]), {})[r["method"]] = float(r["psnr"])
    gains = sorted(by_picture, key=lambda k: by_picture[k]["dcp_guided"] - by_picture[k]["none"])
    pairs = {(p["level"], p["id"]): p for p in list_pairs("haze", "real", 100)}
    real_panels = []
    for key in gains[:3] + gains[-3:]:
        pair = pairs[key]
        degraded, reference = load_image(pair["degraded"], 256), load_image(pair["reference"], 256)
        restored = dcp_restore(degraded, refine=True)
        real_panels.append([(f"{key[1]}: hazy", degraded), (f"DCP {by_picture[key]['dcp_guided']:.1f} dB (none {by_picture[key]['none']:.1f})", restored["restored"]),
                            ("estimated t", restored["transmission"]), ("clear reference", reference)])
    for index, tiles in enumerate(real_panels):
        strip(tiles, figures / f"_real_row_{index}.jpg", size=170)
    images = [Image.open(figures / f"_real_row_{i}.jpg") for i in range(len(real_panels))]
    sheet = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), (30, 30, 30))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(figures / "failure_cases_real.jpg", quality=90)
    for index in range(len(real_panels)):
        (figures / f"_real_row_{index}.jpg").unlink()

    # --- print the headline numbers
    print("\nDark channel of CLEAR pictures (prior says near 0):")
    for name, mean_value, share in zip(names, means, below):
        print(f"  {name:32} mean {mean_value:.3f}   {100 * share:4.0f}% of pixels below 0.1")
    print("\nDCP transmission accuracy (medium severity):")
    for medium in ("haze", "water", "plastic"):
        print(f"  {medium:8} error {mean_of('t_mae', medium, 'medium'):.3f}   correlation {mean_of('t_corr', medium, 'medium'):+.2f}   "
              f"airlight estimated-true {mean_of('airlight_estimated', medium, 'medium') - mean_of('airlight_true', medium, 'medium'):+.2f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
