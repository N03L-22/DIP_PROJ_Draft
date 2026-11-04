"""STAGE 4e - global against local application of the same method, for haze, water and glass.

For each base method we compare, on the same pictures:
  global     the method on the whole picture
  tiles2/4/8 the method run tile by tile on a 2x2, 4x4, 8x8 grid, blended without seams
  dark3      three regions found FROM THE PICTURE (low / medium / high dark channel = thin / medium / thick haze), one run per region
  sharp2     (glass) two regions from local sharpness (blurred areas, i.e. raindrops, against sharp areas)
  depth3     (synthetic haze and water only) three regions from the TRUE depth: an upper bound for what a perfect region finder could give
Pictures: synthetic = clear NYU scenes degraded with the CALIBRATED presets (three severities, so depth is known), real = haze pairs
(O-HAZE, I-HAZE, Dense-Haze, NH-Haze) and raindrop pairs. Water has no real pairs. One job = one picture (all methods and variants).

Output:  results/tables/local_vs_global_<medium>.csv  local_vs_global_summary.csv   results/figures/local_vs_global.png  local_examples_<medium>.jpg
Run from the project root:  .venv/bin/python code/experiments/run_local_vs_global.py
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
from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, LEVELS  # noqa: E402
from degrade.stack import MODELS  # noqa: E402
from experiments.validate_models import read_pairs  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore import local  # noqa: E402
from restore.methods import REGISTRY, build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE = 256
SCENES = 8       # synthetic: scenes per severity
REAL_PAIRS = 40  # real haze pairs used (spread through the list)

# the methods tried locally (a method must be one function of one picture); chosen to span global-statistics and spatial methods
METHODS = {
    "haze": ["stretch", "hist_eq", "gamma_auto", "gray_world", "unsharp", "dcp_guided"],
    "water": ["stretch", "hist_eq", "gamma_auto", "gray_world", "von_kries", "red_comp", "udcp"],
    "glass": ["stretch", "hist_eq", "gamma_auto", "unsharp", "flat_field", "local_stats"],
}
VARIANTS = ("global", "tiles2", "tiles4", "tiles8", "dark3", "sharp2", "depth3")


def apply_variant(variant, function, image, depth):
    """Run one method function on a degraded picture in the given way. Returns None when the variant does not apply."""
    if variant == "global":
        return np.clip(function(image), 0, 1)
    if variant.startswith("tiles"):
        return local.tiled(function, image, int(variant[5:]))
    if variant == "dark3":
        return local.segmented(function, image, local.damage_regions(image, 3))
    if variant == "sharp2":
        return local.segmented(function, image, local.sharpness_regions(image, 2))
    if variant == "depth3":
        return None if depth is None else local.segmented(function, image, local.depth_regions(depth, 3))
    raise ValueError(variant)


def work(job):
    """One job (in a worker): every method, in every variant, on one picture, scored against the clear picture."""
    if job["world"] == "synthetic":
        clear, depth, _ = load_clear_scenes(job["scenes"], SIDE, only=job["scene_index"])[0]
        degraded = np.clip(MODELS[job["medium"]].apply(clear, depth, CALIBRATED_PRESETS[job["medium"]][job["level"]]), 0, 1)
        depth = depth if job["medium"] in ("haze", "water") else None
    else:
        clear, degraded, depth = load_image(job["reference"], SIDE), load_image(job["degraded"], SIDE), None
    rows = []
    for name in job["methods"]:
        function = build_method({"name": name})
        for variant in job.get("variants", VARIANTS):
            if variant == "sharp2" and job["medium"] != "glass":
                continue
            restored = apply_variant(variant, function, degraded, depth)
            if restored is None:
                continue
            scores = full_reference_scores(restored, clear)
            rows.append({"world": job["world"], "medium": job["medium"], "level": job["level"], "picture": job["picture"], "method": name,
                         "variant": variant, "psnr": scores["psnr"], "ssim": scores["ssim"]})
    baseline = full_reference_scores(degraded, clear)  # doing nothing, for reference
    rows.append({"world": job["world"], "medium": job["medium"], "level": job["level"], "picture": job["picture"], "method": "none",
                 "variant": "global", "psnr": baseline["psnr"], "ssim": baseline["ssim"]})
    return rows


def build_jobs(variants=VARIANTS, tag=""):
    """One job per picture. `variants` and `tag` let the tile-size sweep reuse this with its own variant list and checkpoint keys."""
    jobs = []
    for medium, names in METHODS.items():
        assert all(n in REGISTRY for n in names)
        for level in LEVELS:
            for index in range(SCENES):
                jobs.append({"key": f"{tag}syn|{medium}|{level}|{index}", "variants": variants, "world": "synthetic", "medium": medium, "level": level, "picture": f"scene{index}",
                             "scene_index": index, "scenes": SCENES, "methods": names})
    for medium in ("haze", "glass"):
        pairs = read_pairs(medium)
        for pair in [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(REAL_PAIRS, len(pairs))).round().astype(int)]:
            jobs.append({"key": f"{tag}real|{medium}|{pair['id']}", "variants": variants, "world": "real", "medium": medium, "level": "real", "picture": pair["id"],
                         "methods": METHODS[medium], "degraded": pair["degraded"], "reference": pair["reference"]})
    return jobs


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), restval="")
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows, variants=VARIANTS):
    """Per world, medium, method and variant: mean PSNR and the gain over the global form of the same method."""
    summary = []
    for world in ("synthetic", "real"):
        for medium in METHODS:
            own = [r for r in rows if r["world"] == world and r["medium"] == medium]
            if not own:
                continue
            for name in METHODS[medium]:
                means = {}
                for variant in variants:
                    values = [r["psnr"] for r in own if r["method"] == name and r["variant"] == variant]
                    if values:
                        means[variant] = float(np.mean(values))
                for variant, value in means.items():
                    # paired gain: per picture, local minus global, so scene difficulty cancels
                    gains = [r["psnr"] - next(g["psnr"] for g in own if g["picture"] == r["picture"] and g["level"] == r["level"] and g["method"] == name and g["variant"] == "global")
                             for r in own if r["method"] == name and r["variant"] == variant]
                    wins = float(np.mean(np.array(gains) > 0)) if variant != "global" else float("nan")
                    summary.append({"world": world, "medium": medium, "method": name, "variant": variant, "psnr": round(value, 3),
                                    "gain_over_global": round(float(np.mean(gains)), 3), "share_of_pictures_better": round(wins, 3) if wins == wins else "",
                                    "pictures": len(gains)})
            none = [r["psnr"] for r in own if r["method"] == "none"]
            summary.append({"world": world, "medium": medium, "method": "none", "variant": "global", "psnr": round(float(np.mean(none)), 3),
                            "gain_over_global": "", "share_of_pictures_better": "", "pictures": len(none)})
    return summary


def make_figure(summary):
    """Heat map per medium and world: rows are methods, columns are local variants, colour is the PSNR gain over the global form."""
    cells = [(m, w) for m in METHODS for w in ("synthetic", "real") if any(r["world"] == w and r["medium"] == m for r in summary)]
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for axis in axes.ravel():
        axis.axis("off")
    for medium, world in cells:
        axis = axes[0 if world == "synthetic" else 1][list(METHODS).index(medium)]
        axis.axis("on")
        columns = [v for v in VARIANTS[1:] if any(r["variant"] == v and r["world"] == world and r["medium"] == medium for r in summary)]
        grid = np.array([[next(r["gain_over_global"] for r in summary if r["world"] == world and r["medium"] == medium and r["method"] == n and r["variant"] == v) for v in columns] for n in METHODS[medium]], dtype=float)
        limit = max(1.0, np.abs(grid).max())
        axis.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(columns)), columns, rotation=30)
        axis.set_yticks(range(len(METHODS[medium])), METHODS[medium])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                axis.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        axis.set_title(f"{medium} ({world.upper()})")
    figure.suptitle("Local against global: PSNR change (dB) when a method is run locally. Blue = local better, red = local worse", fontsize=12)
    figure.tight_layout()
    figure.savefig(FIGURES / "local_vs_global.png", dpi=110)
    plt.close(figure)


def make_examples():
    """One strip per medium: degraded | global | best local variant by mean gain | clear, for a medium-severity synthetic scene."""
    for medium, method in (("haze", "hist_eq"), ("water", "von_kries"), ("glass", "stretch")):
        clear, depth, _ = load_clear_scenes(SCENES, SIDE, only=2)[0]
        degraded = np.clip(MODELS[medium].apply(clear, depth, CALIBRATED_PRESETS[medium]["medium"]), 0, 1)
        function = build_method({"name": method})
        tiles = [("degraded (SYNTHETIC)", degraded), (f"{method} global", apply_variant("global", function, degraded, None)),
                 (f"{method} tiles4", apply_variant("tiles4", function, degraded, None)), (f"{method} dark3", apply_variant("dark3", function, degraded, None)), ("clear", clear)]
        framed = []
        for caption, picture in tiles:
            tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            canvas = Image.new("RGB", (tile.width, tile.height + 18), (30, 30, 30))
            canvas.paste(tile, (0, 18))
            ImageDraw.Draw(canvas).text((3, 3), caption, fill=(225, 225, 225))
            framed.append(canvas)
        sheet = Image.new("RGB", (sum(t.width for t in framed), framed[0].height))
        x = 0
        for tile in framed:
            sheet.paste(tile, (x, 0))
            x += tile.width
        sheet.save(FIGURES / f"local_examples_{medium}.jpg", quality=92)


SWEEP_GRIDS = (2, 3, 4, 5, 6, 7, 8, 10, 12)  # tile counts per side for the tile-size sweep


def run_tile_sweep(workers=None, resume=True):
    """Which grid size is best? Same experiment with grids 2..12 only (no region variants), results to local_tile_sweep_*.csv and local_tile_sweep.png."""
    variants = ("global",) + tuple(f"tiles{g}" for g in SWEEP_GRIDS)
    jobs = build_jobs(variants, tag="sweep|")
    rows = run_stage("local_tile_sweep", fingerprint(METHODS, variants, SIDE, SCENES, REAL_PAIRS, len(jobs)), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
    save_csv(rows, TABLES / "local_tile_sweep_rows.csv")
    summary = summarise(rows, variants)
    save_csv(summary, TABLES / "local_tile_sweep_summary.csv")
    cells = [(m, w) for w in ("synthetic", "real") for m in METHODS if any(r["world"] == w and r["medium"] == m for r in summary)]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis in axes.ravel():
        axis.axis("off")
    print("\nBest grid per method (gain over the global form, PSNR dB) and the mean over methods per grid:")
    for medium, world in cells:
        axis = axes[0 if world == "synthetic" else 1][list(METHODS).index(medium)]
        axis.axis("on")
        for name in METHODS[medium]:
            own = [next(r for r in summary if r["world"] == world and r["medium"] == medium and r["method"] == name and r["variant"] == f"tiles{g}") for g in SWEEP_GRIDS]
            axis.plot(SWEEP_GRIDS, [r["gain_over_global"] for r in own], marker="o", markersize=3, label=name)
            best = max(own, key=lambda r: r["gain_over_global"])
            print(f"  {world:9} {medium:6} {name:11} best grid {best['variant'][5:]:>2}x{best['variant'][5:]:<2} {best['gain_over_global']:+.2f} dB   all: " + " ".join(f"{r['gain_over_global']:+.1f}" for r in own))
        mean = [float(np.mean([next(r["gain_over_global"] for r in summary if r["world"] == world and r["medium"] == medium and r["method"] == n and r["variant"] == f"tiles{g}") for n in METHODS[medium]])) for g in SWEEP_GRIDS]
        print(f"  {world:9} {medium:6} MEAN over methods: " + " ".join(f"{g}:{m:+.2f}" for g, m in zip(SWEEP_GRIDS, mean)))
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(SWEEP_GRIDS)
        axis.set_xlabel("tiles per side (grid n x n)")
        axis.set_ylabel("PSNR gain over global (dB)")
        axis.set_title(f"{medium} ({world.upper()})")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Tile size sweep: gain of the tiled method over the global one, by number of tiles per side", fontsize=12)
    figure.tight_layout()
    figure.savefig(FIGURES / "local_tile_sweep.png", dpi=110)
    plt.close(figure)
    return True


def run(workers=None, resume=True):
    jobs = build_jobs()
    rows = run_stage("local_vs_global", fingerprint(METHODS, VARIANTS, SIDE, SCENES, REAL_PAIRS, len(jobs)), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
    for medium in METHODS:
        save_csv([r for r in rows if r["medium"] == medium], TABLES / f"local_vs_global_{medium}.csv")
    summary = summarise(rows)
    save_csv(summary, TABLES / "local_vs_global_summary.csv")
    make_figure(summary)
    make_examples()
    print("\nMean PSNR gain of the best local variant over global, per method (share of pictures better):")
    for world in ("synthetic", "real"):
        for medium in METHODS:
            for name in METHODS[medium]:
                own = [r for r in summary if r["world"] == world and r["medium"] == medium and r["method"] == name and r["variant"] != "global"]
                if own:
                    best = max(own, key=lambda r: r["gain_over_global"])
                    print(f"  {world:9} {medium:6} {name:11} best local {best['variant']:7} {best['gain_over_global']:+.2f} dB ({best['share_of_pictures_better']:.0%} of pictures)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tile-sweep", action="store_true", help="run the tile-size sweep (grids 2 to 12) instead of the main comparison")
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    (run_tile_sweep if arguments.tile_sweep else run)(arguments.workers, resume=not arguments.fresh)
