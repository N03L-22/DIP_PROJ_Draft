"""Make the figures for the report from the saved results (no experiments are re-run).

  report_ba_<medium>.jpg        before / after strips on matched synthetic pictures: the base method, the base method followed by
                                a post-processing step, the best sequence found, and the clear reference
  report_ba_real_haze.jpg       the same idea on real haze photographs (one from each of the four datasets)
  report_ba_real_glass.jpg      real raindrop pictures: restoration methods that do not help
  report_methods_<medium>.png   mean PSNR of every method on synthetic pictures and (where it exists) real photographs
  report_gain_over_none.png     gain over doing nothing, all media
  report_synthetic_vs_real.png  each method's synthetic score against its real score (haze and glass)
  report_postprocessing.png     how much a post-processing step adds after each base method (from the step-order search)

Run from the project root, after the experiments:  .venv/bin/python code/experiments/make_report_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_matched_pairs  # noqa: E402
from common.imageio import load_image  # noqa: E402
from experiments.run_experiments import list_pairs  # noqa: E402
from experiments.run_pipelines import SEPARATOR, average_by_pipeline  # noqa: E402
from restore.methods import build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
MEDIA = ("haze", "water", "glass", "plastic")
BLUE, ORANGE, GREEN, GREY = "#3d6299", "#b05d0c", "#2e7d4f", "#888888"

# the strips: (caption, steps). "best" is replaced by the best multi-step sequence of the step-order search for that medium.
STRIPS = {
    "haze":    [("dcp_guided", ["dcp_guided"]), ("dcp_guided > clahe", ["dcp_guided", "clahe"]), ("dcp_guided > unsharp", ["dcp_guided", "unsharp"]),
                ("tarel", ["tarel"]), ("best", None)],
    "water":   [("udcp", ["udcp"]), ("von_kries", ["von_kries"]), ("red_comp > von_kries", ["red_comp", "von_kries"]),
                ("red_comp > von_kries > clahe", ["red_comp", "von_kries", "clahe"]), ("best", None)],
    "glass":   [("dcp_near_global", ["dcp_near_global"]), ("stretch", ["stretch"]), ("stretch > unsharp", ["stretch", "unsharp"]),
                ("homomorphic", ["homomorphic"]), ("best", None)],
    "plastic": [("wb_dcp", ["wb_dcp"]), ("wiener", ["wiener"]), ("wiener > unsharp", ["wiener", "unsharp"]),
                ("stretch", ["stretch"]), ("best", None)],
}


def read_table(name):
    with open(TABLES / f"{name}.csv") as handle:
        return list(csv.DictReader(handle))


def best_sequence(medium):
    """The best multi-step sequence of the step-order search for one medium (chosen on the search pictures)."""
    rows = read_table(f"pipelines_{medium}_search")
    for row in rows:
        row["psnr"], row["ssim"], row["ciede2000"], row["uciqe"], row["length"] = (float(row[k]) for k in ("psnr", "ssim", "ciede2000", "uciqe", "length"))
    averaged = average_by_pipeline(rows)
    name = max((n for n in averaged if averaged[n]["length"] >= 2), key=lambda n: averaged[n]["psnr"])
    return name.split(SEPARATOR)


def save_strips(rows_of_panels, path, size=210):
    """Save several rows of (caption, picture) tiles as one picture. Each row is one scene."""
    sheet_rows = []
    for panels in rows_of_panels:
        tiles = []
        for caption, picture in panels:
            tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            tile.thumbnail((size, size))
            framed = Image.new("RGB", (tile.width, tile.height + 18), (30, 30, 30))
            framed.paste(tile, (0, 18))
            ImageDraw.Draw(framed).text((3, 3), caption[:36], fill=(225, 225, 225))
            tiles.append(framed)
        sheet_rows.append(tiles)
    sheet = Image.new("RGB", (max(sum(t.width for t in r) for r in sheet_rows), sum(max(t.height for t in r) for r in sheet_rows)), (30, 30, 30))
    y = 0
    for tiles in sheet_rows:
        x = 0
        for tile in tiles:
            sheet.paste(tile, (x, y))
            x += tile.width
        y += max(t.height for t in tiles)
    sheet.save(path, quality=92)


def before_after_synthetic(medium):
    """Three held-out scenes at medium severity: degraded, four restorations (with and without post-processing), best sequence, clear."""
    steps_list = [(caption, steps if steps else best_sequence(medium)) for caption, steps in STRIPS[medium]]
    steps_list = [((" > ".join(steps) if caption == "best" else caption), steps) for caption, steps in steps_list]
    rows = []
    for degraded, clear, scene in load_matched_pairs(medium, "medium", 3, 320, start=6):
        panels = [(f"{medium} (medium): {scene}", degraded)]
        for caption, steps in steps_list:
            panels.append((("best: " if caption == " > ".join(best_sequence(medium)) else "") + caption, build_sequence(steps)(degraded)))
        panels.append(("clear reference", clear))
        rows.append(panels)
    save_strips(rows, FIGURES / f"report_ba_{medium}.jpg")


def before_after_real_haze():
    """One real haze photograph from each of the four datasets: degraded, DCP, DCP + CLAHE, fusion with DCP, clear."""
    pairs = list_pairs("haze", "real", 100)
    rows = []
    for source in ("O-HAZE", "I-HAZE", "Dense-Haze", "NH-Haze"):
        pair = next(p for p in pairs if p["id"].startswith(source) and p["level"] == "medium")
        degraded, reference = load_image(pair["degraded"], 320), load_image(pair["reference"], 320)
        rows.append([(pair["id"] + " hazy", degraded)] + [(" > ".join(s), build_sequence(s)(degraded)) for s in (["dcp_guided"], ["dcp_guided", "clahe"], ["fusion_dcp"])] + [("clear reference", reference)])
    save_strips(rows, FIGURES / "report_ba_real_haze.jpg")


def before_after_real_glass():
    """Real raindrop pictures at the three severities: methods that look reasonable on synthetic glass do not help here."""
    pairs = list_pairs("glass", "real", 100)
    rows = []
    for level in ("mild", "medium", "heavy"):
        pair = next(p for p in pairs if p["level"] == level)
        degraded, reference = load_image(pair["degraded"], 320), load_image(pair["reference"], 320)
        rows.append([(f"rain ({level})", degraded)] + [(s[0], build_sequence(s)(degraded)) for s in (["dcp_near_global"], ["stretch"], ["freq_laplacian"])] + [("clean pane", reference)])
    save_strips(rows, FIGURES / "report_ba_real_glass.jpg")


def method_means(medium, source, score="psnr"):
    """Mean score of each method over all pictures of one table, or {} if the table does not exist (no real data)."""
    path = TABLES / f"{medium}_{source}.csv"
    if not path.exists():
        return {}
    rows = read_table(f"{medium}_{source}")
    methods = dict.fromkeys(r["method"] for r in rows)
    if rows[0][score] == "":
        return {}
    return {m: float(np.mean([float(r[score]) for r in rows if r["method"] == m])) for m in methods}


def methods_charts():
    """One horizontal bar chart per medium: every method's mean PSNR on synthetic pictures and on real ones, with 'none' marked."""
    for medium in MEDIA:
        synthetic, real = method_means(medium, "matched"), method_means(medium, "real")
        names = sorted(synthetic, key=lambda n: -synthetic[n])
        figure, axis = plt.subplots(figsize=(8.4, 0.34 * len(names) + 1.4))
        y = np.arange(len(names))
        height = 0.38 if real else 0.6
        axis.barh(y - (height / 2 if real else 0), [synthetic[n] for n in names], height, color=BLUE, label="synthetic pairs (36 pictures)")
        if real:
            axis.barh(y + height / 2, [real.get(n, 0) for n in names], height, color=ORANGE, label="real photographs (36 pictures)")
        axis.axvline(synthetic["none"], color=BLUE, linestyle="--", linewidth=1)
        if real:
            axis.axvline(real["none"], color=ORANGE, linestyle="--", linewidth=1)
        axis.set_yticks(y)
        axis.set_yticklabels(names, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("mean PSNR (dB); dashed lines: no restoration")
        axis.set_title(f"{medium}: every method, all severities")
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        figure.savefig(FIGURES / f"report_methods_{medium}.png", dpi=110)
        plt.close(figure)


def gain_chart():
    """Gain over doing nothing for the best six methods of each medium (synthetic), with the DCP family marked."""
    dcp_family = {"dcp", "dcp_guided", "udcp", "wb_dcp", "dcp_near_global", "dcp_local_airlight", "dcp_clahe", "fusion_dcp"}
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for axis, medium in zip(axes, MEDIA):
        synthetic = method_means(medium, "matched")
        gains = {m: synthetic[m] - synthetic["none"] for m in synthetic if m != "none"}
        top = sorted(gains, key=lambda n: -gains[n])[:6]
        low = sorted(gains, key=lambda n: gains[n])[:2]
        names = top + [n for n in low if n not in top]
        axis.barh(range(len(names)), [gains[n] for n in names], color=[ORANGE if n in dcp_family else BLUE for n in names])
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(medium)
        axis.set_xlabel("PSNR gain over no restoration (dB)")
    figure.suptitle("Best and worst methods per medium on synthetic pictures (orange = DCP family)")
    figure.tight_layout()
    figure.savefig(FIGURES / "report_gain_over_none.png", dpi=110)
    plt.close(figure)


def synthetic_vs_real_chart():
    """For haze and glass: each method as a point, synthetic score across, real score up. Points on the dashed line would mean the two agree."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, medium in zip(axes, ("haze", "glass")):
        synthetic, real = method_means(medium, "matched"), method_means(medium, "real")
        names = [n for n in synthetic if n in real]
        x, y = [synthetic[n] for n in names], [real[n] for n in names]
        axis.scatter(x, y, color=BLUE)
        for n, a, b in zip(names, x, y):
            axis.annotate(n, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
        low, high = min(x + y) - 1, max(x + y) + 1
        axis.plot([low, high], [low, high], "--", color=GREY)
        axis.set_xlabel("PSNR on synthetic pictures (dB)")
        axis.set_ylabel("PSNR on real photographs (dB)")
        axis.set_title(f"{medium}: do synthetic scores predict real ones?")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "report_synthetic_vs_real.png", dpi=110)
    plt.close(figure)


def postprocessing_chart():
    """Heat map of what a post-processing step adds after a base method: PSNR(base > post) minus PSNR(base), from the step-order search."""
    pools = json.loads((PROJECT / "configs" / "pipelines.json").read_text())
    posts_wanted = ("clahe", "unsharp", "hist_eq", "eq_hsi", "gamma_auto", "stretch")
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for axis, medium in zip(axes, MEDIA):
        rows = read_table(f"pipelines_{medium}_search")
        for row in rows:
            for key in ("psnr", "ssim", "ciede2000", "uciqe", "length"):
                row[key] = float(row[key])
        averaged = average_by_pipeline(rows)
        pool = pools[medium]["pool"]
        posts = [p for p in posts_wanted if p in pool]
        bases = [b for b in pool if b not in posts_wanted]
        grid = np.array([[averaged[SEPARATOR.join([b, p])]["psnr"] - averaged[b]["psnr"] for p in posts] for b in bases])
        limit = np.abs(grid).max()
        image = axis.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(posts)))
        axis.set_xticklabels(posts, rotation=40, ha="right", fontsize=8)
        axis.set_yticks(range(len(bases)))
        axis.set_yticklabels(bases, fontsize=8)
        for i in range(len(bases)):
            for j in range(len(posts)):
                axis.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        axis.set_title(medium)
    figure.suptitle("What a post-processing step adds after a base method (change in PSNR, dB; blue = helps, red = hurts)")
    figure.tight_layout()
    figure.savefig(FIGURES / "report_postprocessing.png", dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    for medium in MEDIA:
        before_after_synthetic(medium)
    before_after_real_haze()
    before_after_real_glass()
    methods_charts()
    gain_chart()
    synthetic_vs_real_chart()
    postprocessing_chart()
    print("wrote the report figures to", FIGURES.relative_to(PROJECT))
