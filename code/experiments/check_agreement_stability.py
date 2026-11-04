"""STAGE 5b - how stable are the 'synthetic ranks the methods like real' numbers? Bootstrap intervals, and the same comparison with 90 scenes.

Earlier agreement numbers came from small samples (12 synthetic scenes, 58 to 185 real pairs). For each comparison this script
  1. BOOTSTRAPS both sides: resamples the synthetic scenes and the real pairs with replacement 2000 times, recomputes the rank correlation
     (Spearman over the methods of the mean PSNR), and reports the 5th and 95th percentiles;
  2. RE-MAKES the synthetic side with 90 scenes (glass, original water, milky water) and recomputes the correlation.
Comparisons: glass against real raindrops; water against the real tank pairs (original water, milky water); same-scene twins (haze, glass; from scene_matched.csv).
Output:  results/tables/agreement_stability.csv   results/figures/agreement_stability.png
Run from the project root:  .venv/bin/python code/experiments/check_agreement_stability.py
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DIP_PRESETS", "calibrated")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade import glass, milky, water  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, LEVELS, PRESETS  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE, SCENES, DRAWS = 256, 90, 2000


def read(name):
    with open(TABLES / name) as handle:
        return list(csv.DictReader(handle))


def by_picture(rows, picture_key, psnr_key="psnr"):
    """{picture: {method: mean PSNR over severities}}"""
    groups = {}
    for r in rows:
        groups.setdefault(r[picture_key], {}).setdefault(r["method"], []).append(float(r[psnr_key]))
    return {p: {m: float(np.mean(v)) for m, v in d.items()} for p, d in groups.items()}


def agreement(side_a, side_b, methods, pictures_a=None, pictures_b=None):
    mean = lambda side, pictures: [np.mean([side[p][m] for p in pictures]) for m in methods]  # noqa: E731
    return float(spearmanr(mean(side_a, pictures_a or list(side_a)), mean(side_b, pictures_b or list(side_b)))[0])


def bootstrap(side_a, side_b, methods, rng):
    pa, pb = list(side_a), list(side_b)
    draws = [agreement(side_a, side_b, methods, list(rng.choice(pa, len(pa))), list(rng.choice(pb, len(pb)))) for _ in range(DRAWS)]
    return float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def work(job):
    clear, depth, scene = load_clear_scenes(SCENES, SIDE, only=job["index"])[0]
    rows = []
    for level in LEVELS:
        if job["kind"] == "glass":
            degraded = glass.apply(clear, None, CALIBRATED_PRESETS["glass"][level])
        elif job["kind"] == "water":
            degraded = water.apply(clear, depth, PRESETS["water"][level])
        else:
            degraded = milky.apply(clear, depth, {**job["milky"][level]})
        degraded = np.clip(degraded, 0, 1)
        for name in job["methods"]:
            rows.append({"kind": job["kind"], "picture": scene, "level": level, "method": name,
                         "psnr": full_reference_scores(np.clip(build_method({"name": name})(degraded), 0, 1), clear)["psnr"]})
    return rows


def run(workers=None, resume=True):
    rng = np.random.default_rng(0)
    methods = {"glass": [s["name"] for s in json.loads((PROJECT / "configs" / "glass.json").read_text())["methods"]],
               "water": [s["name"] for s in json.loads((PROJECT / "configs" / "water.json").read_text())["methods"]]}
    milky_levels = json.loads((PROJECT / "configs" / "milky_presets.json").read_text())["levels"]
    big = {}
    for kind, method_list, milky_params in (("glass", methods["glass"], None), ("water", methods["water"], None), ("milky", methods["water"], milky_levels)):
        jobs = [{"key": f"{kind}|{i}", "kind": kind, "index": i, "methods": method_list, "milky": milky_params} for i in range(SCENES)]
        rows = run_stage(f"stability_{kind}", fingerprint(kind, method_list, SCENES, SIDE, milky_params), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
        big[kind] = by_picture(rows, "picture")
    real_glass, real_tank = by_picture(read("glass_real.csv"), "id"), by_picture(read("water_tub.csv"), "id")
    small = {"glass": by_picture(read("glass_matched.csv"), "id"), "water": by_picture(read("water_matched.csv"), "id"),
             "milky": by_picture([r for r in read("milky_test.csv") if r["cloud"] == "milk"], "scene")}
    results = []
    for label, kind, real, mlist in (("glass against real raindrops", "glass", real_glass, methods["glass"]), ("water (original) against real tank", "water", real_tank, methods["water"]),
                                      ("water (milky) against real tank", "milky", real_tank, methods["water"])):
        point = agreement(small[kind], real, mlist)
        low, high = bootstrap(small[kind], real, mlist, rng)
        point90 = agreement(big[kind], real, mlist)
        low90, high90 = bootstrap(big[kind], real, mlist, rng)
        results.append({"comparison": label, "synthetic_scenes_before": len(small[kind]), "real_pairs": len(real), "point_before": round(point, 2), "ci_before": f"{low:+.2f} to {high:+.2f}",
                        "synthetic_scenes_now": len(big[kind]), "point_now": round(point90, 2), "ci_now": f"{low90:+.2f} to {high90:+.2f}"})
    # same-scene twins: bootstrap over the pictures (real and synthetic twin share the picture)
    rows = read("scene_matched.csv")
    for medium in ("haze", "glass"):
        own = [r for r in rows if r["medium"] == medium]
        real = by_picture(own, "picture", "psnr_real")
        twin = by_picture(own, "picture", "psnr_synthetic")
        mlist = sorted({r["method"] for r in own})
        pics = list(real)
        draws = [agreement(real, twin, mlist, list(p), list(p)) for p in (rng.choice(pics, len(pics)) for _ in range(DRAWS))]
        results.append({"comparison": f"same-scene twins, {medium}", "synthetic_scenes_before": len(pics), "real_pairs": len(pics), "point_before": round(agreement(real, twin, mlist), 2),
                        "ci_before": f"{np.percentile(draws, 5):+.2f} to {np.percentile(draws, 95):+.2f}", "synthetic_scenes_now": "", "point_now": "", "ci_now": ""})
    with open(TABLES / "agreement_stability.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nRank agreement (Spearman over methods), with 90% bootstrap interval")
    for r in results:
        print(f"  {r['comparison']:36} before: {r['point_before']:+.2f} ({r['ci_before']}; {r['synthetic_scenes_before']} synthetic scenes, {r['real_pairs']} real pairs)"
              + (f"   with {r['synthetic_scenes_now']} scenes: {r['point_now']:+.2f} ({r['ci_now']})" if r["point_now"] != "" else ""))
    figure, axis = plt.subplots(figsize=(8, 4))
    for k, r in enumerate(results):
        for key, color, off in (("before", "#d98c2f", -0.12), ("now", "#3d6299", 0.12)):
            if r[f"point_{key}"] == "":
                continue
            low, high = [float(x) for x in r[f"ci_{key}"].replace("+", "").split(" to ")]
            axis.errorbar(r[f"point_{key}"], k + off, xerr=[[r[f"point_{key}"] - low], [high - r[f"point_{key}"]]], fmt="o", color=color, capsize=3,
                          label=("original sample" if key == "before" else "90 synthetic scenes") if k == 0 else None)
    axis.set_yticks(range(len(results)), [r["comparison"] for r in results], fontsize=8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("rank agreement (Spearman), point and 90% bootstrap interval")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="x")
    figure.tight_layout()
    figure.savefig(FIGURES / "agreement_stability.png", dpi=110)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
