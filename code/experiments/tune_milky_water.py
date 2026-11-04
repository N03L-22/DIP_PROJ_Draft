"""STAGE 2f - synthetic MILKY water: calibrate it to the real tank photos (TUB), then test whether it ranks the methods like the real photos do.

  1. Cloud colour: the mean colour of the most turbid TUB pictures (stages 9 and 10), where almost nothing but the cloud is left.
  2. Density: for each real turbidity band (mild / medium / heavy) pick the beta whose synthetic signature (dark shift, contrast ratio,
     brightness shift, changed share, sharpness ratio) is nearest to the mean signature of the real pairs, as in tune_models.py.
  3. Test: degrade 12 NYU scenes at the three calibrated densities, run the water methods and the haze methods, and compare the
     method ranking with the one on the real tank pairs (and with the original synthetic water).
  4. INK: the same densities but a near-black cloud. No real data exists for ink, so this is a what-if, not a calibration.
One job = one beta and scene (sweep) or one scene, level and cloud type (test); parallel and resumable.

Output:  configs/milky_presets.json   results/tables/milky_*.csv   results/figures/milky_*.png / .jpg
Run from the project root:  .venv/bin/python code/experiments/tune_milky_water.py
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
from PIL import Image, ImageDraw  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade import milky  # noqa: E402
from experiments.tune_models import METRICS, distance  # noqa: E402
from experiments.validate_models import signature  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
CURATED = PROJECT / "data" / "curated"
SIDE, SCENES, PER_BAND = 256, 12, 40
LEVELS = ("mild", "medium", "heavy")
BETAS = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.5, 5.5, 7.0]
BLURS = [0.0, 0.004, 0.008, 0.015, 0.025]  # forward-scatter blur (share of the longer side)
INK = [0.02, 0.02, 0.03]


def real_pairs():
    """The TUB pairs used (PER_BAND per band, spread through the list, the same ones run_experiments.py uses) as dicts."""
    with open(CURATED / "manifest_water_tub.csv") as handle:
        rows = list(csv.DictReader(handle))
    chosen = []
    for level in LEVELS:
        own = [r for r in rows if r["level"] == level]
        chosen += [own[i] for i in np.linspace(0, len(own) - 1, PER_BAND).round().astype(int)]
    return chosen


def real_targets():
    """Mean signature per band and the spread over all pairs; cloud colour from the stage 9 and 10 pictures."""
    sig = {level: [] for level in LEVELS}
    for r in real_pairs():
        degraded, reference = load_image(CURATED / r["degraded_path"], SIDE), load_image(CURATED / r["reference_path"], SIDE)
        s = signature(degraded, reference)
        sig[r["level"]].append([s[m] for m in METRICS])
    spread = np.std(np.vstack([np.array(v) for v in sig.values()]), axis=0)
    colours = [load_image(p, SIDE).reshape(-1, 3).mean(axis=0) for p in sorted((PROJECT / "data" / "existing" / "TUB" / "images").glob("*_Stage10.jpg")) + sorted((PROJECT / "data" / "existing" / "TUB" / "images").glob("*_Stage09.jpg"))]
    return {level: np.mean(v, axis=0) for level, v in sig.items()}, spread, np.mean(colours, axis=0)


def work(job):
    clear, depth, scene = load_clear_scenes(SCENES, SIDE, only=job["scene_index"])[0]
    if job["kind"] == "sweep":
        s = signature(np.clip(milky.apply(clear, depth, {"beta": job["beta"], "blur": job["blur"], "airlight": job["airlight"]}), 0, 1), clear)
        return [{"beta": job["beta"], "blur": job["blur"], "scene": scene, **s}]
    degraded = np.clip(milky.apply(clear, depth, {"beta": job["beta"], "blur": job["blur"], "airlight": job["airlight"]}), 0, 1)
    return [{"cloud": job["cloud"], "level": job["level"], "scene": scene, "method": spec["name"], "psnr": full_reference_scores(build_method(spec)(degraded), clear)["psnr"]}
            for spec in job["methods"]]


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), restval="")
        writer.writeheader()
        writer.writerows(rows)


def means(path, level=None, key="psnr"):
    groups = {}
    with open(path) as handle:
        for r in csv.DictReader(handle):
            if level in (None, "all") or r["level"] == level:
                groups.setdefault(r["method"], []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def run(workers=None, resume=True):
    targets, spread, colour = real_targets()
    airlight = [round(float(c), 3) for c in colour]
    print(f"cloud colour measured on the most turbid tank pictures: {airlight}")
    # ---- 2. sweep beta
    jobs = [{"key": f"sweep|{b}|{u}|{i}", "kind": "sweep", "beta": b, "blur": u, "scene_index": i, "airlight": airlight} for b in BETAS for u in BLURS for i in range(SCENES)]
    rows = run_stage("milky_sweep", fingerprint(BETAS, BLURS, airlight, SCENES, SIDE), jobs, work, workers=workers, memory_per_worker_gb=0.6, resume=resume)
    sweep = [{"beta": b, "blur": u, **{m: float(np.mean([r[m] for r in rows if r["beta"] == b and r["blur"] == u])) for m in METRICS}} for b in BETAS for u in BLURS]
    save_csv(sweep, TABLES / "milky_sweep.csv")
    presets, best_rows = {}, []
    for level in LEVELS:
        best = min(sweep, key=lambda r: distance(r, targets[level], spread))
        presets[level] = {"beta": best["beta"], "blur": best["blur"], "airlight": airlight}
        best_rows.append({"level": level, "beta": best["beta"], "blur": best["blur"], "distance": round(distance(best, targets[level], spread), 3),
                          **{f"synthetic_{m}": round(best[m], 3) for m in METRICS}, **{f"real_{m}": round(float(v), 3) for m, v in zip(METRICS, targets[level])}})
    save_csv(best_rows, TABLES / "milky_best.csv")
    (PROJECT / "configs" / "milky_presets.json").write_text(json.dumps({"airlight_measured": airlight, "levels": presets}, indent=2))
    # ---- 3. test the methods
    water_methods = json.loads((PROJECT / "configs" / "water.json").read_text())["methods"]
    haze_methods = json.loads((PROJECT / "configs" / "haze.json").read_text())["methods"]
    methods = list({s["name"]: s for s in water_methods + haze_methods}.values())
    jobs = [{"key": f"test|{cloud}|{level}|{i}", "kind": "test", "cloud": cloud, "level": level, "scene_index": i, "beta": presets[level]["beta"], "blur": presets[level]["blur"],
             "airlight": airlight if cloud == "milk" else INK, "methods": methods} for cloud in ("milk", "ink") for level in LEVELS for i in range(SCENES)]
    results = run_stage("milky_test", fingerprint(presets, INK, [m["name"] for m in methods], SCENES, SIDE), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
    save_csv(results, TABLES / "milky_test.csv")
    # ---- compare rankings
    summary = []
    for list_name, real_file, names in (("water methods", "water_tub.csv", [s["name"] for s in water_methods]), ("haze methods", "water_tub_hazemethods.csv", [s["name"] for s in haze_methods])):
        for band in ("all",) + LEVELS:
            real = means(TABLES / real_file, band)
            milk = {}
            for r in results:
                if r["cloud"] == "milk" and (band == "all" or r["level"] == band):
                    milk.setdefault(r["method"], []).append(r["psnr"])
            milk = {k: float(np.mean(v)) for k, v in milk.items()}
            row = {"methods": list_name, "band": band, "spearman_milky_vs_real": round(spearmanr([real[n] for n in names], [milk[n] for n in names])[0], 2)}
            if list_name == "water methods":
                old = means(TABLES / "water_matched.csv", band)
                row["spearman_original_vs_real"] = round(spearmanr([real[n] for n in names], [old[n] for n in names])[0], 2)
            row["best_real"] = max(names, key=lambda n: real[n])
            row["best_milky"] = max(names, key=lambda n: milk[n])
            summary.append(row)
    save_csv(summary, TABLES / "milky_agreement.csv")
    # ---- print
    print("\nCalibrated density per band (synthetic signature against real):")
    for r in best_rows:
        print(f"  {r['level']:7} beta {r['beta']} blur {r['blur']}  distance {r['distance']}")
    print("\nRanking agreement with the real tank photos (Spearman over methods):")
    for r in summary:
        print(f"  {r['methods']:14} {r['band']:7} milky {r['spearman_milky_vs_real']:+.2f}" + (f"   original synthetic water {r['spearman_original_vs_real']:+.2f}" if "spearman_original_vs_real" in r else "") + f"   best real {r['best_real']}, best milky {r['best_milky']}")
    print("\nMean PSNR by cloud type (all levels): none / DCP guided / CLAHE / red_comp / best method")
    for cloud in ("milk", "ink"):
        own = {}
        for r in results:
            if r["cloud"] == cloud:
                own.setdefault(r["method"], []).append(r["psnr"])
        own = {k: np.mean(v) for k, v in own.items()}
        best = max(own, key=own.get)
        print(f"  {cloud}: none {own['none']:.2f}  dcp_guided {own['dcp_guided']:.2f}  clahe {own['clahe']:.2f}  red_comp {own['red_comp']:.2f}  best {best} {own[best]:.2f}")
    make_figures(sweep, targets, spread, presets, airlight)
    return True


def make_figures(sweep, targets, spread, presets, airlight):
    """Left: how far each beta is from the real signature, per band. Right: example clear | milk | ink | real tank pair at each density."""
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for level in LEVELS:
        axis.plot(BETAS, [distance(next(r for r in sweep if r["beta"] == b and r["blur"] == presets[level]["blur"]), targets[level], spread) for b in BETAS], marker="o", markersize=3, label=f"{level} (blur {presets[level]['blur']})")
    axis.set_xscale("log")
    axis.set_xlabel("beta (cloud density)")
    axis.set_ylabel("distance from real tank signature")
    axis.legend()
    axis.grid(alpha=0.25)
    axis.set_title("Which density looks like the real milky water?")
    figure.tight_layout()
    figure.savefig(FIGURES / "milky_tuning.png", dpi=110)
    plt.close(figure)
    clear, depth, _ = load_clear_scenes(SCENES, SIDE, only=3)[0]
    pair = {r["level"]: r for r in real_pairs()}
    rows = []
    for level in LEVELS:
        milk = np.clip(milky.apply(clear, depth, {"beta": presets[level]["beta"], "blur": presets[level]["blur"], "airlight": airlight}), 0, 1)
        ink = np.clip(milky.apply(clear, depth, {"beta": presets[level]["beta"], "blur": presets[level]["blur"], "airlight": INK}), 0, 1)
        real = load_image(CURATED / pair[level]["degraded_path"], SIDE)
        rows.append([(f"clear (SYNTHETIC scene)", clear), (f"milky {level}, beta {presets[level]['beta']}", milk), (f"INK {level} (what-if)", ink), (f"real tank, {level} (REAL)", real)])
    tiles = []
    for row in rows:
        framed = []
        for caption, picture in row:
            image = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            image.thumbnail((SIDE, SIDE))
            canvas = Image.new("RGB", (SIDE, SIDE * 3 // 4 + 16), (30, 30, 30))
            canvas.paste(image, (0, 16))
            ImageDraw.Draw(canvas).text((3, 2), caption, fill=(225, 225, 225))
            framed.append(canvas)
        tiles.append(framed)
    sheet = Image.new("RGB", (SIDE * 4, sum(t[0].height for t in tiles)), (30, 30, 30))
    y = 0
    for framed in tiles:
        for k, canvas in enumerate(framed):
            sheet.paste(canvas, (k * SIDE, y))
        y += framed[0].height
    sheet.save(FIGURES / "milky_examples.jpg", quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
