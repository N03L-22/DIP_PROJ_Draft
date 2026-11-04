"""STAGE 2d - tune the synthetic haze and glass models so that they look like the real thing, then check that it helped.

From validate_models.py we know our synthetic pictures differ from the real ones (real fog is greyer and lower-contrast than our
airlight of 0.92 gives; real raindrops change almost nothing except where the drops are, while our veil lights the whole picture).
Here we sweep the models' settings and keep the ones whose SIGNATURE matches the real pairs:

  1. SWEEP   every combination of the settings on a grid, on 12 NYU scenes, giving the mean signature of each combination.
              Haze: fog density beta x airlight A, each with two structures: depth from the scene (haze thickens with distance)
              or flat (the same everywhere, like a fog machine in a room).
              Glass: droplets only (no veil, no smudge, as for rain on a pane): number, size, strength and softness of the drops.
  2. MATCH    for each real severity level (mild / medium / heavy) pick the combination whose signature is nearest to the real mean.
              Distance = root mean square of (synthetic mean - real mean) / (real spread) over the five signature numbers.
  3. CHECK    make synthetic pictures with the matched settings, run every restoration method on them, and see whether their
              ranking of the methods now agrees better with the ranking on REAL pictures (Spearman, as in Stage 5).
One job = one combination (sweep) or one scene (check); parallel and resumable.

Output:  results/tables/tuning_haze.csv  tuning_glass.csv  tuning_best.csv  tuning_check.csv   configs/calibrated_presets.json
         results/figures/tuning_haze.png  tuning_glass.png  tuning_check.png
Run from the project root, after validate_models.py:  .venv/bin/python code/experiments/tune_models.py
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade import glass, haze  # noqa: E402
from experiments.validate_models import signature  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.ranking import ranking_agreement  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE, SCENES = 256, 12
LEVELS = ("mild", "medium", "heavy")
METRICS = ("dark_shift", "contrast_ratio", "brightness_shift", "changed_share", "sharpness_ratio")

BETAS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.5, 5.5]
AIRLIGHTS = [round(a, 2) for a in np.arange(0.50, 1.001, 0.05)]
GLASS_GRID = {"droplets": [20, 40, 80, 160, 300], "droplet_max": [0.02, 0.03, 0.05], "droplet_strength": [0.3, 0.5, 0.8], "droplet_blur": [0.004, 0.008, 0.015]}


def apply_variant(medium, params, clear, depth):
    """Degrade one clear scene with the given settings. Haze 'flat' structure means the same thickness everywhere (constant depth 0.5)."""
    if medium == "haze":
        depth_map = np.full_like(depth, 0.5) if params["structure"] == "flat" else depth
        return np.clip(haze.apply(clear, depth_map, {"beta": params["beta"], "airlight": params["airlight"]}), 0, 1)
    return np.clip(glass.apply(clear, None, {"veil": 0.0, "smudge": 0.0, **{k: v for k, v in params.items() if k != "structure"}}), 0, 1)


def work(job):
    """One job (in a worker). 'sweep': mean signature of one setting combination. 'check': restore one calibrated scene with every method."""
    if job["kind"] == "sweep":
        signatures = []
        for index in range(SCENES):
            clear, depth, _ = load_clear_scenes(SCENES, SIDE, only=index)[0]
            signatures.append(signature(apply_variant(job["medium"], job["params"], clear, depth), clear))
        row = {"medium": job["medium"], **{k: v for k, v in job["params"].items()}}
        row.update({m: float(np.mean([s[m] for s in signatures])) for m in METRICS})
        return [row]
    clear, depth, scene = load_clear_scenes(SCENES, SIDE, only=job["scene_index"])[0]
    degraded = apply_variant(job["medium"], job["params"], clear, depth)
    return [{"medium": job["medium"], "level": job["level"], "scene": scene, "method": spec["name"],
             "psnr": full_reference_scores(build_method(spec)(degraded), clear)["psnr"]} for spec in job["methods"]]


def save_csv(rows, path):
    """Write dicts as a CSV. Rows may have different keys (haze and glass settings differ); the columns are all keys seen, blank where missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), restval="")
        writer.writeheader()
        writer.writerows(rows)


def real_targets(medium):
    """Mean and spread of each signature number over the real pairs, by the curated severity level: {level: (means, spreads_over_all)}."""
    with open(TABLES / "model_signature.csv") as handle:
        rows = [r for r in csv.DictReader(handle) if r["world"] == "real" and r["medium"] == medium]
    with open(PROJECT / "data" / "curated" / f"manifest_{medium}.csv") as handle:
        level_of = {f"{r['source']}_{r['id']}": r["level"] for r in csv.DictReader(handle)}
    spread = np.array([np.std([float(r[m]) for r in rows]) for m in METRICS])
    return {lv: np.array([np.mean([float(r[m]) for r in rows if level_of[r["picture"]] == lv]) for m in METRICS]) for lv in LEVELS}, spread


def distance(row, target, spread):
    """How far a combination's signature is from the real target, in units of the real spread (root mean square over the numbers)."""
    return float(np.sqrt(np.mean(((np.array([row[m] for m in METRICS]) - target) / spread) ** 2)))


def run(workers=None, resume=True):
    stamp = fingerprint(SIDE, SCENES, BETAS, AIRLIGHTS, GLASS_GRID)
    haze_combos = [{"structure": s, "beta": b, "airlight": a} for s in ("scene", "flat") for b in BETAS for a in AIRLIGHTS]
    glass_combos = [dict(zip(GLASS_GRID, values), droplet_min=0.01, structure="drops") for values in itertools.product(*GLASS_GRID.values())]
    jobs = [{"key": f"haze|{c['structure']}|{c['beta']}|{c['airlight']}", "kind": "sweep", "medium": "haze", "params": c} for c in haze_combos]
    jobs += [{"key": "glass|" + "|".join(str(c[k]) for k in GLASS_GRID), "kind": "sweep", "medium": "glass", "params": c} for c in glass_combos]
    rows = run_stage("tune_sweep", stamp, jobs, work, workers=workers, memory_per_worker_gb=0.6, resume=resume)
    haze_rows, glass_rows = [r for r in rows if r["medium"] == "haze"], [r for r in rows if r["medium"] == "glass"]
    save_csv(haze_rows, TABLES / "tuning_haze.csv")
    save_csv(glass_rows, TABLES / "tuning_glass.csv")

    # --- MATCH: for each real level and each structure, the combination nearest to the real signature
    best_rows, calibrated = [], {"haze": {}, "glass": {}}
    for medium, sweep_rows, structures in (("haze", haze_rows, ("scene", "flat")), ("glass", glass_rows, ("drops",))):
        targets, spread = real_targets(medium)
        for structure in structures:
            own = [r for r in sweep_rows if r["structure"] == structure]
            for level in LEVELS:
                best = min(own, key=lambda r: distance(r, targets[level], spread))
                best_rows.append({"medium": medium, "structure": structure, "level": level, "distance": round(distance(best, targets[level], spread), 3),
                                  **{k: best[k] for k in (("beta", "airlight") if medium == "haze" else tuple(GLASS_GRID))},
                                  **{f"synthetic_{m}": round(best[m], 3) for m in METRICS}, **{f"real_{m}": round(float(v), 3) for m, v in zip(METRICS, targets[level])}})
    save_csv(best_rows, TABLES / "tuning_best.csv")
    print("\nBest-matching settings for each real severity level (distance in units of the real spread; 0 = identical signature)")
    for r in best_rows:
        settings = f"beta {r['beta']}, airlight {r['airlight']}" if r["medium"] == "haze" else f"droplets {r['droplets']}, max radius {r['droplet_max']}, strength {r['droplet_strength']}, blur {r['droplet_blur']}"
        print(f"  {r['medium']:6}{r['structure']:7}{r['level']:8} distance {r['distance']:.2f}   {settings}")
    # the structure with the smaller total distance is kept for haze
    totals = {s: sum(r["distance"] for r in best_rows if r["medium"] == "haze" and r["structure"] == s) for s in ("scene", "flat")}
    chosen_structure = min(totals, key=totals.get)
    print(f"  haze structure: scene depth total distance {totals['scene']:.2f}, flat {totals['flat']:.2f} -> keep '{chosen_structure}'")
    for r in best_rows:
        if r["medium"] == "haze" and r["structure"] == chosen_structure:
            calibrated["haze"][r["level"]] = {"structure": chosen_structure, "beta": r["beta"], "airlight": r["airlight"]}
        if r["medium"] == "glass":
            calibrated["glass"][r["level"]] = {k: r[k] for k in GLASS_GRID} | {"droplet_min": 0.01, "structure": "drops"}
    (PROJECT / "configs" / "calibrated_presets.json").write_text(json.dumps(calibrated, indent=2))

    # --- figure: signature distance over the haze grid (best structure), one panel per real level; and the glass best match
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    targets, spread = real_targets("haze")
    for axis, level in zip(axes, LEVELS):
        grid = np.array([[distance(next(r for r in haze_rows if r["structure"] == chosen_structure and r["beta"] == b and r["airlight"] == a), targets[level], spread)
                          for a in AIRLIGHTS] for b in BETAS])
        image = axis.imshow(np.minimum(grid, 4), origin="lower", aspect="auto", cmap="viridis_r")
        axis.set_xticks(range(0, len(AIRLIGHTS), 2))
        axis.set_xticklabels([AIRLIGHTS[i] for i in range(0, len(AIRLIGHTS), 2)], fontsize=8)
        axis.set_yticks(range(len(BETAS)))
        axis.set_yticklabels(BETAS, fontsize=7)
        i, j = np.unravel_index(np.argmin(grid), grid.shape)
        axis.plot(j, i, "r*", markersize=14)
        axis.set_xlabel("airlight A")
        axis.set_ylabel("fog density beta")
        axis.set_title(f"real {level} haze: distance of the synthetic signature (red star = best)")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(f"Tuning the haze model ('{chosen_structure}' structure): which settings look like the real fog?")
    figure.tight_layout()
    figure.savefig(FIGURES / "tuning_haze.png", dpi=105)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.2
    glass_best = [r for r in best_rows if r["medium"] == "glass"]
    for axis, metrics in zip(axes, (("contrast_ratio", "sharpness_ratio"), ("changed_share", "brightness_shift", "dark_shift"))):
        x = np.arange(len(metrics))
        for offset, r in enumerate(glass_best):
            axis.bar(x + (offset - 1) * width, [r[f"synthetic_{m}"] for m in metrics], width, label=f"synthetic, tuned to real {r['level']}", color=("#6fa3d8", "#3d6299", "#1f3a63")[offset])
            axis.bar(x + (offset - 1) * width, [r[f"real_{m}"] for m in metrics], width, fill=False, edgecolor="#b05d0c", linewidth=1.6)
        axis.set_xticks(x)
        axis.set_xticklabels(metrics, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.suptitle("Glass tuned to raindrops: filled bars synthetic, orange outlines the real pairs of the same level")
    figure.tight_layout()
    figure.savefig(FIGURES / "tuning_glass.png", dpi=105)
    plt.close(figure)

    # --- CHECK: do the calibrated synthetic pictures rank the restoration methods like the real pictures do?
    check_jobs, method_sets = [], {}
    for medium in ("haze", "glass"):
        methods = json.loads((PROJECT / "configs" / f"{medium}.json").read_text())["methods"]
        method_sets[medium] = [m["name"] for m in methods]
        for level in LEVELS:
            for index in range(SCENES):
                check_jobs.append({"key": f"{medium}|{level}|{index}", "kind": "check", "medium": medium, "level": level, "scene_index": index,
                                   "params": calibrated[medium][level], "methods": methods})
    check = run_stage("tune_check", fingerprint(stamp, calibrated), check_jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
    save_csv(check, TABLES / "tuning_check.csv")
    summary = []
    for medium in ("haze", "glass"):
        with open(TABLES / f"{medium}_real.csv") as handle:
            real = list(csv.DictReader(handle))
        with open(TABLES / f"{medium}_matched.csv") as handle:
            old = list(csv.DictReader(handle))
        names = method_sets[medium]
        mean = lambda rows, name, level=None: float(np.mean([float(r["psnr"]) for r in rows if r["method"] == name and (level is None or r["level"] == level)]))
        for level in (None,) + LEVELS:
            real_scores = [mean(real, n, level) for n in names]
            old_scores = [mean(old, n, level) for n in names]
            new_scores = [mean(check, n, level) for n in names]
            summary.append({"medium": medium, "level": level or "all",
                            "agreement_old_synthetic": round(ranking_agreement(old_scores, real_scores)["spearman"], 3),
                            "agreement_calibrated": round(ranking_agreement(new_scores, real_scores)["spearman"], 3),
                            "best_real": names[int(np.argmax(real_scores))], "best_old": names[int(np.argmax(old_scores))], "best_calibrated": names[int(np.argmax(new_scores))]})
    save_csv(summary, TABLES / "tuning_agreement.csv")
    print("\nDoes calibration make synthetic data predict real rankings better? (Spearman with the real ranking)")
    for r in summary:
        print(f"  {r['medium']:6}{r['level']:8} old synthetic {r['agreement_old_synthetic']:+.2f}   calibrated {r['agreement_calibrated']:+.2f}   best real: {r['best_real']:14} old: {r['best_old']:14} calibrated: {r['best_calibrated']}")
    figure, axis = plt.subplots(figsize=(8, 3.8))
    labels = [f"{r['medium']} {r['level']}" for r in summary]
    x = np.arange(len(summary))
    axis.bar(x - 0.2, [r["agreement_old_synthetic"] for r in summary], 0.4, label="original synthetic pairs", color="#888888")
    axis.bar(x + 0.2, [r["agreement_calibrated"] for r in summary], 0.4, label="calibrated synthetic pairs", color="#3d6299")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("ranking agreement with real (Spearman)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "tuning_check.png", dpi=105)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
