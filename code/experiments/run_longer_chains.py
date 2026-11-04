"""STAGE 4h - do chains of 4 and 5 steps help? (after the mathematical check of Stage 4g predicted that they would not)

Prediction (analyse_chain_limits.py): once a chain has undone the affine damage (and the blur, for plastic) every further step can only add estimation
error, so the held-out gain of an extra step beyond the 2nd or 3rd should be zero within noise. This stage tests it.
Design: three disjoint sets of scenes of the calibrated matched pairs.
   A  scenes 0-5, medium severity   search. Chains of 1 to 3 steps: all of them (Stage 4d). Chains of 4 and 5 steps: BEAM search, each of the best BEAM chains of
                                    the previous length is extended by every unused step of the pool.
   B  scenes 6-11, all severities   the best chain of each length 1..5 (chosen on A) is scored here; the length is chosen here with the one-standard-error rule:
                                    the shortest length whose score is within one paired standard error of the best.
   C  scenes 12-23, all severities  the final report of the chosen length and of every length, never used to choose anything.
One job = one chain on one set and severity; parallel and resumable.

Output:  results/tables/longer_<medium>_beam.csv  longer_<medium>_eval.csv  longer_summary.csv   results/figures/longer_chains.png
Run from the project root:  .venv/bin/python code/experiments/run_longer_chains.py
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["DIP_PRESETS"] = "calibrated"  # the matched pairs of the calibrated presets (haze, glass); water and plastic are identical in both sets

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.parallel import fingerprint, run_stage  # noqa: E402
from experiments import run_pipelines  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SEP = " > "
LEVELS = ("mild", "medium", "heavy")
BEAM = 40
MEDIA = ("haze", "water", "glass", "plastic")


def job(medium, steps, phase, level, start, pictures, key_prefix):
    return {"key": f"{key_prefix}|{SEP.join(steps)}|{level}|{start}", "steps": list(steps), "phase": phase, "medium": medium, "level": level, "start": start,
            "pictures": pictures, "max_side": 256}


def mean_psnr(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["pipeline"], []).append(float(r["psnr"]))
    return {k: float(np.mean(v)) for k, v in groups.items() if np.all(np.isfinite(v))}


def run(workers=None, resume=True):
    pools = json.loads((PROJECT / "configs" / "pipelines.json").read_text())
    summary, curves = [], {}
    for medium in MEDIA:
        pool = pools[medium]["pool"]
        with open(TABLES / f"pipelines_{medium}_search.csv") as handle:
            search_rows = list(csv.DictReader(handle))
        a_means = mean_psnr(search_rows)  # chains of 1 to 3 steps on set A
        length_of = lambda name: len(name.split(SEP))  # noqa: E731
        # --- beam search for lengths 4 and 5 on set A (medium severity, scenes 0-5, the same pictures as Stage 4d)
        beam_rows = []
        for length in (4, 5):
            previous = sorted((c for c in a_means if length_of(c) == length - 1), key=lambda c: -a_means[c])[:BEAM]
            chains = [c.split(SEP) + [step] for c in previous for step in pool if step not in c.split(SEP)]
            rows = run_stage(f"longer_{medium}_beam{length}", fingerprint(medium, pool, length, BEAM, len(chains)),
                             [job(medium, c, "search", "medium", 0, 6, f"beam{length}") for c in chains], run_pipelines.work, workers=workers,
                             memory_per_worker_gb=0.8, resume=resume)
            beam_rows += rows
            a_means.update(mean_psnr(rows))
        with open(TABLES / f"longer_{medium}_beam.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(beam_rows[0].keys()))
            writer.writeheader()
            writer.writerows(beam_rows)
        winners = {n: max((c for c in a_means if length_of(c) == n), key=lambda c: a_means[c]) for n in range(1, 6)}
        print(f"\n{medium}: best chain per length on A: " + " | ".join(f"{n}: {winners[n]} ({a_means[winners[n]]:.2f})" for n in winners), flush=True)
        # --- B and C: the five winners (and 'none') at every severity
        steps_of = {n: winners[n].split(SEP) for n in winners}
        steps_of[0] = ["none"]
        jobs = [job(medium, steps_of[n], "verify", level, start, count, f"{name}{n}") for n in steps_of for level in LEVELS for name, start, count in (("B", 6, 6), ("C", 12, 12))]
        rows = run_stage(f"longer_{medium}_eval", fingerprint(medium, winners), jobs, run_pipelines.work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
        with open(TABLES / f"longer_{medium}_eval.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        # per picture scores for each length; B = scenes 6-11 (the first 6 pictures of each severity), C = the rest
        with open(PROJECT / "data" / "synthetic" / "matched_calibrated" / "manifest_matched.csv") as handle:
            ids = [r["id"] for r in csv.DictReader(handle) if r["medium"] == medium and r["level"] == "medium"]
        set_of = {**{i: "B" for i in ids[6:12]}, **{i: "C" for i in ids[12:24]}}
        per = {}
        for r in rows:
            per.setdefault((set_of[r["picture"]], r["pipeline"], r["level"], r["picture"]), float(r["psnr"]))
        def score(subset, length):
            name = SEP.join(steps_of[length])
            return np.array([v for (s, p, lv, pic), v in sorted(per.items()) if s == subset and p == name])
        curve = {}
        for subset in ("B", "C"):
            base = score(subset, 0)
            curve[subset] = {n: score(subset, n) for n in range(0, 6)}
        # one-standard-error rule on B
        means_b = {n: curve["B"][n].mean() for n in range(1, 6)}
        best = max(means_b, key=means_b.get)
        se_best = np.std(curve["B"][best] - curve["B"][max(1, best - 1)] if best > 1 else curve["B"][best], ddof=1) / np.sqrt(len(curve["B"][best]))
        chosen = min(n for n in range(1, 6) if means_b[n] >= means_b[best] - se_best)
        curves[medium] = curve
        print(f"  held-out B (18 pictures) / C (36 pictures), PSNR dB:", flush=True)
        for n in range(0, 6):
            d = "" if n < 2 else f"   gain over {n - 1} steps: B {np.mean(curve['B'][n] - curve['B'][n - 1]):+.2f} +- {np.std(curve['B'][n] - curve['B'][n - 1], ddof=1) / np.sqrt(len(curve['B'][n])):.2f},  C {np.mean(curve['C'][n] - curve['C'][n - 1]):+.2f} +- {np.std(curve['C'][n] - curve['C'][n - 1], ddof=1) / np.sqrt(len(curve['C'][n])):.2f}"
            print(f"    {n} steps: B {curve['B'][n].mean():6.2f}   C {curve['C'][n].mean():6.2f}{d}", flush=True)
        print(f"  one-standard-error rule on B picks {chosen} step(s)", flush=True)
        summary.append({"medium": medium, "chosen_length": chosen, **{f"chain{n}": SEP.join(steps_of[n]) for n in range(1, 6)},
                        **{f"B{n}": round(float(curve["B"][n].mean()), 3) for n in range(0, 6)}, **{f"C{n}": round(float(curve["C"][n].mean()), 3) for n in range(0, 6)}})
    with open(TABLES / "longer_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for axis, medium in zip(axes, MEDIA):
        for subset, color in (("B", "#d98c2f"), ("C", "#3d6299")):
            axis.errorbar(range(0, 6), [curves[medium][subset][n].mean() for n in range(0, 6)],
                          yerr=[curves[medium][subset][n].std(ddof=1) / np.sqrt(len(curves[medium][subset][n])) for n in range(0, 6)], marker="o", capsize=3, color=color,
                          label=("B: used to choose the length" if subset == "B" else "C: reported, never used"))
        axis.set_xlabel("steps in the chain (0 = doing nothing)")
        axis.set_ylabel("PSNR (dB), held-out")
        axis.set_title(medium)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Best chain of each length (chosen on set A), SYNTHETIC; error bars: standard error over pictures", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "longer_chains.png", dpi=110)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
