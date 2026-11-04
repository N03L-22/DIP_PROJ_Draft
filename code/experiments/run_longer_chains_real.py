"""STAGE 4i - chains of 1 to 5 steps on REAL pictures: real haze (185 pairs) and real tank water (TUB, 1,183 pairs).

The same design as run_longer_chains.py (synthetic), with three disjoint sets of real pairs:
   A  search      chains of 1 to 3 steps: all of them; 4 and 5 steps: beam search (the best BEAM chains of the previous length, extended by every unused step)
   B  chooses     the best chain of each length 1..5 is scored; the length is chosen with the one-standard-error rule
   C  reports     never used to choose anything
Pool of steps: the pool of Stage 4d for the medium (configs/pipelines.json). Scored against the clear (haze) or least-turbid (tank water) picture.
Output:  results/tables/longer_real_<medium>_*.csv   longer_real_summary.csv   results/figures/longer_chains_real.png
Run from the project root:  .venv/bin/python code/experiments/run_longer_chains_real.py
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

from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from experiments.validate_models import read_pairs  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.methods import build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
CURATED = PROJECT / "data" / "curated"
SEP = " > "
BEAM, SIDE = 40, 256


def pair_sets(medium):
    """{'A': [...], 'B': [...], 'C': [...]} of dicts with the degraded and reference file and an id; disjoint by construction."""
    if medium == "haze":
        pairs = [{"degraded": p["degraded"], "reference": p["reference"], "id": p["id"]} for p in read_pairs("haze")]
        return {"A": pairs[0::8], "B": pairs[3::8], "C": pairs[1::4]}
    with open(CURATED / "manifest_water_tub.csv") as handle:
        rows = list(csv.DictReader(handle))
    out = {"A": [], "B": [], "C": []}
    for level in ("mild", "medium", "heavy"):
        own = [r for r in rows if r["level"] == level]
        for i, r in enumerate(own):
            item = {"degraded": str(CURATED / r["degraded_path"]), "reference": str(CURATED / r["reference_path"]), "id": r["id"]}
            if i % 40 == 0:
                out["A"].append(item)
            elif i % 40 == 13:
                out["B"].append(item)
            elif i % 20 == 7:
                out["C"].append(item)
    return out


def work(job):
    """One job (in a worker): one chain on every picture of one set; one row per picture."""
    run = build_sequence(job["steps"])
    return [{"set": job["set"], "pipeline": SEP.join(job["steps"]), "picture": pair["id"],
             "psnr": full_reference_scores(np.clip(run(load_image(pair["degraded"], SIDE)), 0, 1), load_image(pair["reference"], SIDE))["psnr"]} for pair in job["pairs"]]


def mean_psnr(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["pipeline"], []).append(float(r["psnr"]))
    return {k: float(np.mean(v)) for k, v in groups.items() if np.all(np.isfinite(v))}


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(workers=None, resume=True):
    pools = json.loads((PROJECT / "configs" / "pipelines.json").read_text())
    summary, curves = [], {}
    for medium in ("haze", "water"):
        pool, sets = pools[medium]["pool"], pair_sets(medium)
        print(f"\n{medium} (real): sets A/B/C = {len(sets['A'])}/{len(sets['B'])}/{len(sets['C'])} pairs", flush=True)

        # --- exhaustive 1 to 3 steps on A, beam for 4 and 5
        chains = [c for n in (1, 2, 3) for c in itertools.permutations(pool, n)]
        rows = run_stage(f"real_{medium}_A123", fingerprint(medium, pool, [p["id"] for p in sets["A"]]),
                         [{"key": SEP.join(c), "steps": list(c), "set": "A", "pairs": sets["A"]} for c in chains], work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
        a_means = mean_psnr(rows)
        beam_rows = list(rows)
        length_of = lambda name: len(name.split(SEP))  # noqa: E731
        for length in (4, 5):
            previous = sorted((c for c in a_means if length_of(c) == length - 1), key=lambda c: -a_means[c])[:BEAM]
            extended = [c.split(SEP) + [step] for c in previous for step in pool if step not in c.split(SEP)]
            rows = run_stage(f"real_{medium}_A{length}", fingerprint(medium, pool, length, BEAM, len(extended), [p["id"] for p in sets["A"]]),
                             [{"key": SEP.join(c), "steps": list(c), "set": "A", "pairs": sets["A"]} for c in extended], work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
            a_means.update(mean_psnr(rows))
            beam_rows += rows
        save_csv(beam_rows, TABLES / f"longer_real_{medium}_search.csv")
        winners = {n: max((c for c in a_means if length_of(c) == n), key=lambda c: a_means[c]) for n in range(1, 6)}
        print(f"{medium}: best chain per length on A: " + " | ".join(f"{n}: {winners[n]} ({a_means[winners[n]]:.2f})" for n in winners), flush=True)
        steps_of = {n: winners[n].split(SEP) for n in winners}
        steps_of[0] = ["none"]
        jobs = [{"key": f"{s}|{n}|{SEP.join(steps_of[n])}", "steps": steps_of[n], "set": s, "pairs": sets[s]} for n in steps_of for s in ("B", "C")]
        rows = run_stage(f"real_{medium}_eval", fingerprint(medium, winners, [p["id"] for p in sets["B"] + sets["C"]]), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
        save_csv(rows, TABLES / f"longer_real_{medium}_eval.csv")
        per = {(r["set"], r["pipeline"], r["picture"]): float(r["psnr"]) for r in rows}
        curve = {s: {n: np.array([per[(s, SEP.join(steps_of[n]), p["id"])] for p in sets[s]]) for n in range(6)} for s in ("B", "C")}
        means_b = {n: curve["B"][n].mean() for n in range(1, 6)}
        best = max(means_b, key=means_b.get)
        diff = curve["B"][best] - curve["B"][best - 1] if best > 1 else curve["B"][best]
        se_best = diff.std(ddof=1) / np.sqrt(len(diff))
        chosen = min(n for n in range(1, 6) if means_b[n] >= means_b[best] - se_best)
        curves[medium] = curve
        print("  held-out PSNR (dB):", flush=True)
        for n in range(6):
            extra = ""
            if n >= 2:
                se = lambda d: np.std(d, ddof=1) / np.sqrt(len(d))  # noqa: E731
                db, dc = curve["B"][n] - curve["B"][n - 1], curve["C"][n] - curve["C"][n - 1]
                extra = f"   gain over {n - 1}: B {db.mean():+.2f} +- {se(db):.2f}, C {dc.mean():+.2f} +- {se(dc):.2f}"
            print(f"    {n} steps: B {curve['B'][n].mean():6.2f}  C {curve['C'][n].mean():6.2f}{extra}", flush=True)
        print(f"  one-standard-error rule on B picks {chosen} step(s)", flush=True)
        summary.append({"medium": medium, "chosen_length": chosen, **{f"chain{n}": SEP.join(steps_of[n]) for n in range(1, 6)},
                        **{f"B{n}": round(float(curve["B"][n].mean()), 3) for n in range(6)}, **{f"C{n}": round(float(curve["C"][n].mean()), 3) for n in range(6)}})
    save_csv(summary, TABLES / "longer_real_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for axis, medium, title in zip(axes, ("haze", "water"), ("real haze", "real tank water (TUB)")):
        for subset, color, label in (("B", "#d98c2f", "B: used to choose the length"), ("C", "#3d6299", "C: reported, never used")):
            axis.errorbar(range(6), [curves[medium][subset][n].mean() for n in range(6)], yerr=[curves[medium][subset][n].std(ddof=1) / np.sqrt(len(curves[medium][subset][n])) for n in range(6)],
                          marker="o", capsize=3, color=color, label=label)
        axis.set_xlabel("steps in the chain (0 = doing nothing)")
        axis.set_ylabel("PSNR (dB), held-out")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Best chain of each length (chosen on set A), REAL pictures; error bars: standard error", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "longer_chains_real.png", dpi=110)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
