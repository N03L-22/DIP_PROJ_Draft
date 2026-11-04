"""STAGE 4f - haze removal by SUBTRACTION combined with the other tools: every ordered chain of up to three steps.

The subtraction methods (local-glow subtraction `blind_veil_local`, low-frequency subtraction `lowfreq_subtract`) were only ever tried alone, and the step-order
search of Stage 4d never had them in its pool. Here the pool holds both subtraction methods next to the DCP family and the contrast tools, and every ordered
chain of one, two or three DIFFERENT steps is scored (9 steps: 9 + 72 + 504 = 585 chains), on two kinds of pictures:
   real       the real haze pairs from the half of each dataset NOT used to measure the subtraction method's priors (set A: every 4th pair, search; set B: the pairs between, held-out check)
   synthetic  the calibrated synthetic haze pairs (set A: 6 scenes at medium severity; set B: 6 other scenes at all three severities)
Choosing the best chain on set A and judging it on set B keeps the verdict honest. One job = one chain; parallel and resumable.

Output:  results/tables/subtraction_<world>_search.csv  _verify.csv  subtraction_summary.csv  results/figures/subtraction_combos.png
Run from the project root:  .venv/bin/python code/experiments/run_subtraction_combos.py
"""
import argparse
import csv
import itertools
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
DATA = PROJECT / "data"
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE = 256
SEP = " > "
SUBTRACTION = ("blind_veil_local", "lowfreq_subtract")
POOL = ["blind_veil_local", "lowfreq_subtract", "dcp_guided", "clahe", "hist_eq", "unsharp", "gamma_auto", "tarel", "gray_world"]


def pair_sets(world):
    """(set A, set B): lists of dicts with the degraded file, the clear file and the level."""
    if world == "real":
        # only the half of each dataset that was NOT used to measure the priors of the subtraction method (odd positions within each dataset)
        pairs, seen = [], {}
        for p in read_pairs("haze"):
            seen[p["source"]] = seen.get(p["source"], -1) + 1
            if seen[p["source"]] % 2 == 1:
                pairs.append({"degraded": p["degraded"], "reference": p["reference"], "level": "real", "id": p["id"]})
        return pairs[0::4], pairs[2::4]
    with open(DATA / "synthetic" / "matched_calibrated" / "manifest_matched.csv") as handle:
        rows = [r for r in csv.DictReader(handle) if r["medium"] == "haze"]
    as_pair = lambda r: {"degraded": str(DATA / r["degraded_path"]), "reference": str(DATA / r["reference_path"]), "level": r["level"], "id": r["id"]}  # noqa: E731
    medium = [r for r in rows if r["level"] == "medium"]
    return [as_pair(r) for r in medium[:6]], [as_pair(r) for level in ("mild", "medium", "heavy") for r in [x for x in rows if x["level"] == level][6:12]]


def work(job):
    """One job (in a worker): one chain on every picture of one set; one row per picture."""
    run = build_sequence(job["steps"])
    rows = []
    for pair in job["pairs"]:
        scores = full_reference_scores(np.clip(run(load_image(pair["degraded"], SIDE)), 0, 1), load_image(pair["reference"], SIDE))
        rows.append({"world": job["world"], "phase": job["phase"], "chain": job["name"], "length": len(job["steps"]), "level": pair["level"], "picture": pair["id"],
                     "psnr": scores["psnr"], "ssim": scores["ssim"]})
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def chain_means(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["chain"], []).append(r["psnr"])
    # a chain that produces an invalid picture (NaN) on any picture is dropped from the ranking, and reported
    broken = sorted(k for k, v in groups.items() if not np.all(np.isfinite(v)))
    if broken:
        print(f"  dropped chains with invalid output: {broken}")
    return {k: float(np.mean(v)) for k, v in groups.items() if k not in broken}


def has_subtraction(name):
    return any(step in SUBTRACTION for step in name.split(SEP))


def run(workers=None, resume=True):
    summary, verify_means = [], {}
    chains = [c for length in (1, 2, 3) for c in itertools.permutations(POOL, length)]
    print(f"{len(chains)} chains from a pool of {len(POOL)} steps")
    for world in ("real", "synthetic"):
        set_a, set_b = pair_sets(world)
        jobs = [{"key": f"{world}|A|{SEP.join(c)}", "world": world, "phase": "search", "name": SEP.join(c), "steps": list(c), "pairs": set_a} for c in chains]
        rows = run_stage(f"subtraction_{world}_search", fingerprint(POOL, world, [p["id"] for p in set_a]), jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
        save_csv(rows, TABLES / f"subtraction_{world}_search.csv")
        search = chain_means(rows)
        ranked = sorted(search, key=lambda c: -search[c])
        best_with = next(c for c in ranked if has_subtraction(c) and c.count(SEP) >= 1)
        best_without = next(c for c in ranked if not has_subtraction(c) and c.count(SEP) >= 1)
        finalists = list(dict.fromkeys(ranked[:6] + [best_with, best_without, "blind_veil_local", "lowfreq_subtract", "dcp_guided", "dcp_guided > clahe", "gamma_auto"]))
        jobs = [{"key": f"{world}|B|{c}", "world": world, "phase": "verify", "name": c, "steps": c.split(SEP), "pairs": set_b} for c in finalists]
        none_job = {"key": f"{world}|B|none", "world": world, "phase": "verify", "name": "none", "steps": ["none"], "pairs": set_b}
        results = run_stage(f"subtraction_{world}_verify", fingerprint(POOL, world, finalists, [p["id"] for p in set_b]), jobs + [none_job], work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
        save_csv(results, TABLES / f"subtraction_{world}_verify.csv")
        verify = chain_means(results)
        verify_means[world] = verify
        # order effects: mean search PSNR of chains whose subtraction step is first, middle or last, against chains with no subtraction (same length)
        by_position = {}
        for c, value in search.items():
            steps = c.split(SEP)
            for k, step in enumerate(steps):
                if step in SUBTRACTION and len(steps) >= 2:
                    by_position.setdefault((len(steps), "first" if k == 0 else "last" if k == len(steps) - 1 else "middle"), []).append(value)
        without = {n: float(np.mean([v for c, v in search.items() if not has_subtraction(c) and c.count(SEP) == n - 1])) for n in (2, 3)}
        print(f"\n== {world}: search set A ({len(set_a)} pictures), held-out set B ({len(set_b)} pictures)")
        print(f"  best in search: {ranked[0]} ({search[ranked[0]]:.2f} dB); best with subtraction: {best_with}; best without: {best_without}")
        print("  held-out PSNR (dB):  " + "   ".join(f"{c}: {verify[c]:.2f}" for c in [best_with, best_without, "blind_veil_local", "lowfreq_subtract", "dcp_guided", "dcp_guided > clahe", "none"]))
        for (n, position), values in sorted(by_position.items()):
            print(f"  chains of {n} steps with a subtraction step {position:6}: mean {np.mean(values):.2f} dB   (chains with none: {without[n]:.2f})")
        summary.append({"world": world, "best_in_search": ranked[0], "best_with_subtraction": best_with, "best_without_subtraction": best_without,
                        **{f"heldout_{c.replace(SEP, '_then_')}": round(verify[c], 3) for c in [best_with, best_without, "blind_veil_local", "lowfreq_subtract", "dcp_guided", "dcp_guided > clahe", "none"]}})
    with open(TABLES / "subtraction_summary.csv", "w", newline="") as handle:
        keys = list(dict.fromkeys(k for r in summary for k in r))
        writer = csv.DictWriter(handle, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(summary)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, world in zip(axes, ("real", "synthetic")):
        verify = verify_means[world]
        names = sorted(verify, key=lambda c: verify[c])
        axis.barh(range(len(names)), [verify[c] for c in names], color=["#c0504d" if has_subtraction(c) else "#3d6299" for c in names])
        axis.set_yticks(range(len(names)), [c.replace("blind_veil_local", "veil_local").replace("lowfreq_subtract", "lowfreq_sub") for c in names], fontsize=7)
        axis.axvline(verify["none"], color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("held-out PSNR (dB); dashed line = no restoration")
        axis.set_title(f"{world} haze: red = chains with a subtraction step")
        axis.grid(alpha=0.25, axis="x")
    figure.tight_layout()
    figure.savefig(FIGURES / "subtraction_combos.png", dpi=110)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
