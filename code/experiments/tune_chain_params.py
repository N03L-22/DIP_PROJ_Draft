"""STAGE 4k - should the settings INSIDE the best chains change? Tune the parameters of the best real-picture chains on set A, judge on sets B and C.

Real haze, chain  dcp_guided > gray_world > clahe  with the defaults  omega 0.95, t0 0.1, patch 15 | clip 0.01, tiles 8.
Real tank water, chain  red_comp > unsharp  with the defaults  strength 1.0 | sigma 0.008, amount 1.0.
Every combination of the grid is scored on set A; the best one is then scored, next to the defaults, on the held-out sets B and C
(the same three disjoint sets as run_longer_chains_real.py). One job = one combination; parallel and resumable.
Output: results/tables/tune_chain_<medium>.csv  tune_chain_summary.csv      Run from the project root:  .venv/bin/python code/experiments/tune_chain_params.py
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from experiments.run_longer_chains_real import pair_sets  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.methods import build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES = PROJECT / "results" / "tables"
SIDE = 256

CHAINS = {
    "haze": {"grid": {"omega": [0.85, 0.95, 0.99], "t0": [0.05, 0.1, 0.2, 0.3], "patch": [7, 15, 31], "clip_limit": [0.005, 0.01, 0.02, 0.04], "tiles": [4, 8, 16]},
             "default": {"omega": 0.95, "t0": 0.1, "patch": 15, "clip_limit": 0.01, "tiles": 8}},
    "water": {"grid": {"strength": [0.5, 1.0, 1.5, 2.0], "sigma": [0.004, 0.008, 0.016, 0.03], "amount": [0.5, 1.0, 2.0]},
              "default": {"strength": 1.0, "sigma": 0.008, "amount": 1.0}},
}


def steps(medium, p):
    if medium == "haze":
        return [{"name": "dcp_guided", "params": {"omega": p["omega"], "t0": p["t0"], "patch": p["patch"]}}, {"name": "gray_world"},
                {"name": "clahe", "params": {"clip_limit": p["clip_limit"], "tiles": p["tiles"]}}]
    return [{"name": "red_comp", "params": {"strength": p["strength"]}}, {"name": "unsharp", "params": {"sigma": p["sigma"], "amount": p["amount"]}}]


def work(job):
    run = build_sequence(steps(job["medium"], job["params"]))
    return [{"set": job["set"], "key": job["label"], "picture": pair["id"],
             "psnr": full_reference_scores(np.clip(run(load_image(pair["degraded"], SIDE)), 0, 1), load_image(pair["reference"], SIDE))["psnr"]} for pair in job["pairs"]]


def label(p):
    return "|".join(f"{k}={v}" for k, v in p.items())


def run(workers=None, resume=True):
    summary = []
    for medium, spec in CHAINS.items():
        sets = pair_sets(medium)
        combos = [dict(zip(spec["grid"], values)) for values in itertools.product(*spec["grid"].values())]
        jobs = [{"key": f"A|{label(c)}", "medium": medium, "params": c, "label": label(c), "set": "A", "pairs": sets["A"]} for c in combos]
        rows = run_stage(f"tune_chain_{medium}", fingerprint(medium, spec, [p["id"] for p in sets["A"]]), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
        mean_a = {}
        for r in rows:
            mean_a.setdefault(r["key"], []).append(r["psnr"])
        mean_a = {k: float(np.mean(v)) for k, v in mean_a.items()}
        best = max(mean_a, key=mean_a.get)
        best_params = next(c for c in combos if label(c) == best)
        finals = [{"key": f"{s}|{name}", "medium": medium, "params": prm, "label": name, "set": s, "pairs": sets[s]}
                  for name, prm in (("default", spec["default"]), ("tuned", best_params)) for s in ("A", "B", "C")]
        results = run_stage(f"tune_chain_{medium}_final", fingerprint(medium, best, spec["default"], [p["id"] for s in "ABC" for p in sets[s]]), finals, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
        with open(TABLES / f"tune_chain_{medium}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["combination", "psnr_set_A"])
            writer.writerows(sorted(mean_a.items(), key=lambda kv: -kv[1]))
        def scores(name, s):
            return np.array([r["psnr"] for r in results if r["set"] == s and r["key"] == name])
        # the results rows' "key" is the job label ("default" or "tuned")
        out = {"medium": medium, "default_params": label(spec["default"]), "tuned_params": best}
        print(f"\n{medium} (real): best combination on A: {best}")
        for s in ("A", "B", "C"):
            d, t = scores("default", s), scores("tuned", s)
            gain = t - d
            out.update({f"default_{s}": round(float(d.mean()), 2), f"tuned_{s}": round(float(t.mean()), 2), f"gain_{s}": round(float(gain.mean()), 2),
                        f"se_{s}": round(float(gain.std(ddof=1) / np.sqrt(len(gain))), 2)})
            print(f"  set {s} ({len(d):2d} pairs): default {d.mean():6.2f}  tuned {t.mean():6.2f}  gain {gain.mean():+.2f} +- {gain.std(ddof=1) / np.sqrt(len(gain)):.2f}")
        summary.append(out)
    with open(TABLES / "tune_chain_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in summary for k in r)))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    run(None, True)
