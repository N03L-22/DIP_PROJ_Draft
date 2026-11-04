"""STAGE 4b - parameter sweeps: change one setting of one method at a time and see how the result moves.

This is the 'learning' experiment. For every entry of configs/sweeps.json (a method, one of its settings, and a list of
values) we restore the same matched synthetic pictures once per value and average the scores. The plot for each sweep
shows PSNR and SSIM against the setting, with a dashed line for 'no restoration' so you can see when a setting helps
and when it makes things worse. One job = one setting value (all pictures), run in parallel and resumable.

Output:  results/tables/sweeps_<medium>.csv           one row per method / setting / value (averages over the pictures)
         results/figures/sweeps_<medium>.png          one small plot per sweep
Run from the project root:  .venv/bin/python code/experiments/run_sweeps.py --medium haze --level medium
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_matched_pairs  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.no_reference import uciqe  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]


def work(job):
    """One job (in a worker): restore every picture with one method at one setting value and return the averaged scores as one row."""
    pairs = load_matched_pairs(job["medium"], job["level"], job["pictures"], job["max_side"])
    method = build_method({"name": job["method"], "params": job["params"]})
    collected = []
    for degraded, clear, _ in pairs:
        restored = method(degraded)
        scores = full_reference_scores(restored, clear)
        scores["uciqe"] = uciqe(restored)
        collected.append(scores)
    row = {"medium": job["medium"], "level": job["level"], "method": job["method"], "param": job["param"], "value": job["value"]}
    row.update({key: float(np.mean([s[key] for s in collected])) for key in collected[0]})
    return [row]


def plot_sweeps(rows, sweeps, baseline, title, path):
    """One small plot per sweep: PSNR (solid, left axis) and SSIM (dotted, right axis) against the setting; dashed line = no restoration."""
    columns = 3
    figure_rows = -(-len(sweeps) // columns)
    figure, axes = plt.subplots(figure_rows, columns, figsize=(4.3 * columns, 3.1 * figure_rows), squeeze=False)
    for axis, sweep in zip(axes.ravel(), sweeps):
        chosen = [r for r in rows if r["method"] == sweep["method"] and r["param"] == sweep["param"]]
        values = [r["value"] for r in chosen]
        positive = [v for v in values if v > 0]
        axis.plot(values, [r["psnr"] for r in chosen], "o-", color="#3d6299")
        axis.axhline(baseline["psnr"], color="#3d6299", linestyle="--", linewidth=1)
        axis.set_title(f"{sweep['method']}: {sweep['param']}", fontsize=10)
        axis.set_xlabel(sweep["param"])
        axis.set_ylabel("PSNR (dB)", color="#3d6299")
        if len(positive) == len(values) and max(values) / min(values) >= 50:
            axis.set_xscale("log")
        second = axis.twinx()
        second.plot(values, [r["ssim"] for r in chosen], "s:", color="#b05d0c")
        second.set_ylabel("SSIM", color="#b05d0c")
        axis.grid(alpha=0.25)
    for unused in axes.ravel()[len(sweeps):]:
        unused.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=105)
    plt.close(figure)


def run(medium, level="medium", pictures=6, max_side=320, workers=None, resume=True):
    """Run all sweeps for one medium."""
    sweeps_text = (PROJECT / "configs" / "sweeps.json").read_text()
    sweeps = json.loads(sweeps_text)[medium]
    common = {"medium": medium, "level": level, "pictures": pictures, "max_side": max_side}
    # job 0 is the baseline (no restoration); every other job is one value of one setting of one method
    jobs = [{"key": "baseline", "method": "none", "params": {}, "param": "-", "value": 0, **common}]
    for sweep in sweeps:
        for value in sweep["values"]:
            jobs.append({"key": f"{sweep['method']}|{sweep['param']}|{value}", "method": sweep["method"], "params": {sweep["param"]: value},
                         "param": sweep["param"], "value": value, **common})
    rows = run_stage(f"sweeps_{medium}", fingerprint(medium, level, pictures, max_side, sweeps_text), jobs, work,
                     workers=workers, memory_per_worker_gb=0.8, resume=resume)
    baseline = next(r for r in rows if r["method"] == "none")
    rows = [r for r in rows if r["method"] != "none"]
    table = PROJECT / "results" / "tables" / f"sweeps_{medium}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    title = f"{medium}: how each setting changes the result ({level} severity, {pictures} pictures; dashed = no restoration)"
    plot_sweeps(rows, sweeps, baseline, title, PROJECT / "results" / "figures" / f"sweeps_{medium}.png")
    print(f"no restoration: PSNR {baseline['psnr']:.2f} dB, SSIM {baseline['ssim']:.3f}")
    for sweep in sweeps:
        own = [r for r in rows if r["method"] == sweep["method"] and r["param"] == sweep["param"]]
        best = max(own, key=lambda r: r["psnr"])
        print(f"  {sweep['method']:20}{sweep['param']:16} best value {best['value']!s:>7}  PSNR {best['psnr']:.2f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medium", required=True, choices=["haze", "water", "glass", "plastic"])
    parser.add_argument("--level", default="medium", choices=["mild", "medium", "heavy"], help="severity of the pictures to restore")
    parser.add_argument("--pictures", type=int, default=6, help="how many pictures to average over")
    parser.add_argument("--max-side", type=int, default=320, help="shrink pictures to this longest side (keeps sweeps quick)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.medium, arguments.level, arguments.pictures, arguments.max_side, arguments.workers, resume=not arguments.fresh)
