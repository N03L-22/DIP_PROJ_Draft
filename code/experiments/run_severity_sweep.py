"""STAGE 4c - severity sweep: how well does each method cope as the degradation gets stronger?

We degrade clear NYU scenes at a ladder of strengths (0.25 x to 3 x the medium preset), restore each with every
method in configs/<medium>.json, and plot PSNR against strength. The point where a curve drops below the 'none' curve
is the strength at which that method stops helping (or starts to hurt). One job = one strength and one scene.

Output:  results/tables/severity_<medium>.csv     results/figures/severity_<medium>.png
Run from the project root:  .venv/bin/python code/experiments/run_severity_sweep.py --medium haze
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade.presets import severity_params  # noqa: E402
from degrade.stack import MODELS  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
FACTORS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # multiples of the 'medium' preset


def work(job):
    """One job (in a worker): degrade one scene at one strength, restore it with every method, return one row per method."""
    clear, depth, _ = load_clear_scenes(job["scenes"], job["max_side"], only=job["scene_index"])[0]
    degraded = np.clip(MODELS[job["medium"]].apply(clear, depth, severity_params(job["medium"], job["factor"])), 0, 1)
    rows = []
    for spec in job["methods"]:
        scores = full_reference_scores(build_method(spec)(degraded), clear)
        rows.append({"medium": job["medium"], "strength": job["factor"], "scene": job["scene_index"], "method": spec["name"],
                     "psnr": scores["psnr"], "ssim": scores["ssim"]})
    return rows


def run(medium, scenes=6, max_side=320, workers=None, resume=True):
    """Run the severity sweep for one medium."""
    config_text = (PROJECT / "configs" / f"{medium}.json").read_text()
    methods = json.loads(config_text)["methods"]
    jobs = [{"key": f"{factor}|{index}", "medium": medium, "factor": factor, "scene_index": index, "scenes": scenes,
             "max_side": max_side, "methods": methods} for factor in FACTORS for index in range(scenes)]
    rows = run_stage(f"severity_{medium}", fingerprint(medium, scenes, max_side, config_text), jobs, work,
                     workers=workers, memory_per_worker_gb=0.8, resume=resume)
    # average over scenes: one value per strength and method
    averaged = []
    for factor in FACTORS:
        for spec in methods:
            own = [r for r in rows if r["strength"] == factor and r["method"] == spec["name"]]
            averaged.append({"medium": medium, "strength": factor, "method": spec["name"],
                             "psnr": float(np.mean([r["psnr"] for r in own])), "ssim": float(np.mean([r["ssim"] for r in own]))})
    table = PROJECT / "results" / "tables" / f"severity_{medium}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(averaged[0].keys()))
        writer.writeheader()
        writer.writerows(averaged)
    # plot PSNR and SSIM against strength, one line per method ('none' is thick, dashed and black as the reference)
    figure, (psnr_axis, ssim_axis) = plt.subplots(1, 2, figsize=(12, 4.6))
    for spec in methods:
        own = [r for r in averaged if r["method"] == spec["name"]]
        style = {"linestyle": "--", "linewidth": 2.6, "color": "black"} if spec["name"] == "none" else {"marker": "o", "linewidth": 1.2, "markersize": 3}
        psnr_axis.plot([r["strength"] for r in own], [r["psnr"] for r in own], label=spec["name"], **style)
        ssim_axis.plot([r["strength"] for r in own], [r["ssim"] for r in own], label=spec["name"], **style)
    for axis, label in ((psnr_axis, "PSNR (dB)"), (ssim_axis, "SSIM")):
        axis.set_xlabel("degradation strength (1 = the medium preset)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    psnr_axis.legend(fontsize=7, ncol=2)
    figure.suptitle(f"{medium}: how each method copes as the degradation grows ({scenes} scenes; dashed black = no restoration)")
    figure.tight_layout()
    path = PROJECT / "results" / "figures" / f"severity_{medium}.png"
    figure.savefig(path, dpi=105)
    plt.close(figure)
    print(f"wrote {table.relative_to(PROJECT)} and {path.relative_to(PROJECT)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medium", required=True, choices=["haze", "water", "glass", "plastic"])
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.medium, arguments.scenes, arguments.max_side, arguments.workers, resume=not arguments.fresh)
