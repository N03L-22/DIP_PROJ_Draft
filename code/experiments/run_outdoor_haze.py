"""STAGE 2g - does the SCENE explain why synthetic haze ranks methods differently from real haze? Indoor against outdoor scenes.

The synthetic haze of the earlier stages is made from INDOOR NYU rooms; the real haze pictures are mostly outdoor. Three synthetic haze sets, the same 17 haze methods:
   nyu    90 NYU rooms (indoor), our calibrated haze model, 3 severities          (the baseline: what we had); nyu_original: the same rooms with the original hand-chosen haze
   reside 100 RESIDE-SOTS outdoor pictures: clear scene + haze made by the RESIDE authors (their own depth and settings)   (outdoor, haze not ours)
   diode  90 DIODE outdoor pictures with LASER-SCANNED depth, our calibrated haze model, 3 severities                     (outdoor, our haze, true depth)
For each we ask: do the methods rank like they do on the REAL haze pairs (haze_real.csv)? Spearman and Kendall over the 17 methods; also the best method and DCP.
One job = one picture (all severities, all methods); parallel and resumable.
Output:  results/tables/outdoor_haze.csv  outdoor_haze_summary.csv   results/figures/outdoor_haze.png
Run from the project root:  .venv/bin/python code/experiments/run_outdoor_haze.py
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
from PIL import Image  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.imageio import load_image, normalise_depth  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade import haze  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, LEVELS, PRESETS  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
EXISTING = PROJECT / "data" / "existing"
SIDE, PICTURES = 256, 90


def diode_scenes():
    """The DIODE validation outdoor pictures: [(rgb file, depth file, mask file)], evenly spread (empty list when DIODE is not downloaded yet)."""
    root = EXISTING / "DIODE" / "val" / "outdoor"
    files = sorted(root.rglob("*.png"))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, min(PICTURES, len(files))).round().astype(int)] if files else []
    return [(f, f.with_name(f.stem + "_depth.npy"), f.with_name(f.stem + "_depth_mask.npy")) for f in pick]


def diode_load(rgb, depth_file, mask_file):
    """Clear picture and a 0 (near) .. 1 (far) depth map at 256 pixels. Pixels without a laser return (mostly sky) are treated as the farthest."""
    clear = load_image(rgb, SIDE)
    depth = np.load(depth_file).squeeze().astype(np.float64)
    valid = np.load(mask_file).squeeze() > 0
    depth = np.where(valid, depth, depth[valid].max() if valid.any() else 1.0)
    depth = np.asarray(Image.fromarray(depth).resize(clear.shape[1::-1], Image.BILINEAR))
    return clear, normalise_depth(depth)


def work(job):
    """One job (in a worker): one picture, every method; PSNR and SSIM against the clear picture."""
    if job["world"] == "reside":
        clear, hazy = load_image(job["clear"], SIDE), load_image(job["hazy"], SIDE)
        cases = [("reside", hazy)]
    else:
        if job["world"] in ("nyu", "nyu_original"):
            clear, depth, _ = load_clear_scenes(PICTURES, SIDE, only=job["index"])[0]
        else:
            clear, depth = diode_load(*job["files"])
        presets = PRESETS if job["world"] == "nyu_original" else CALIBRATED_PRESETS  # the first hand-chosen haze, or the one fitted to real pictures
        cases = [(level, np.clip(haze.apply(clear, depth, presets["haze"][level]), 0, 1)) for level in LEVELS]
    rows = []
    for level, degraded in cases:
        base = full_reference_scores(degraded, clear)
        rows.append({"world": job["world"], "level": level, "picture": job["picture"], "method": "none", "psnr": base["psnr"], "ssim": base["ssim"]})
        for name in job["methods"]:
            if name == "none":
                continue
            scores = full_reference_scores(np.clip(build_method({"name": name})(degraded), 0, 1), clear)
            rows.append({"world": job["world"], "level": level, "picture": job["picture"], "method": name, "psnr": scores["psnr"], "ssim": scores["ssim"]})
    return rows


def build_jobs(methods):
    jobs = {"nyu": [{"key": f"nyu|{i}", "world": "nyu", "index": i, "picture": f"nyu{i}", "methods": methods} for i in range(PICTURES)],
            "nyu_original": [{"key": f"nyuo|{i}", "world": "nyu_original", "index": i, "picture": f"nyu{i}", "methods": methods} for i in range(PICTURES)]}
    hazy = sorted((EXISTING / "RESIDE-SOTS" / "SOTS" / "outdoor" / "hazy").glob("*.jpg"))
    pairs = [(h, EXISTING / "RESIDE-SOTS" / "SOTS" / "outdoor" / "gt" / (h.stem.split("_")[0] + ".png")) for h in hazy]
    pairs = [p for p in pairs if p[1].exists()][::5]
    jobs["reside"] = [{"key": f"reside|{h.stem}", "world": "reside", "hazy": str(h), "clear": str(c), "picture": h.stem, "methods": methods} for h, c in pairs]
    jobs["diode"] = [{"key": f"diode|{f.stem}", "world": "diode", "files": [str(f), str(d), str(m)], "picture": f.stem, "methods": methods} for f, d, m in diode_scenes()]
    return jobs


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(workers=None, resume=True):
    methods = [s["name"] for s in json.loads((PROJECT / "configs" / "haze.json").read_text())["methods"]]
    groups = {}
    with open(TABLES / "haze_real.csv") as handle:
        for r in csv.DictReader(handle):
            groups.setdefault(r["method"], []).append(float(r["psnr"]))
    real = {m: float(np.mean(v)) for m, v in groups.items()}
    rows = []
    jobs = build_jobs(methods)
    for world, world_jobs in jobs.items():
        if not world_jobs:
            print(f"{world}: no pictures (not downloaded yet), skipped")
            continue
        rows += run_stage(f"outdoor_haze_{world}", fingerprint(methods, world, len(world_jobs), SIDE), world_jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
    save_csv(rows, TABLES / "outdoor_haze.csv")
    summary = []
    print("\nWhich scenes make synthetic haze rank the methods like REAL haze does? (Spearman and Kendall over the methods)")
    for world in ("nyu_original", "nyu", "reside", "diode"):
        own = [r for r in rows if r["world"] == world]
        if not own:
            continue
        for level in sorted({r["level"] for r in own}) + ["all"]:
            part = [r for r in own if level == "all" or r["level"] == level]
            means = {m: float(np.mean([r["psnr"] for r in part if r["method"] == m])) for m in methods}
            order = [m for m in methods if m in real]
            rho, tau = spearmanr([means[m] for m in order], [real[m] for m in order])[0], kendalltau([means[m] for m in order], [real[m] for m in order])[0]
            best = max(means, key=means.get)
            summary.append({"world": world, "level": level, "pictures": len({r["picture"] for r in part}), "spearman": round(rho, 3), "kendall": round(tau, 3), "best_method": best, "best_psnr": round(means[best], 2),
                            "dcp_guided": round(means["dcp_guided"], 2), "dcp_clahe": round(means["dcp_clahe"], 2), "none": round(means["none"], 2)})
            print(f"  {world:7}{level:8} {summary[-1]['pictures']:3} pictures  Spearman {rho:+.2f}  Kendall {tau:+.2f}   best {best} ({means[best]:.2f})   DCP {means['dcp_guided']:.2f}  DCP>CLAHE {means['dcp_clahe']:.2f}  none {means['none']:.2f}")
    save_csv(summary, TABLES / "outdoor_haze_summary.csv")
    print(f"  REAL haze reference: best {max(real, key=real.get)} ({max(real.values()):.2f}), DCP {real['dcp_guided']:.2f}, DCP>CLAHE {real['dcp_clahe']:.2f}, none {real['none']:.2f}")
    worlds = [w for w in ("nyu_original", "nyu", "reside", "diode") if any(r["world"] == w for r in rows)]
    figure, axes = plt.subplots(1, len(worlds), figsize=(5 * len(worlds), 4.6))
    order = [m for m in methods if m in real]
    for axis, world in zip(np.atleast_1d(axes), worlds):
        own = [r for r in rows if r["world"] == world]
        means = {m: float(np.mean([r["psnr"] for r in own if r["method"] == m])) for m in order}
        axis.scatter([means[m] for m in order], [real[m] for m in order], color="#3d6299")
        for m in order:
            axis.annotate(m, (means[m], real[m]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        low, high = min(list(means.values()) + list(real.values())) - 1, max(list(means.values()) + list(real.values())) + 1
        axis.plot([low, high], [low, high], "--", color="#888888")
        rho = spearmanr([means[m] for m in order], [real[m] for m in order])[0]
        axis.set_title({"nyu_original": "indoor rooms, ORIGINAL haze", "nyu": "indoor rooms, CALIBRATED haze", "reside": "outdoor, RESIDE haze", "diode": "outdoor, DIODE depth, our haze"}[world] + f"\nSpearman {rho:+.2f}", fontsize=9)
        axis.set_xlabel("mean PSNR on synthetic haze (dB)")
        axis.set_ylabel("mean PSNR on REAL haze (dB)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "outdoor_haze.png", dpi=110)
    plt.close(figure)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
