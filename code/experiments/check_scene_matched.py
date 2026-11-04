"""STAGE 2e - the cleanest test of a synthetic model: same scene, real degradation against synthetic degradation.

Every real pair gives us a clear picture and its real degraded picture. We degrade the SAME clear picture with our calibrated
model, then run every restoration method on both versions. If the synthetic model is right, each method should score about the
same on both, and the ranking of methods should agree. This removes the scene content (indoor against outdoor, bright walls) from
the comparison, which the earlier checks could not do.

  haze    the clear picture is degraded with our ramp model, using tau and A calibrated on the OTHER half of the same dataset's
          pairs (Stage 2c); only the held-out half is tested
  glass   the clean pane picture gets the calibrated droplets-only model (settings from Stage 2d, medium severity)
One job = one real pair; parallel and resumable.

Output:  results/tables/scene_matched.csv  scene_matched_summary.csv   results/figures/scene_matched.png  scene_matched_examples.jpg
Run from the project root, after validate_models.py and tune_models.py:  .venv/bin/python code/experiments/check_scene_matched.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade import glass  # noqa: E402
from experiments.validate_models import SOURCES, ramp_model, read_pairs  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.ranking import ranking_agreement  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SIDE = 256


def synthetic_version(job, clear):
    """The calibrated synthetic degradation of one clear picture (haze: ramp model; glass: droplets only)."""
    if job["medium"] == "haze":
        return ramp_model(clear, job["tau"], job["airlight"])
    return np.clip(glass.apply(clear, None, {"veil": 0.0, "smudge": 0.0, **job["glass_params"]}), 0, 1)


def work(job):
    """One job (in a worker): restore the real degraded picture and its synthetic twin with every method; PSNR against the clear picture."""
    clear, real = load_image(job["reference"], SIDE), load_image(job["degraded"], SIDE)
    synthetic = synthetic_version(job, clear)
    rows = []
    for spec in job["methods"]:
        method = build_method(spec)
        rows.append({"medium": job["medium"], "source": job["source"], "picture": job["id"], "method": spec["name"],
                     "psnr_real": full_reference_scores(method(real), clear)["psnr"], "psnr_synthetic": full_reference_scores(method(synthetic), clear)["psnr"]})
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), restval="")
        writer.writeheader()
        writer.writerows(rows)


def run(workers=None, resume=True):
    summary_text = (TABLES / "model_validation_summary.csv").read_text()
    with open(TABLES / "model_validation_summary.csv") as handle:
        calibration = {r["source"]: (float(r["calibrated_tau"]), float(r["calibrated_airlight"])) for r in csv.DictReader(handle)}
    glass_params = {k: v for k, v in json.loads((PROJECT / "configs" / "calibrated_presets.json").read_text())["glass"]["medium"].items() if k != "structure"}
    jobs = []
    haze_methods = json.loads((PROJECT / "configs" / "haze.json").read_text())["methods"]
    glass_methods = json.loads((PROJECT / "configs" / "glass.json").read_text())["methods"]
    for source in SOURCES:
        own = [p for p in read_pairs("haze") if p["source"] == source]
        for i, pair in enumerate(own):
            if i % 2 == 1:  # the half that was NOT used to calibrate tau and A
                jobs.append({"key": pair["id"], "medium": "haze", "tau": calibration[source][0], "airlight": calibration[source][1], "methods": haze_methods, **pair})
    for pair in read_pairs("glass"):
        jobs.append({"key": "glass|" + pair["id"], "medium": "glass", "glass_params": glass_params, "methods": glass_methods, **pair})
    rows = run_stage("scene_matched", fingerprint(summary_text, glass_params, len(jobs)), jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
    save_csv(rows, TABLES / "scene_matched.csv")

    summary, scatter = [], {}
    for medium in ("haze", "glass"):
        own = [r for r in rows if r["medium"] == medium]
        names = list(dict.fromkeys(r["method"] for r in own))
        real = [float(np.mean([r["psnr_real"] for r in own if r["method"] == n])) for n in names]
        synthetic = [float(np.mean([r["psnr_synthetic"] for r in own if r["method"] == n])) for n in names]
        agreement = ranking_agreement(synthetic, real)
        # per picture: does the synthetic twin predict how much a method helps that picture? (gain over 'none', real against synthetic)
        gains_real, gains_syn = [], []
        for picture in {r["picture"] for r in own}:
            base = next(r for r in own if r["picture"] == picture and r["method"] == "none")
            for r in own:
                if r["picture"] == picture and r["method"] != "none":
                    gains_real.append(r["psnr_real"] - base["psnr_real"])
                    gains_syn.append(r["psnr_synthetic"] - base["psnr_synthetic"])
        summary.append({"medium": medium, "pictures": len({r["picture"] for r in own}), "methods": len(names), "spearman": round(agreement["spearman"], 3), "kendall": round(agreement["kendall"], 3),
                        "gain_correlation_per_picture": round(float(np.corrcoef(gains_real, gains_syn)[0, 1]), 3),
                        "best_on_real": names[int(np.argmax(real))], "best_on_synthetic_twin": names[int(np.argmax(synthetic))],
                        "mean_abs_psnr_difference_dB": round(float(np.mean(np.abs(np.array(real) - np.array(synthetic)))), 2)})
        scatter[medium] = (names, real, synthetic)
    save_csv(summary, TABLES / "scene_matched_summary.csv")
    print("\nSame scene, real against synthetic degradation (held-out half for haze):")
    for s in summary:
        print(f"  {s['medium']:6} {s['pictures']} pictures: method ranking Spearman {s['spearman']:+.2f}, Kendall {s['kendall']:+.2f}; gain-over-nothing correlation per picture {s['gain_correlation_per_picture']:+.2f}; "
              f"mean |PSNR difference| {s['mean_abs_psnr_difference_dB']} dB; best on real: {s['best_on_real']}, on synthetic twin: {s['best_on_synthetic_twin']}")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, medium in zip(axes, ("haze", "glass")):
        names, real, synthetic = scatter[medium]
        axis.scatter(synthetic, real, color="#3d6299")
        for n, a, b in zip(names, synthetic, real):
            axis.annotate(n, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
        low, high = min(real + synthetic) - 1, max(real + synthetic) + 1
        axis.plot([low, high], [low, high], "--", color="#888888")
        axis.set_xlabel("mean PSNR on the synthetic twin (dB)")
        axis.set_ylabel("mean PSNR on the real degraded picture (dB)")
        s = next(x for x in summary if x["medium"] == medium)
        axis.set_title(f"{medium}: same scenes, Spearman {s['spearman']:+.2f}")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "scene_matched.png", dpi=105)
    plt.close(figure)

    # examples: clear | real degraded | synthetic twin, for one pair of each haze dataset and two glass pairs
    tiles_rows = []
    for source in SOURCES:
        job = next(j for j in jobs if j["medium"] == "haze" and j["id"].startswith(source))
        clear, real = load_image(job["reference"], SIDE), load_image(job["degraded"], SIDE)
        tiles_rows.append([("clear", clear), (f"{source}: real haze", real), (f"synthetic twin (tau {job['tau']:.1f}, A {job['airlight']:.2f})", synthetic_version(job, clear))])
    for job in [j for j in jobs if j["medium"] == "glass"][:2]:
        clear, real = load_image(job["reference"], SIDE), load_image(job["degraded"], SIDE)
        tiles_rows.append([("clean pane", clear), ("real raindrops", real), ("synthetic droplets", synthetic_version(job, clear))])
    images = []
    for tiles in tiles_rows:
        framed = []
        for caption, picture in tiles:
            tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            canvas = Image.new("RGB", (tile.width, tile.height + 18), (30, 30, 30))
            canvas.paste(tile, (0, 18))
            ImageDraw.Draw(canvas).text((3, 3), caption[:44], fill=(225, 225, 225))
            framed.append(canvas)
        images.append(framed)
    sheet = Image.new("RGB", (max(sum(t.width for t in r) for r in images), sum(max(t.height for t in r) for r in images)), (30, 30, 30))
    y = 0
    for framed in images:
        x = 0
        for tile in framed:
            sheet.paste(tile, (x, y))
            x += tile.width
        y += max(t.height for t in framed)
    sheet.save(FIGURES / "scene_matched_examples.jpg", quality=92)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
