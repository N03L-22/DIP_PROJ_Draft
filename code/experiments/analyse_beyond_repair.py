"""STAGE 4m - why are some gaps so large, and which pictures are beyond repair?

A picture is BEYOND REPAIR when the damage has destroyed the information a restoration would need, so that even the exact inverse cannot bring it back:
  * SYNTHETIC (8-bit pictures): the inverse divides by t, which multiplies the rounding error of 0.0011 (1/sqrt(12) grey levels) by 1/t. Where the amplified error exceeds a typical
    scene contrast of 0.2 (that is, t < 0.0055) a pixel carries less than one unit of signal-to-noise: it cannot be recovered. A picture with more than half of its pixels below that
    limit is beyond repair.
  * REAL pictures: no t is known, so we use what is left of the scene's detail: the sharpness ratio (the detail of the degraded picture relative to the clear one) from the signature
    numbers. Below about 0.1 almost nothing is left; we check whether, in that range, ANY of the methods tried improves the picture by more than 0.5 dB.
Output: results/tables/beyond_repair_*.csv   results/figures/beyond_repair.jpg    Run from the project root:  .venv/bin/python code/experiments/analyse_beyond_repair.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.imageio import load_image  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS  # noqa: E402
from experiments.analyse_chain_limits import MODELS, quantise, transmission  # noqa: E402
from experiments.validate_models import read_pairs, signature  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
CURATED = PROJECT / "data" / "curated"
LIMIT = 0.0011 / 0.2  # t below which the amplified rounding error exceeds a typical scene contrast


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def synthetic():
    out = []
    for medium in ("haze", "water"):
        with open(TABLES / f"theory_chains_synthetic_{medium}.csv") as handle:
            rows = list(csv.DictReader(handle))
        chains = [k for k in rows[0] if k not in {"world", "medium", "level", "picture", "oracle_affine", "oracle_lut", "true_inverse", "none"}]
        for r in rows:
            clear, depth, _ = load_clear_scenes(24, 256, only=int(r["picture"].replace("scene", "")))[0]
            t = transmission(medium, depth, CALIBRATED_PRESETS[medium][r["level"]])
            lost = float(np.mean(t.min(axis=2) < LIMIT))      # share of pixels where at least one colour is unrecoverable
            best = max(float(r[c]) for c in chains)
            inverse = float(r["true_inverse"])
            gap = inverse - best
            kind = "beyond repair" if lost > 0.5 else "nearly as good as the theory" if gap <= 3 else "estimation-limited (large gap)" if gap >= 10 else "in between"
            out.append({"medium": medium, "level": r["level"], "picture": r["picture"], "share_unrecoverable": round(lost, 3), "best_chain": round(best, 2), "true_inverse": round(inverse, 2),
                        "gap": round(gap, 2), "class": kind})
    save_csv(out, TABLES / "beyond_repair_synthetic.csv")
    print("SYNTHETIC: classes by medium and severity (54 pictures per medium; 18 per severity)")
    for medium in ("haze", "water"):
        for level in ("mild", "medium", "heavy"):
            own = [r for r in out if r["medium"] == medium and r["level"] == level]
            counts = {k: sum(r["class"] == k for r in own) for k in ("beyond repair", "nearly as good as the theory", "in between", "estimation-limited (large gap)")}
            print(f"  {medium:6}{level:7} mean share of unrecoverable pixels {np.mean([r['share_unrecoverable'] for r in own]):.2f}  " + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    return out


def real():
    sets = []
    haze_rows = list(csv.DictReader(open(TABLES / "haze_real.csv")))
    for p in read_pairs("haze"):
        sets.append(("real haze", p["id"], p["degraded"], p["reference"], haze_rows))
    tub_rows = list(csv.DictReader(open(TABLES / "water_tub.csv")))
    tub_ids = {r["id"] for r in tub_rows}
    manifest = {f"TUB_{r['id']}": r for r in csv.DictReader(open(CURATED / "manifest_water_tub.csv"))}
    for i in sorted(tub_ids):
        r = manifest[i]
        sets.append(("real tank water", i, str(CURATED / r["degraded_path"]), str(CURATED / r["reference_path"]), tub_rows))
    out = []
    for world, picture, degraded, reference, table in sets:
        scores = {r["method"]: float(r["psnr"]) for r in table if r["id"] == picture}
        none = scores["none"]
        best_method = max((m for m in scores if m != "none"), key=lambda m: scores[m])
        sig = signature(load_image(degraded, 256), load_image(reference, 256))
        out.append({"world": world, "picture": picture, "sharpness_ratio": round(sig["sharpness_ratio"], 3), "contrast_ratio": round(sig["contrast_ratio"], 3), "none": round(none, 2),
                    "best_method": best_method, "best_gain": round(scores[best_method] - none, 2), "degraded": degraded, "reference": reference})
    save_csv([{k: v for k, v in r.items() if k not in ("degraded", "reference")} for r in out], TABLES / "beyond_repair_real.csv")
    print("\nREAL: what is left of the detail against what the best method can do")
    summary = []
    for world in ("real haze", "real tank water"):
        own = [r for r in out if r["world"] == world]
        for low, high in ((0.0, 0.1), (0.1, 0.3), (0.3, 10)):
            part = [r for r in own if low <= r["sharpness_ratio"] < high]
            if part:
                summary.append({"world": world, "sharpness_ratio": f"{low} to {high if high < 10 else 'above'}", "pictures": len(part), "mean_best_gain_dB": round(float(np.mean([r["best_gain"] for r in part])), 2),
                                "share_with_no_gain_over_0.5dB": round(float(np.mean([r["best_gain"] <= 0.5 for r in part])), 2)})
                print(f"  {world:16} sharpness ratio {summary[-1]['sharpness_ratio']:14} {len(part):3} pictures  mean best gain {summary[-1]['mean_best_gain_dB']:+.2f} dB  no gain above 0.5 dB: {summary[-1]['share_with_no_gain_over_0.5dB']:.0%}")
    save_csv(summary, TABLES / "beyond_repair_real_summary.csv")
    sources = {}
    for r in out:
        if r["world"] == "real haze" and r["sharpness_ratio"] < 0.1:
            sources[r["picture"].split("_")[0]] = sources.get(r["picture"].split("_")[0], 0) + 1
    print("  real haze pictures with sharpness ratio below 0.1, by dataset:", sources)
    return out


def tile(caption, picture):
    image = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((250, 250))
    canvas = Image.new("RGB", (250, 250 * 3 // 4 + 30), (30, 30, 30))
    canvas.paste(image, (0, 30))
    for k, line in enumerate(caption.split("\n")[:2]):
        ImageDraw.Draw(canvas).text((3, 2 + 12 * k), line, fill=(225, 225, 225))
    return canvas


def figure(real_rows, synthetic_rows):
    """Beyond-repair examples: degraded | best method | true inverse or clear."""
    rows = []
    # synthetic heavy haze
    pick = max((r for r in synthetic_rows if r["medium"] == "haze"), key=lambda r: r["share_unrecoverable"])
    clear, depth, _ = load_clear_scenes(24, 256, only=int(pick["picture"].replace("scene", "")))[0]
    from degrade import haze  # noqa: E402
    degraded = quantise(haze.apply(clear, depth, CALIBRATED_PRESETS["haze"][pick["level"]]))
    best = build_method({"name": "hist_eq"})(degraded)
    rows.append([tile(f"synthetic haze, {pick['level']}\n{100 * pick['share_unrecoverable']:.0f}% of pixels unrecoverable", degraded), tile("a method (hist_eq)\n", best), tile("clear\n", clear)])
    for world in ("real haze", "real tank water"):
        own = [r for r in real_rows if r["world"] == world]
        worst = min(own, key=lambda r: r["sharpness_ratio"])
        d, c = load_image(worst["degraded"], 256), load_image(worst["reference"], 256)
        rows.append([tile(f"{world}: {worst['picture'][:28]}\nsharpness ratio {worst['sharpness_ratio']}", d), tile(f"best method: {worst['best_method']}\ngain {worst['best_gain']:+.1f} dB", build_method({"name": worst["best_method"]})(d)), tile("clear (reference)\n", c)])
        good = max((r for r in own if r["best_gain"] > 0), key=lambda r: r["best_gain"] * (r["sharpness_ratio"] > 0.3), default=None)
    sheet = Image.new("RGB", (3 * 250, sum(r[0].height for r in rows)), (30, 30, 30))
    y = 0
    for row in rows:
        for k, t in enumerate(row):
            sheet.paste(t, (k * 250, y))
        y += row[0].height
    sheet.save(FIGURES / "beyond_repair.jpg", quality=92)


if __name__ == "__main__":
    s = synthetic()
    r = real()
    figure(r, s)
