"""STAGE 4d - does the ORDER of steps matter, and can any sequence beat the DCP family?

For each medium, configs/pipelines.json lists a pool of steps (white balance, DCP, CLAHE, sharpening ...). We try EVERY ordered
sequence of one, two or three different steps from the pool, for example "gray_world > dcp_guided > unsharp":
    1. SEARCH   run every sequence on scenes set A (medium severity) and rank them by PSNR.
    2. VERIFY   take the best few sequences plus the DCP reference methods and test them on scenes set B, which the
                search never saw, at all three severities. Picking the winner on the same pictures it is judged on would
                flatter it, so only the held-out numbers are used for the verdict.
    3. REPORT   which sequence wins, by how much, on what share of the pictures, and which two-step orders differ most.
One job = one sequence (search) or one sequence at one severity (verify); both run in parallel and are resumable.

Output:  results/tables/pipelines_<medium>_search.csv   results/tables/pipelines_<medium>_verify.csv
         results/tables/pipelines_<medium>_order_effects.csv   results/figures/pipelines_<medium>.png and _example.jpg
Run from the project root:  .venv/bin/python code/experiments/run_pipelines.py --medium haze
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
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_matched_pairs  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.no_reference import uciqe  # noqa: E402
from restore.methods import build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
LEVELS = ("mild", "medium", "heavy")
SEPARATOR = " > "  # how a sequence is written: "gray_world > dcp_guided"


def work(job):
    """One job (in a worker): run one sequence on the pictures of one set at one severity; one row per picture."""
    run = build_sequence(job["steps"])
    rows = []
    for degraded, clear, picture_id in load_matched_pairs(job["medium"], job["level"], job["pictures"], job["max_side"], start=job["start"]):
        restored = run(degraded)
        scores = full_reference_scores(restored, clear)
        rows.append({"medium": job["medium"], "phase": job["phase"], "level": job["level"], "pipeline": SEPARATOR.join(job["steps"]),
                     "length": len(job["steps"]), "picture": picture_id, "psnr": scores["psnr"], "ssim": scores["ssim"],
                     "ciede2000": scores["ciede2000"], "uciqe": uciqe(restored)})
    return rows


def average_by_pipeline(rows, level=None):
    """Average the per-picture rows into one row per pipeline (optionally only one severity): {pipeline: {psnr, ssim, ...}}."""
    grouped = {}
    for row in rows:
        if level is None or row["level"] == level:
            grouped.setdefault(row["pipeline"], []).append(row)
    return {name: {key: float(np.mean([r[key] for r in group])) for key in ("psnr", "ssim", "ciede2000", "uciqe")} | {"length": group[0]["length"]}
            for name, group in grouped.items()}


def save_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(medium, workers=None, resume=True, quick=False):
    """Search, verify and report for one medium. quick=True is a small version (sequences of up to 2 steps, 2 pictures) to test the pipeline."""
    settings = json.loads((PROJECT / "configs" / "pipelines.json").read_text())
    if quick:
        settings = {**settings, "max_length": 2, "search_pictures": 2, "verify_pictures": 2, "top_k": 3}
    pool, references = settings[medium]["pool"], settings[medium]["references"]
    common = {"medium": medium, "max_side": settings["max_side"]}
    stamp = fingerprint(medium, settings)

    # --- 1. SEARCH: every ordered sequence of 1..max_length different steps, on scenes set A at medium severity
    sequences = [list(s) for n in range(1, settings["max_length"] + 1) for s in itertools.permutations(pool, n)]
    search_jobs = [{"key": SEPARATOR.join(s), "steps": s, "phase": "search", "level": "medium", "start": 0,
                    "pictures": settings["search_pictures"], **common} for s in sequences]
    print(f"{medium}: {len(sequences)} sequences from a pool of {len(pool)} steps", flush=True)
    search_rows = run_stage(f"pipelines_{medium}_search", stamp, search_jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
    search = average_by_pipeline(search_rows)
    ranked = sorted(search, key=lambda name: -search[name]["psnr"])

    # --- 2. VERIFY: the best few sequences, the DCP references and 'none', on scenes set B at every severity
    # the candidate we will judge is the best MULTI-STEP sequence of the search: chosen without seeing set B
    candidate = next(name for name in ranked if search[name]["length"] >= 2)
    finalists = list(dict.fromkeys(ranked[: settings["top_k"]] + [candidate]))
    contenders = [name.split(SEPARATOR) for name in finalists] + [[r] for r in references if SEPARATOR.join([r]) not in finalists] + [["none"]]
    verify_jobs = [{"key": f"{SEPARATOR.join(steps)}|{level}", "steps": steps, "phase": "verify", "level": level,
                    "start": settings["search_pictures"], "pictures": settings["verify_pictures"], **common}
                   for steps in contenders for level in LEVELS]
    verify_rows = run_stage(f"pipelines_{medium}_verify", stamp, verify_jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)

    # --- 3. REPORT
    save_table(search_rows, PROJECT / "results" / "tables" / f"pipelines_{medium}_search.csv")
    save_table(verify_rows, PROJECT / "results" / "tables" / f"pipelines_{medium}_verify.csv")
    held_out = average_by_pipeline(verify_rows)  # averaged over the three severities and the held-out pictures
    best_reference = max(references, key=lambda r: held_out[r]["psnr"])
    best_sequence = candidate  # not re-picked on the held-out pictures, so the verdict is not flattered by choosing the winner twice
    best_on_heldout = max((name for name in held_out if name not in references and name != "none"), key=lambda n: held_out[n]["psnr"])
    # per-picture comparison of the best sequence against the best reference, on the same held-out pictures
    picture_scores = {}
    for row in verify_rows:
        picture_scores.setdefault((row["level"], row["picture"]), {})[row["pipeline"]] = row["psnr"]
    wins = sum(1 for s in picture_scores.values() if s[best_sequence] > s[best_reference])
    print(f"\n{medium} - held-out verdict ({len(picture_scores)} picture/severity cases)")
    print(f"  none                     PSNR {held_out['none']['psnr']:.2f}  SSIM {held_out['none']['ssim']:.3f}")
    for ref in references:
        print(f"  {ref:24} PSNR {held_out[ref]['psnr']:.2f}  SSIM {held_out[ref]['ssim']:.3f}   (reference)")
    print(f"  candidate (best multi-step sequence in the search): {best_sequence}")
    print(f"                           PSNR {held_out[best_sequence]['psnr']:.2f}  SSIM {held_out[best_sequence]['ssim']:.3f}  "
          f"({held_out[best_sequence]['psnr'] - held_out[best_reference]['psnr']:+.2f} dB over {best_reference}, wins on {wins}/{len(picture_scores)} cases)")
    print(f"  (for information: the finalist with the highest held-out PSNR is {best_on_heldout}, {held_out[best_on_heldout]['psnr']:.2f} dB; "
          f"choosing it here would be slightly optimistic)")
    for level in LEVELS:
        by_level = average_by_pipeline(verify_rows, level)
        print(f"    {level:7} best sequence {by_level[best_sequence]['psnr']:.2f} dB vs {best_reference} {by_level[best_reference]['psnr']:.2f} dB vs none {by_level['none']['psnr']:.2f} dB")

    # how much does swapping the order of two steps change the result? (length-2 sequences, search set)
    order_rows = []
    for a, b in itertools.combinations(pool, 2):
        forward, backward = search[SEPARATOR.join([a, b])]["psnr"], search[SEPARATOR.join([b, a])]["psnr"]
        order_rows.append({"first_second": f"{a} > {b}", "second_first": f"{b} > {a}", "psnr_forward": round(forward, 3),
                           "psnr_backward": round(backward, 3), "difference_db": round(forward - backward, 3)})
    order_rows.sort(key=lambda r: -abs(r["difference_db"]))
    save_table(order_rows, PROJECT / "results" / "tables" / f"pipelines_{medium}_order_effects.csv")
    print("  order matters most for:", "; ".join(f"{r['first_second']} vs reversed ({r['difference_db']:+.1f} dB)" for r in order_rows[:3]))

    # --- figures: bar chart of the held-out PSNR, and a picture of the best sequence next to the references
    names = sorted(held_out, key=lambda n: -held_out[n]["psnr"])
    figure, axis = plt.subplots(figsize=(9, 0.38 * len(names) + 1.6))
    colours = ["#b05d0c" if n in references else ("#888888" if n == "none" else "#3d6299") for n in names]
    axis.barh(range(len(names)), [held_out[n]["psnr"] for n in names], color=colours)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("PSNR (dB), averaged over held-out pictures and the three severities")
    axis.set_title(f"{medium}: best step sequences (blue) against the DCP references (orange)\nand no restoration (grey)", fontsize=10)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(PROJECT / "results" / "figures" / f"pipelines_{medium}.png", dpi=105)
    plt.close(figure)

    degraded, clear, _ = load_matched_pairs(medium, "medium", 1, settings["max_side"], start=settings["search_pictures"])[0]
    panels = [("degraded", degraded), (f"best: {best_sequence}", build_sequence(best_sequence.split(SEPARATOR))(degraded)),
              (best_reference, build_sequence([best_reference])(degraded)), ("clear", clear)]
    tiles = []
    for caption, picture in panels:
        tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
        framed = Image.new("RGB", (tile.width, tile.height + 16), (30, 30, 30))
        framed.paste(tile, (0, 16))
        ImageDraw.Draw(framed).text((3, 2), caption[:60], fill=(220, 220, 220))
        tiles.append(framed)
    strip = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), (30, 30, 30))
    x = 0
    for tile in tiles:
        strip.paste(tile, (x, 0))
        x += tile.width
    strip.save(PROJECT / "results" / "figures" / f"pipelines_{medium}_example.jpg", quality=90)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medium", required=True, choices=["haze", "water", "glass", "plastic"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.medium, arguments.workers, resume=not arguments.fresh)
