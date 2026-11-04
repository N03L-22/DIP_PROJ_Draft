"""STAGE 4 - run every restoration method for one medium over one set of pictures and record the scores.

For each picture pair and each method in configs/<medium>.json:  restore, then score (against the clear reference when
there is one, plus the reference-free scores). Results go to results/tables/<medium>_<source>.csv, and one example
montage per severity level goes to results/figures/. One job = one picture (all methods), run in parallel; an interrupted
run resumes from results/checkpoints/.

  --medium  haze | water | glass | plastic
  --source  matched  our synthetic pairs (exact reference)            [all four media]
            real     public photographs with a real reference         [haze, glass; water has no reference]
  --per-level  how many pictures per severity level (default 20)
  --workers    parallel processes (default 16, at most 20)

Run from the project root, e.g.:  .venv/bin/python code/experiments/run_experiments.py --medium haze --source matched
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import matched_dir  # noqa: E402
from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.no_reference import no_reference_scores  # noqa: E402
from restore.methods import build_method  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
LEVELS = ("mild", "medium", "heavy")


def list_pairs(medium, source, per_level):
    """The picture pairs to use: a list of dicts with the degraded file, the reference file (or None), the level and the id."""
    if source.startswith("matched"):  # "matched" = original presets, "matched_calibrated" = presets fitted to real pictures
        manifest_file, base = (matched_dir() if source == "matched" else DATA / "synthetic" / source) / "manifest_matched.csv", DATA
    else:  # "real" = the public photographs; "tub" = the paired tank photographs (water only)
        manifest_file, base = DATA / "curated" / (f"manifest_{medium}_tub.csv" if source == "tub" else f"manifest_{medium}.csv"), DATA / "curated"
    if not manifest_file.exists():
        return []
    with open(manifest_file) as handle:
        rows = [r for r in csv.DictReader(handle) if r["medium"] == medium]
    pairs = []
    for level in LEVELS:
        at_level = [r for r in rows if r["level"] == level]
        if not at_level:
            continue
        # take pictures spread through the list, not just the first few, so different scenes are used
        for row in [at_level[i] for i in np.linspace(0, len(at_level) - 1, min(per_level, len(at_level))).round().astype(int)]:
            reference = str(base / row["reference_path"]) if row.get("reference_path") else None
            # ids such as "01" repeat across datasets (O-HAZE, I-HAZE, ...), so the id must include the dataset name
            picture_id = f"{row['source']}_{row['id']}" if row.get("source") else row["id"]
            pairs.append({"level": level, "id": picture_id, "degraded": str(base / row["degraded_path"]), "reference": reference})
    return pairs


def work(job):
    """One job (runs in a worker process): restore one picture with every method and score each result. Returns one row per method."""
    methods = [(spec["name"], build_method(spec)) for spec in job["methods"]]
    degraded = load_image(job["degraded"], max_side=job["max_side"])
    reference = load_image(job["reference"], max_side=job["max_side"]) if job["reference"] else None
    if reference is not None and reference.shape != degraded.shape:  # defensive: the two should already match
        resized = Image.fromarray(np.round(reference * 255).astype(np.uint8)).resize(degraded.shape[1::-1])
        reference = np.asarray(resized, dtype=np.float64) / 255.0
    rows = []
    for name, method in methods:
        started = time.time()
        restored = method(degraded)
        row = {"source": job["source"], "medium": job["medium"], "level": job["level"], "id": job["id"], "method": name,
               "seconds": round(time.time() - started, 3)}
        row.update(full_reference_scores(restored, reference) if reference is not None else {"psnr": "", "ssim": "", "ciede2000": ""})
        row.update(no_reference_scores(restored))
        rows.append(row)
    return rows


def save_montages(medium, source, pairs, config):
    """One example picture per severity level: degraded | each method | reference, saved as a single strip."""
    methods = [(spec["name"], build_method(spec)) for spec in config["methods"]]
    for level in LEVELS:
        first = next((p for p in pairs if p["level"] == level), None)
        if first is None:
            continue
        degraded = load_image(first["degraded"], max_side=config["max_side"])
        panels = [("degraded", degraded)] + [(name, method(degraded)) for name, method in methods]
        if first["reference"]:
            panels.append(("reference", load_image(first["reference"], max_side=config["max_side"])))
        tiles = []
        for caption, picture in panels:
            tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            tile.thumbnail((200, 200))
            framed = Image.new("RGB", (tile.width, tile.height + 16), (30, 30, 30))
            framed.paste(tile, (0, 16))
            ImageDraw.Draw(framed).text((3, 2), caption, fill=(220, 220, 220))
            tiles.append(framed)
        per_row = 9  # tiles per row of the montage
        rows_of_tiles = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
        montage = Image.new("RGB", (max(sum(t.width for t in r) for r in rows_of_tiles), sum(max(t.height for t in r) for r in rows_of_tiles)), (30, 30, 30))
        y = 0
        for tile_row in rows_of_tiles:
            x = 0
            for tile in tile_row:
                montage.paste(tile, (x, y))
                x += tile.width
            y += max(t.height for t in tile_row)
        out = PROJECT / "results" / "figures" / f"{medium}_{source}_{level}_example.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        montage.save(out, quality=90)


def print_summary(results, config):
    """Print the average of each score for every method and level."""
    print(f"\n{'level':8}{'method':20}{'psnr':>7}{'ssim':>7}{'ciede':>7}{'uciqe':>7}{'dark':>7}")
    for level in LEVELS:
        for spec in config["methods"]:
            subset = [r for r in results if r["level"] == level and r["method"] == spec["name"]]
            if not subset:
                continue

            def mean(key):
                return np.mean([float(r[key]) for r in subset]) if subset[0][key] != "" else float("nan")
            print(f"{level:8}{spec['name']:20}{mean('psnr'):7.2f}{mean('ssim'):7.3f}{mean('ciede2000'):7.1f}{mean('uciqe'):7.3f}{mean('dark_channel'):7.3f}")


def run(medium, source, per_level=20, workers=None, resume=True, methods_from=None):
    """Run the whole stage for one medium and source. Returns True if there was data to run.

    methods_from: use another medium's method list (e.g. haze methods on turbid water) and tag the output files with it.
    """
    tag = f"_{methods_from}methods" if methods_from and methods_from != medium else ""
    config_text = (PROJECT / "configs" / f"{methods_from or medium}.json").read_text()
    config = json.loads(config_text)
    pairs = list_pairs(medium, source, per_level)
    if not pairs:
        print(f"no {source} data for {medium}: skipped")
        return False
    jobs = [{"key": f"{p['level']}|{p['id']}", "medium": medium, "source": source, "methods": config["methods"],
             "max_side": config["max_side"], **p} for p in pairs]
    signature = fingerprint(medium, source, per_level, config_text, tag)
    results = run_stage(f"{medium}_{source}{tag}", signature, jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
    out_table = PROJECT / "results" / "tables" / f"{medium}_{source}{tag}.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    with open(out_table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {out_table.relative_to(PROJECT)} ({len(results)} rows)")
    save_montages(medium, source + tag, pairs, config)
    print_summary(results, config)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medium", required=True, choices=["haze", "water", "glass", "plastic"])
    parser.add_argument("--source", required=True, choices=["matched", "matched_calibrated", "real", "tub"])
    parser.add_argument("--methods-from", choices=["haze", "water", "glass", "plastic"], default=None, help="use another medium's method list")
    parser.add_argument("--per-level", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.medium, arguments.source, arguments.per_level, arguments.workers, resume=not arguments.fresh, methods_from=arguments.methods_from)
