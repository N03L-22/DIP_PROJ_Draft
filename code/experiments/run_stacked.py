"""STAGE 6b - stacked degradations: a scene seen through TWO or THREE media at once (for example water behind fogged glass).

The proposal's stacked model:  layer 1 is nearest the scene, the last layer is nearest the camera. Three questions:
  1. How much harder is a stack than its single layers?  (score with no restoration)
  2. Which order should we undo the layers in?  Two natural strategies, each using one specialist step per layer:
        OUTERMOST FIRST: undo the layer nearest the camera first (the reverse of the order they were applied)
        INNERMOST FIRST: undo the layer nearest the scene first (the same order they were applied)
  3. Does a search over step sequences find something better?  Same method as Stage 4d: rank every sequence of up to three
     steps on scenes set A, then judge the best ones on different scenes (set B), so the winner is not flattered.
One job = one sequence (search) or one sequence at one severity (verify); parallel and resumable.

Output:  results/tables/stacked_<stack>_search.csv   stacked_<stack>_verify.csv   stacked_summary.csv
         results/figures/stacked_summary.png   stacked_examples.jpg
Run from the project root:  .venv/bin/python code/experiments/run_stacked.py
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

from common.datasets import load_clear_scenes  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade.presets import active_presets  # noqa: E402

PRESETS = active_presets()  # DIP_PRESETS=calibrated switches the whole run to the calibrated settings
from degrade.stack import apply_stack  # noqa: E402
from experiments.run_pipelines import SEPARATOR, average_by_pipeline, save_table  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.no_reference import uciqe  # noqa: E402
from restore.methods import build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SCENE_POOL = 12  # the clear scenes come from this many evenly spaced NYU pictures; set A is the first half, set B the second


def stack_layers(layer_names, level):
    """The stack as a list of layers for degrade.stack.apply_stack, every layer at the given severity preset."""
    return [{"type": name, "params": PRESETS[name][level]} for name in layer_names]


def work(job):
    """One job (in a worker): run one step sequence on some scenes degraded by one stack at one severity. One row per scene."""
    run = build_sequence(job["steps"])
    rows = []
    for position in job["scene_positions"]:
        clear, depth, scene = load_clear_scenes(SCENE_POOL, job["max_side"], only=position)[0]
        degraded = apply_stack(clear, depth, stack_layers(job["layers"], job["level"]))
        restored = run(degraded)
        scores = full_reference_scores(restored, clear)
        rows.append({"stack": job["stack"], "phase": job["phase"], "level": job["level"], "pipeline": SEPARATOR.join(job["steps"]),
                     "length": len(job["steps"]), "picture": scene, "psnr": scores["psnr"], "ssim": scores["ssim"],
                     "ciede2000": scores["ciede2000"], "uciqe": uciqe(restored)})
    return rows


def flatten(step_lists):
    """Join lists of steps into one sequence, dropping a repeated step if two specialists share it (stretch twice adds nothing)."""
    steps = []
    for part in step_lists:
        steps += [s for s in part if not steps or steps[-1] != s]
    return steps


def run(workers=None, resume=True):
    settings_text = (PROJECT / "configs" / "stacks.json").read_text()
    settings = json.loads(settings_text)
    summary_rows, examples = [], []
    for stack_name, stack in settings["stacks"].items():
        layers, pool = stack["layers"], stack["pool"]
        common = {"stack": stack_name, "layers": layers, "max_side": settings["max_side"]}
        stamp = fingerprint(stack_name, settings_text)
        set_a = list(range(settings["search_scenes"]))
        set_b = list(range(settings["search_scenes"], settings["search_scenes"] + settings["verify_scenes"]))

        # --- SEARCH: every ordered sequence of 1..max_length different steps, on scenes set A at medium severity
        sequences = [list(s) for n in range(1, settings["max_length"] + 1) for s in itertools.permutations(pool, n)]
        search_jobs = [{"key": SEPARATOR.join(s), "steps": s, "phase": "search", "level": "medium", "scene_positions": set_a, **common} for s in sequences]
        print(f"\n{stack_name}: {len(sequences)} sequences from a pool of {len(pool)} steps", flush=True)
        search_rows = run_stage(f"stacked_{stack_name}_search", stamp, search_jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
        search = average_by_pipeline(search_rows)
        ranked = sorted(search, key=lambda name: -search[name]["psnr"])
        candidate = next(name for name in ranked if search[name]["length"] >= 2)  # best multi-step sequence, chosen without seeing set B

        # --- the two ordered strategies built from the specialists: outermost layer first, or innermost layer first
        specialists = [settings["specialists"][layer] for layer in layers]
        outermost_first = flatten(list(reversed(specialists)))
        innermost_first = flatten(specialists)
        contenders = {"none": ["none"], "outermost first": outermost_first, "innermost first": innermost_first, "searched best": candidate.split(SEPARATOR)}
        for name in settings["generic"]:
            contenders[f"single: {name}"] = [name]
        for name in ranked[: settings["top_k"]]:
            contenders.setdefault(f"search #{ranked.index(name) + 1}: {name}", name.split(SEPARATOR))

        # --- VERIFY on scenes set B at every severity listed in the config
        verify_jobs = [{"key": f"{label}|{level}", "steps": steps, "phase": "verify", "level": level, "scene_positions": set_b, **common}
                       for label, steps in contenders.items() for level in settings["levels"]]
        verify_rows = run_stage(f"stacked_{stack_name}_verify", stamp, verify_jobs, work, workers=workers, memory_per_worker_gb=0.8, resume=resume)
        save_table(search_rows, PROJECT / "results" / "tables" / f"stacked_{stack_name}_search.csv")
        save_table(verify_rows, PROJECT / "results" / "tables" / f"stacked_{stack_name}_verify.csv")

        by_steps = average_by_pipeline(verify_rows)
        by_level = {level: average_by_pipeline(verify_rows, level) for level in settings["levels"]}
        generic_best = max((l for l in contenders if l.startswith("single")), key=lambda l: by_steps[SEPARATOR.join(contenders[l])]["psnr"])
        row = {"stack": stack_name, "layers": " + ".join(layers), "none": by_steps["none"]["psnr"],
               "best_single": by_steps[SEPARATOR.join(contenders[generic_best])]["psnr"], "best_single_name": generic_best.replace("single: ", ""),
               "outermost_first": by_steps[SEPARATOR.join(outermost_first)]["psnr"], "outermost_first_steps": SEPARATOR.join(outermost_first),
               "innermost_first": by_steps[SEPARATOR.join(innermost_first)]["psnr"], "innermost_first_steps": SEPARATOR.join(innermost_first),
               "searched_best": by_steps[candidate]["psnr"], "searched_best_steps": candidate}
        for level in settings["levels"]:
            row[f"none_{level}"] = by_level[level]["none"]["psnr"]
            row[f"outermost_{level}"] = by_level[level][SEPARATOR.join(outermost_first)]["psnr"]
            row[f"innermost_{level}"] = by_level[level][SEPARATOR.join(innermost_first)]["psnr"]
            row[f"searched_{level}"] = by_level[level][candidate]["psnr"]
        summary_rows.append(row)
        print(f"  none {row['none']:.2f} | best single ({row['best_single_name']}) {row['best_single']:.2f} | "
              f"outermost first [{row['outermost_first_steps']}] {row['outermost_first']:.2f} | "
              f"innermost first [{row['innermost_first_steps']}] {row['innermost_first']:.2f} | searched [{candidate}] {row['searched_best']:.2f}", flush=True)

        # --- one example picture per stack (first scene of set B, medium severity)
        clear, depth, scene = load_clear_scenes(SCENE_POOL, settings["max_side"], only=set_b[0])[0]
        degraded = apply_stack(clear, depth, stack_layers(layers, "medium"))
        examples.append((stack_name, [("degraded", degraded), ("outermost first", build_sequence(outermost_first)(degraded)),
                                      ("innermost first", build_sequence(innermost_first)(degraded)),
                                      (f"searched: {candidate}"[:34], build_sequence(candidate.split(SEPARATOR))(degraded)), ("clear", clear)]))

    save_table(summary_rows, PROJECT / "results" / "tables" / "stacked_summary.csv")

    # --- summary chart: held-out PSNR of each strategy for each stack
    strategies = (("none", "none", "#888888"), ("best_single", "best single step", "#b05d0c"), ("innermost_first", "innermost first", "#2e7d4f"),
                  ("outermost_first", "outermost first", "#3d6299"), ("searched_best", "searched best", "#6a3d99"))
    figure, axis = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(summary_rows))
    for index, (key, label, colour) in enumerate(strategies):
        axis.bar(x + (index - 2) * 0.16, [r[key] for r in summary_rows], 0.16, label=label, color=colour)
    axis.set_xticks(x)
    axis.set_xticklabels([r["stack"] for r in summary_rows], fontsize=9)
    axis.set_ylabel("PSNR (dB) on held-out scenes, averaged over medium and heavy")
    axis.set_title("Stacked media: does the order of undoing the layers matter?")
    axis.legend(fontsize=8, ncol=5, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(PROJECT / "results" / "figures" / "stacked_summary.png", dpi=105)
    plt.close(figure)

    # --- example strips, one row per stack
    rows_of_tiles = []
    for stack_name, panels in examples:
        tiles = []
        for caption, picture in panels:
            tile = Image.fromarray(np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8))
            tile.thumbnail((190, 190))
            framed = Image.new("RGB", (tile.width, tile.height + 16), (30, 30, 30))
            framed.paste(tile, (0, 16))
            ImageDraw.Draw(framed).text((3, 2), (stack_name + ": " + caption if caption == "degraded" else caption)[:34], fill=(220, 220, 220))
            tiles.append(framed)
        rows_of_tiles.append(tiles)
    sheet = Image.new("RGB", (max(sum(t.width for t in r) for r in rows_of_tiles), sum(max(t.height for t in r) for r in rows_of_tiles)), (30, 30, 30))
    y = 0
    for tiles in rows_of_tiles:
        xpos = 0
        for tile in tiles:
            sheet.paste(tile, (xpos, y))
            xpos += tile.width
        y += max(t.height for t in tiles)
    sheet.save(PROJECT / "results" / "figures" / "stacked_examples.jpg", quality=90)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore saved progress and start this stage again")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
