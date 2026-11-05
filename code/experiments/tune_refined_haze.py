"""STAGE 2i - calibrate the REFINED haze model (degrade/haze_refined.py) to the real haze pictures, as tune_models.py did for the first model.

For each real severity (mild, medium, heavy) pick the density beta and the blur whose synthetic signature (dark shift, contrast, brightness, changed share, sharpness)
is nearest to the mean signature of the real pairs. 12 NYU scenes. Output: configs/refined_haze_presets.json, results/tables/refined_haze_tuning.csv
Run from the project root:  .venv/bin/python code/experiments/tune_refined_haze.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from degrade import haze_refined  # noqa: E402
from experiments.tune_models import BETAS, LEVELS, METRICS, distance, real_targets  # noqa: E402
from experiments.validate_models import signature  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
BLURS = [0.0, 0.008, 0.013, 0.02]


def main():
    targets, spread = real_targets("haze")
    scenes = load_clear_scenes(12, 256)
    rows = []
    for beta in BETAS:
        for blur in BLURS:
            sigs = [signature(np.clip(haze_refined.apply(clear, depth, {"beta": beta, "blur": blur}), 0, 1), clear) for clear, depth, _ in scenes]
            rows.append({"beta": beta, "blur": blur, **{m: float(np.mean([s[m] for s in sigs])) for m in METRICS}})
    presets, out = {}, []
    for level in LEVELS:
        best = min(rows, key=lambda r: distance(r, targets[level], spread))
        presets[level] = {"beta": best["beta"], "blur": best["blur"]}
        out.append({"level": level, "beta": best["beta"], "blur": best["blur"], "distance": round(distance(best, targets[level], spread), 3),
                    **{f"synthetic_{m}": round(best[m], 3) for m in METRICS}, **{f"real_{m}": round(float(v), 3) for m, v in zip(METRICS, targets[level])}})
        print(f"  {level:7} beta {best['beta']} blur {best['blur']}  distance {out[-1]['distance']}")
    (PROJECT / "configs" / "refined_haze_presets.json").write_text(json.dumps(presets, indent=2))
    with open(PROJECT / "results" / "tables" / "refined_haze_tuning.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print("compare: the first calibrated haze had distances 0.34 / 0.23 / 0.17 (scene structure, tuning_best.csv)")


if __name__ == "__main__":
    main()
