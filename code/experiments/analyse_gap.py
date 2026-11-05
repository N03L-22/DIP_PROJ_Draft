"""STAGE 4j - why is the measured score so far below the best possible? Replace what DCP has to ESTIMATE by the TRUE value, one piece at a time (SYNTHETIC haze).

DCP must estimate two things from the hazy picture alone: the glow colour A and the transmission t(x) per pixel. On synthetic haze both are known, so
   L1  DCP as is (estimates both)
   L2  DCP with the TRUE glow A (estimates t only)
   L3  the TRUE t(x) with the ESTIMATED glow (J = (I - A_est)/max(t, t0) + A_est)
   L4  the TRUE t(x) and the TRUE glow (the best possible, with the same floor t0)
   L5  L4 without the floor t0 (the true inverse, limited only by 8-bit rounding)
Each step up shows how much one estimate costs. 18 held-out pictures (scenes 6 to 11, three severities), calibrated haze.
Output: results/tables/gap_haze.csv   Run from the project root:  .venv/bin/python code/experiments/analyse_gap.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.datasets import load_clear_scenes  # noqa: E402
from common.filters import dark_channel  # noqa: E402
from degrade import haze  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, LEVELS  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from restore.dcp import dcp_restore, estimate_airlight, recover  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
T0 = 0.1


def psnr(a, b):
    return full_reference_scores(np.clip(a, 0, 1), b)["psnr"]


def main():
    rows = []
    for index in range(6, 12):
        clear, depth, scene = load_clear_scenes(24, 256, only=index)[0]
        for level in LEVELS:
            params = {**haze.DEFAULTS, **CALIBRATED_PRESETS["haze"][level]}
            hazy = np.round(np.clip(haze.apply(clear, depth, params), 0, 1) * 255) / 255
            true_a = np.full(3, params["airlight"])
            true_t = haze.transmission(depth, params)[:, :, 0]
            est_a = estimate_airlight(hazy, dark_channel(hazy, 15))
            rows.append({"level": level, "scene": scene, "none": psnr(hazy, clear),
                         "L1_dcp": psnr(dcp_restore(hazy, refine=True)["restored"], clear),
                         "L2_true_A": psnr(dcp_restore(hazy, refine=True, airlight=true_a)["restored"], clear),
                         "L3_true_t_est_A": psnr(recover(hazy, true_t, est_a, T0), clear),
                         "L4_true_t_true_A_floor": psnr(recover(hazy, true_t, true_a, T0), clear),
                         "L5_true_inverse": psnr((hazy - true_a) / np.maximum(true_t, 1e-6)[:, :, None] + true_a, clear),
                         "A_error": float(np.mean(np.abs(est_a - true_a)))})
    with open(PROJECT / "results" / "tables" / "gap_haze.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    keys = ["none", "L1_dcp", "L2_true_A", "L3_true_t_est_A", "L4_true_t_true_A_floor", "L5_true_inverse"]
    print("mean PSNR (dB), calibrated synthetic haze, 18 held-out pictures")
    for level in LEVELS + ("all",):
        own = [r for r in rows if level == "all" or r["level"] == level]
        print(f"  {level:7}" + "  ".join(f"{k}: {np.mean([r[k] for r in own]):5.2f}" for k in keys) + f"   mean |A error| {np.mean([r['A_error'] for r in own]):.2f}")


if __name__ == "__main__":
    main()
