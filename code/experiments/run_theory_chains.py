"""STAGE 4l - chains DERIVED FROM THE INVERSE EQUATIONS, and a check of which of them the earlier searches had already tried.

The damage of each medium is (blur then) a per-colour affine map, I_c = a_c (G * J)_c + b_c, with a_c (the transmission) and b_c = A_c (1 - a_c) (the glow); a_c depends on depth for haze and water.
The inverse is therefore a short list of operations, and each operation has a function that approximates it:
   undo the glow and the gain, per colour, estimated from the picture   -> stretch_rgb (a separate range per colour); white balance (gray_world, von_kries) for the gain only
   undo the glow and the gain per pixel, with a depth-dependent t       -> dcp_guided, tarel, blind_veil_local (they estimate t(x) and A, then divide)
   tidy the range left over after a division by t                       -> stretch, gamma_auto
   undo the blur                                                        -> wiener, richardson_lucy (with the blur width; 'matched' = the true width of the severity)
   undo a slow film                                                     -> flat_field
Chains are built from those functions in the order the inverse asks for (the layer applied last is undone first). Each is scored on held-out pictures next to the best chains of
the earlier searches, the oracle ceilings and doing nothing. SYNTHETIC: scenes 6-11 (set B) and 12-23 (set C), three severities, 8-bit pictures, the calibrated presets;
REAL: real haze (sets B and C of run_longer_chains_real.py) and real tank water.
Output: results/tables/theory_chains_*.csv  theory_chains_summary.csv  results/figures/theory_chains.png     One job = one picture, parallel and resumable.
Run from the project root:  .venv/bin/python code/experiments/run_theory_chains.py
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
from common.imageio import load_image  # noqa: E402
from common.parallel import fingerprint, run_stage  # noqa: E402
from degrade import glass, haze, plastic, water  # noqa: E402
from degrade.presets import CALIBRATED_PRESETS, LEVELS  # noqa: E402
from experiments.analyse_chain_limits import oracle_affine, oracle_lut, psnr, quantise, true_inverse  # noqa: E402
from experiments.run_longer_chains_real import pair_sets  # noqa: E402
from restore.methods import REGISTRY, build_sequence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES, FIGURES = PROJECT / "results" / "tables", PROJECT / "results" / "figures"
SEP = " > "
MODELS = {"haze": haze, "water": water, "glass": glass, "plastic": plastic}
BLUR = {"mild": 0.003, "medium": 0.006, "heavy": 0.012}  # the true blur of each plastic severity (the 'matched' deblur)

THEORY = {
    "haze": ["dcp_guided > stretch", "gray_world > dcp_guided", "gray_world > dcp_guided > stretch", "dcp_guided > gray_world", "dcp_guided > gray_world > stretch_rgb",
             "tarel > stretch", "blind_veil_local > stretch", "stretch_rgb > dcp_guided", "dcp_guided > gamma_auto"],
    "water": ["stretch_rgb", "gray_world > stretch_rgb", "red_comp > stretch_rgb", "stretch_rgb > von_kries", "udcp > stretch_rgb", "red_comp > von_kries > stretch_rgb", "von_kries > stretch_rgb"],
    "plastic": ["stretch > wiener", "wiener > stretch", "stretch_rgb > wiener", "wiener > stretch_rgb", "stretch_rgb > richardson_lucy", "stretch_rgb > wiener_matched", "wiener_matched > stretch_rgb"],
    "glass": ["flat_field > stretch", "stretch > flat_field", "flat_field > stretch_rgb", "stretch_rgb"],
}
REAL_SETS = {"haze": "haze", "water": "water"}


def spec(name, level):
    """Steps of a chain as specs; 'wiener_matched' is the Wiener filter with the true blur width of the severity."""
    steps = []
    for step in name.split(SEP):
        steps.append({"name": "wiener", "params": {"blur_sigma": BLUR[level]}} if step == "wiener_matched" else {"name": step})
    return steps


def work(job):
    medium, level = job["medium"], job["level"]
    chains = THEORY[medium] + job["searched"]
    row = {"world": job["world"], "medium": medium, "level": level, "picture": job["picture"]}
    if job["world"] == "synthetic":
        clear, depth, _ = load_clear_scenes(24, 256, only=job["index"])[0]
        params = CALIBRATED_PRESETS[medium][level]
        degraded = quantise(MODELS[medium].apply(clear, depth if medium != "glass" else None, params))
        row["oracle_affine"], row["oracle_lut"] = psnr(oracle_affine(degraded, clear), clear), psnr(oracle_lut(degraded, clear), clear)
        inverse = true_inverse(medium, degraded, depth, params)
        row["true_inverse"] = float("nan") if inverse is None or medium == "plastic" else psnr(inverse, clear)
    else:
        clear, degraded = load_image(job["reference"], 256), load_image(job["degraded"], 256)
    row["none"] = psnr(degraded, clear)
    for name in chains:
        row[name] = psnr(build_sequence(spec(name, level))(degraded), clear)
    return [row]


def save_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def run(workers=None, resume=True):
    searched = {r["medium"]: r["chains"].split(" | ") for r in csv.DictReader(open(TABLES / "chain_limits_summary.csv"))}
    real_best = {"haze": ["dcp_guided > gray_world > clahe", "dcp_guided > clahe"], "water": ["red_comp > unsharp", "unsharp"]}
    assert all(s in REGISTRY for chain in sum(THEORY.values(), []) for s in chain.split(SEP) if s != "wiener_matched")
    summary, store = [], {}
    for world, media in (("synthetic", ("haze", "water", "plastic", "glass")), ("real", ("haze", "water"))):
        for medium in media:
            jobs = []
            if world == "synthetic":
                for index in range(6, 24):
                    for level in LEVELS:
                        jobs.append({"key": f"{medium}|{index}|{level}", "world": world, "medium": medium, "level": level, "index": index, "picture": f"scene{index}",
                                     "searched": searched[medium]})
            else:
                sets = pair_sets(medium)
                for s in ("B", "C"):
                    for pair in sets[s]:
                        jobs.append({"key": f"{medium}|{s}|{pair['id']}", "world": world, "medium": medium, "level": "real", "picture": f"{s}|{pair['id']}", "degraded": pair["degraded"],
                                     "reference": pair["reference"], "searched": real_best[medium]})
            rows = run_stage(f"theory_{world}_{medium}", fingerprint(THEORY[medium], world, medium, len(jobs)), jobs, work, workers=workers, memory_per_worker_gb=1.0, resume=resume)
            save_csv(rows, TABLES / f"theory_chains_{world}_{medium}.csv")
            store[(world, medium)] = rows
            names = ["none"] + THEORY[medium] + (searched[medium] if world == "synthetic" else real_best[medium])
            reference = max(searched[medium] if world == "synthetic" else real_best[medium], key=lambda n: np.mean([r[n] for r in rows]))
            print(f"\n{world} {medium}: {len(rows)} pictures; reference = best earlier chain: {reference} ({np.mean([r[reference] for r in rows]):.2f} dB)")
            for name in dict.fromkeys(names):
                values = np.array([r[name] for r in rows])
                gain = values - np.array([r[reference] for r in rows])
                tag = "THEORY" if name in THEORY[medium] else "earlier" if name != "none" else "       "
                print(f"  {tag} {name:42} {values.mean():6.2f} dB   vs best earlier chain {gain.mean():+.2f} +- {gain.std(ddof=1) / np.sqrt(len(gain)):.2f}")
                summary.append({"world": world, "medium": medium, "chain": name, "kind": tag.strip().lower(), "psnr": round(float(values.mean()), 3), "gain_vs_best_earlier": round(float(gain.mean()), 3),
                                "se": round(float(gain.std(ddof=1) / np.sqrt(len(gain))), 3), "pictures": len(rows)})
            if world == "synthetic":
                for key in ("oracle_affine", "oracle_lut", "true_inverse"):
                    vals = [r[key] for r in rows if r.get(key) == r.get(key)]
                    if vals:
                        print(f"  ceiling {key:36} {np.mean(vals):6.2f} dB")
                        summary.append({"world": world, "medium": medium, "chain": key, "kind": "ceiling", "psnr": round(float(np.mean(vals)), 3), "pictures": len(vals)})
    save_csv(summary, TABLES / "theory_chains_summary.csv")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, resume=not arguments.fresh)
