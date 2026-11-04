"""STAGE 5 - can synthetic data stand in for real data? Compare how the two rank the restoration methods.

For each medium that has both kinds of pictures we average each method's score over the matched synthetic pairs and over the
real photographs, then ask how alike the two rankings are (Spearman and Kendall: +1 identical order, 0 unrelated, -1 reversed).
Haze and glass are compared on PSNR (both have a real reference). Water has no real reference, so it is compared on the
reference-free UCIQE score instead. A low agreement means conclusions drawn from synthetic data must not be trusted for real pictures.

Needs the tables written by run_experiments.py (matched and real). Output: results/tables/ranking_agreement.csv
Run from the project root:  .venv/bin/python code/experiments/compare_real_vs_synthetic.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.ranking import ranking_agreement  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TABLES = PROJECT / "results" / "tables"
SCORE_USED = {"haze": "psnr", "glass": "psnr", "water": "uciqe"}  # which score the rankings are built from


def method_means(medium, source, score, level=None):
    """Average `score` over the pictures for every method: {method: mean}. Optionally only one severity level."""
    with open(TABLES / f"{medium}_{source}.csv") as handle:
        rows = [r for r in csv.DictReader(handle) if level is None or r["level"] == level]
    methods = dict.fromkeys(r["method"] for r in rows)
    return {m: float(np.mean([float(r[score]) for r in rows if r["method"] == m])) for m in methods}


def main():
    """Compare the rankings for haze, glass and water and write results/tables/ranking_agreement.csv."""
    output = []
    for medium, score in SCORE_USED.items():
        for level in (None, "mild", "medium", "heavy"):
            synthetic, real = method_means(medium, "matched", score, level), method_means(medium, "real", score, level)
            shared = [m for m in synthetic if m in real]  # the same methods in the same order, so the two lists line up
            agreement = ranking_agreement([synthetic[m] for m in shared], [real[m] for m in shared])
            best_synthetic, best_real = max(shared, key=synthetic.get), max(shared, key=real.get)
            output.append({"medium": medium, "level": level or "all", "score": score, "methods": len(shared),
                           "spearman": round(agreement["spearman"], 3), "kendall": round(agreement["kendall"], 3),
                           "best_on_synthetic": best_synthetic, "best_on_real": best_real})
            print(f"{medium:8}{(level or 'all'):8} ranking agreement  Spearman {agreement['spearman']:+.2f}  Kendall {agreement['kendall']:+.2f}   "
                  f"best synthetic: {best_synthetic:16} best real: {best_real}")

    with open(TABLES / "ranking_agreement.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {(TABLES / 'ranking_agreement.csv').relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
