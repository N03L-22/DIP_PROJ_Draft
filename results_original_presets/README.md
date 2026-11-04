# Results with the ORIGINAL (hand-chosen) presets — frozen 2026-11-04

A copy of `results/` taken **before** the calibrated presets were adopted for the experiments. Do not re-run into this folder.

| | Before (this folder) | After (`../results/`) |
|---|---|---|
| Presets | `PRESETS` in `code/degrade/presets.py` | `CALIBRATED_PRESETS` (run with `DIP_PRESETS=calibrated`) |
| Matched data | `data/synthetic/matched/` | `data/synthetic/matched_calibrated/` |
| Haze | veil-style fog, airlight 0.92, beta 0.5/1.2/2.4 | scene-depth fog, airlight 0.60-0.70, beta 1.3/2.2/5.5 |
| Glass | veil + smudge + droplets | raindrops only |
| Water, plastic | unchanged | unchanged |

`tables/*_matched_calibrated.csv` here is the first calibrated Stage 4 run (20 scenes per level, matched stage only; described as F48 in `../log.md`). The full calibrated rerun of every stage is in `../results/`.
The real-data tables (`*_real.csv`), model validation, tuning, scene-matched and frequency tables do not depend on the presets, so they are the same in both folders.
Compare a table by name: `results_original_presets/tables/X.csv` against `results/tables/X.csv`.
