#!/bin/bash
# Run every experiment of Stages 4 and 4b/4c in order. Takes roughly 20 to 30 minutes.
# Run from the project root:  bash code/experiments/run_all.sh
set -e
PY=.venv/bin/python
EXP=code/experiments

echo "== Stage 4: every method on the matched synthetic pairs, all four media =="
for medium in haze water glass plastic; do
  $PY $EXP/run_experiments.py --medium "$medium" --source matched --per-level 12
done

echo "== Stage 4: real photographs (haze and glass have a real reference, water has none) =="
$PY $EXP/run_experiments.py --medium haze  --source real --per-level 12
$PY $EXP/run_experiments.py --medium glass --source real --per-level 12
$PY $EXP/run_experiments.py --medium water --source real --per-level 12

echo "== Stage 4b: parameter sweeps (one setting at a time) =="
for medium in haze water glass plastic; do
  $PY $EXP/run_sweeps.py --medium "$medium" --level medium
done

echo "== Stage 4c: severity sweeps (how each method copes as the degradation grows) =="
for medium in haze water glass plastic; do
  $PY $EXP/run_severity_sweep.py --medium "$medium"
done

echo "== all experiments finished =="
