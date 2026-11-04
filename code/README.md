# Code for the DIP project

Forward models for haze, water, glass and plastic (and a stack that chains them), the data tools that built `data/curated/`, and the checks that keep them honest. The demo page is in `../demo/`.

## One-time setup
```bash
cd Subjects/DIP/Projects/DIP-Project
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
```
(`.venv` is a local tool folder; never hand it in.)

## What is where
| Folder | Holds |
|---|---|
| `degrade/` | `haze.py`, `water.py`, `glass.py`, `plastic.py` (one model each, same `apply(clean, depth_map, params)` shape), `stack.py` (chain them, load or save a stack as JSON), `depth.py` (ramp and layered depth), `ops.py` (blur, smooth noise, mixing) |
| `common/` | `imageio.py` (load and save), `rng.py` (a random generator identical in Python and JavaScript), `filters.py` (Gaussian blur, FFT Gaussian, box filter, patch minimum, dark channel), `stats.py`, `datasets.py` (load picture pairs and clear scenes), `parallel.py` (run jobs on many cores within a memory budget, resumable), `rar_reader.py` (read the password-protected UIEB archive) |
| `data_tools/` | `select_curated.py` (haze, glass, water picks), `export_nyu.py` (depth frames), `curation.py` (shared helpers) |
| `tests/` | `test_degrade.py` (17 checks of the models), `test_restore.py` (74 checks of filters, restoration and metrics), `test_parallel.py` (11 checks of the parallel runner), `parity_check.py` (Python against `demo/synth.js`) |
| `restore/` | Stage 3 restoration: `dcp.py` (Dark Channel Prior from scratch), `medium_aware.py` (UDCP, local airlight, near-global transmission), `baselines.py` (white balance, CLAHE, Retinex, homomorphic, Wiener, equalisation, gamma, stretch), `extra.py` (techniques from the course notes, deblurring, sharpening, cleanup), `fusion.py` (pyramid fusion, Tarel), `methods.py` (registry of 38 methods and `build_sequence`) |
| `metrics/` | Stage 3b: `full_reference.py` (PSNR, SSIM, CIEDE2000), `no_reference.py` (entropy, contrast, UCIQE, dark channel, colour cast), `ranking.py` (do two experiments rank methods alike?) |
| `experiments/` | Stages 4 to 5: one script per stage plus `run_stages.py` that runs them in order |
| `../configs/` | `haze.json`, `water.json`, `glass.json`, `plastic.json` (which methods to run), `sweeps.json` (settings to sweep), `pipelines.json` (the step pool for the order search) |

## Commands (from the project root)
```bash
.venv/bin/python code/tests/test_degrade.py        # model checks, expect 17 of 17
.venv/bin/python code/tests/parity_check.py        # Python and the demo page agree, expect PASS (needs node)
.venv/bin/python code/data_tools/select_curated.py # rebuild the curated haze, glass and water sets (about 1 minute)
.venv/bin/python code/data_tools/export_nyu.py     # rebuild the 150 NYU depth frames (about 30 seconds)
.venv/bin/python code/degrade/make_matched_pairs.py # rebuild the 288 matched synthetic pairs
.venv/bin/python code/tests/test_restore.py         # 74 checks; and test_parallel.py (11 checks)
```

## Running the experiments (stage by stage, in parallel, resumable)
```bash
.venv/bin/python code/experiments/run_stages.py            # all stages, about 7 minutes with 16 workers
.venv/bin/python code/experiments/run_stages.py --quick    # a small version to check everything works (about 3 minutes)
.venv/bin/python code/experiments/run_stages.py --only sweeps,pipelines   # just some stages
.venv/bin/python code/experiments/run_stages.py --force pipelines         # repeat a stage (reuses its saved jobs)
.venv/bin/python code/experiments/run_stages.py --only real --fresh       # repeat a stage and discard its saved jobs
.venv/bin/python code/experiments/compare_real_vs_synthetic.py            # Stage 5 (needs the matched and real tables)
```
Stages run one after another and each waits for the previous. Inside a stage the work is spread over at most **16 worker processes (never more than 20)** and a memory budget of at most **16 GB**; if memory runs short the runner halves the workers by itself. Every finished job is saved in `results/checkpoints/`, so if a run is stopped, running the same command again continues where it left off. Each script also runs on its own, for example `run_experiments.py --medium haze --source matched`.

## Scripts added later (see `docs/stages.md` for the questions they answer)
```bash
.venv/bin/python code/degrade/make_matched_pairs.py --presets calibrated   # matched pairs with the presets fitted to real pictures (default); --presets original for the first set
DIP_PRESETS=calibrated .venv/bin/python code/experiments/run_stages.py     # run the preset-dependent stages on the calibrated presets (without it: original)
.venv/bin/python code/data_tools/make_tub_manifest.py                      # pair the TUB tank photos (data/existing/TUB) with their least-turbid reference
.venv/bin/python code/experiments/run_experiments.py --medium water --source tub [--methods-from haze]   # every method on the real tank pairs
.venv/bin/python code/experiments/run_local_vs_global.py [--tile-sweep]    # tile by tile against the whole picture (code/restore/local.py)
.venv/bin/python code/experiments/tune_milky_water.py                      # milky and inky water (code/degrade/milky.py), calibrated to the tank
.venv/bin/python code/experiments/run_subtraction_combos.py                # subtraction combined with the other tools
.venv/bin/python code/experiments/analyse_chain_limits.py                  # the mathematical ceilings for chains
.venv/bin/python code/experiments/run_longer_chains.py                     # chains of 1 to 5 steps, synthetic; run_longer_chains_real.py: real haze and tank water
.venv/bin/python code/experiments/make_report_tables.py && .venv/bin/python code/experiments/make_report_figures.py && .venv/bin/python code/experiments/make_water_figures.py && .venv/bin/python code/experiments/make_pipeline_figures.py   # then: cd report/final && pdflatex report.tex (twice)
.venv/bin/python code/tests/test_local.py                                  # 7 checks of the local-application wrapper
```
`results_original_presets/` holds the results of the first run (original presets), frozen for comparison with `results/`.

## Using a model from Python
```python
from common.imageio import load_image, save_image
from degrade.depth import ramp_depth
from degrade.stack import apply_stack

picture = load_image("some_clear_photo.jpg", max_side=800)
depth = ramp_depth(*picture.shape[:2])               # 0 = near, 1 = far
stack = [{"type": "water"}, {"type": "glass", "params": {"droplets": 40, "seed": 3}}]
save_image("result.png", apply_stack(picture, depth, stack))   # first layer is nearest the scene
```
The same stack (saved from the demo page as JSON, or by `degrade.stack.save_stack`) gives the same picture in both places.

## House rules these files follow
- The forward models are written by hand (NumPy only); SciPy is used only for the dark-channel patch minimum in `common/stats.py`.
- No machine learning and no learned depth anywhere. Libraries are used only for generic primitives the proposal allows: SciPy and scikit-image for PSNR, SSIM, CIEDE2000, CLAHE and histogram equalisation, a median filter and a morphological opening.
- Raw data in `data/existing/` and `data/depth/` is read-only and never edited; everything derived goes in `data/curated/`.
