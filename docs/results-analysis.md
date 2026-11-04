---
type: note
kind: reference
title: DIP Project — what the experiments show (methods, parameters, step order, DCP)
---

# What the experiments show

Results of Stages 4 to 5 of [[Subjects/DIP/Projects/DIP-Project/docs/stages|the plan]], run on 2026-11-04. Every number comes from a table in `results/tables/` (named under each heading), produced by the code in `code/experiments/`. Findings and changes are logged in [[Subjects/DIP/Projects/DIP-Project/log|log]].

> [!warning] Read this first: how far to trust these numbers
> - **Synthetic pairs favour methods that undo the model that made them.** Glass and plastic are simulated as `image × t + constant`, which a plain contrast stretch inverts almost exactly. Water is simulated with a per-colour fade that colour-balancing methods undo. So "wins on synthetic data" is weaker evidence than "wins on real photographs". The proposal already warned about this for haze and DCP.
> - **Real data is small.** 36 real haze pairs, 36 real raindrop pairs, 890 real water photos with *no* reference. Plastic has no real data at all yet.
> - **Real glass here means raindrops**, not fog or dirt on a pane, so it only partly tests the glass model.
> - **Sweeps and the step-order search use 6 pictures at 256 to 320 px.** Small differences (under about 1 dB) are not reliable. The step-order verdicts use *held-out* pictures that the search never saw.
> - PSNR is a pixel-error score and can disagree with how a picture looks. Check the example strips in `results/figures/` before believing a single number.

## 1. The question: did anything beat DCP?

**Test:** for each medium, every ordered sequence of one to three steps from a pool of seven (259 sequences) was ranked on six pictures (set A). The best *multi-step* sequence was then compared with the DCP methods on six *different* pictures (set B) at all three severities (18 cases). Tables `pipelines_<medium>_verify.csv`, figures `pipelines_<medium>.png`.

| Medium | Best sequence (chosen on set A) | Its PSNR on held-out | Best DCP method | Its PSNR | No restoration | Verdict |
|---|---|---|---|---|---|---|
| **Haze** | `dcp_guided > unsharp` | 14.72 | `dcp_guided` | 15.26 | 10.45 | **No.** DCP alone is as good or better on these synthetic pairs (0 of 18 wins). |
| **Water** | `red_comp > von_kries > unsharp` | **17.98** | `wb_dcp` | 13.49 | 15.38 | **Yes, by 4.5 dB, 18 of 18 cases.** Every DCP method is *worse than doing nothing*. |
| **Glass** | `stretch > unsharp` | 22.74 | `dcp_near_global` | 22.72 | 18.41 | **A tie** (+0.02 dB, 10 of 18). A single plain `stretch` scored 26.9 dB. |
| **Plastic** | `wiener > unsharp > dcp_guided` | 17.25 | `wb_dcp` | 15.47 | 16.03 | **Yes, by 1.8 dB (14 of 18)**, but at *mild* severity doing nothing (21.8 dB) beats every restoration (17.1 dB). |

**On real haze photographs the answer is yes** (`haze_real.csv`, 36 pairs from O-HAZE, I-HAZE, Dense-Haze and NH-Haze): DCP followed by CLAHE (`dcp_clahe`) beats guided DCP by **+1.28 dB** (standard error 0.22) and **+0.092 SSIM**, better on 29 of 36 photographs (SSIM: 30 of 36). Guided DCP alone beats no restoration by +1.32 dB (22 of 36). So on real fog the winning idea is *DCP first, then a local-contrast step*, not a different algorithm.

**Order matters** (`pipelines_<medium>_order_effects.csv`, two-step sequences on the search pictures, only the order swapped):

| Medium | Biggest order effects |
|---|---|
| Haze | `tarel > gamma_auto` is **+2.7 dB** better than `gamma_auto > tarel`; `tarel > hist_eq` +2.1 dB; `dcp_guided > unsharp` +1.2 dB |
| Water | `von_kries > eq_hsi` +1.3 dB; `udcp > eq_hsi` is 1.1 dB *worse* than the reverse; `von_kries > dcp_guided` 1.0 dB worse than the reverse |
| Glass | `unsharp > stretch` is **3.1 dB worse** than `stretch > unsharp`; `dcp_local_airlight > top_hat` +1.8 dB; `clahe > unsharp` +1.1 dB |
| Plastic | `wiener > dcp_guided` +1.7 dB; `richardson_lucy > dcp_guided` +1.4 dB; `unsharp > dcp_guided` +1.2 dB |

*A plausible explanation (not separately tested):* steps that assume the picture follows a physical model (white balance, DCP, deblur) probably work best first, while the picture still looks like the physical situation; tone and contrast steps (CLAHE, gamma, stretch) reshape the picture and can break those assumptions. Sharpening probably works best late because it amplifies noise left by earlier steps. For `unsharp > stretch`, sharpening first pushes values to the ends of the range, so the stretch has less room to use.

## 2. Which single method works best

Mean PSNR over 36 matched synthetic pictures (12 per severity) and, where there is one, over the real photographs. Tables `<medium>_matched.csv` and `<medium>_real.csv`; example strips `<medium>_<source>_<level>_example.jpg`.

| Medium | Top on **synthetic** (mean dB) | DCP family (mean dB) | No restoration | Top on **real** |
|---|---|---|---|---|
| Haze | `dcp_guided` 15.6, `fusion_dcp` 15.0, `dcp` 15.0, `dcp_clahe` 14.9, `gamma_auto` 14.7 | `dcp_guided` 15.6, `dcp` 15.0 | 10.5 | `dcp_clahe` 15.0, `fusion_dcp` 15.0, `clahe` 14.2, `fusion` 13.9, `dcp_guided` 13.7 (none 12.4) |
| Water | `von_kries` 16.6, `eq_avg_hist` 15.4, `fusion` 15.3, `hist_eq` 15.3, `white_patch` 15.2, `red_comp` 15.2 | `wb_dcp` 11.4, `udcp` 10.1 | 14.1 | no reference; see below |
| Glass | `stretch` 28.5, `dcp_near_global` 22.5, `hist_eq` 19.8 | `dcp_near_global` 22.5, `dcp` 15.6 | 18.5 | **none** 24.2 (`freq_laplacian` 24.1, `dcp_near_global` 22.2, `stretch` 18.7) |
| Plastic | `stretch` 19.9, `hist_eq` 17.6, `clahe` 17.4, `eq_hsi` 17.0 | `wb_dcp` 13.2, `dcp` 12.7 | 15.8 | no real data |

**What this says:**
- **Haze:** the DCP family is clearly best on synthetic haze, and gamma and the histogram methods help too (13.6 to 14.7 dB against 10.5). Retinex and top-hat are *worse than doing nothing*. On the real photographs plain contrast methods (Tarel 12.1, global equalisation 11.6, stretch 11.2) fall *below* doing nothing (12.4), while DCP and CLAHE-based methods help. In the example strips the plain DCP output has a blue cast; CLAHE after it recovers contrast.
- **Water:** the winners are **colour fixes**, not dehazing: `von_kries`, `red_comp`, `white_patch`, and the histogram methods. DCP and UDCP are worse than doing nothing on this synthetic water. A likely reason (not tested): UDCP assumes the dark channel of green and blue is near zero, but the bright, colourful NYU indoor scenes break that, so much of what it removes as "veil" is scene.
- **Glass:** on the synthetic veil a contrast stretch is best, because it exactly undoes the model. **On real raindrops nothing helps**: not one method beats the untouched picture. This is the clearest example of the synthetic-versus-real gap.
- **Plastic:** contrast and tone methods help (stretch, equalisation, CLAHE). The deblurring methods give only small gains, and only at mild strength (mild severity: `unsharp` 22.7, `richardson_lucy` 21.6, `wiener` 20.8 against 21.1 for nothing); at heavy strength every deblur is about 10.7 to 10.9 dB, the same as nothing (10.7). The Wiener setting assumes one fixed blur strength, but the blur grows with severity.
- **Real water** has no reference, so only reference-free scores exist (`water_real.csv`): UCIQE is highest for `udcp` (0.422), `stretch` (0.418) and `eq_hsi` (0.413) against 0.310 untouched. But UCIQE rewards contrast and saturation, and UDCP has the *worst* PSNR on the synthetic water, so UCIQE and fidelity disagree. **Real water needs a visual check and, better, your own captures with a clear reference.**

## 3. What each setting does (parameter sweeps)

Tables `sweeps_<medium>.csv`, figures `sweeps_<medium>.png`. One setting changed at a time, medium severity, 6 pictures. "Spread" is the best PSNR minus the worst over the tested values; a large spread means the setting must be chosen with care.

| Method and setting | Spread | Best value | What happens (plain words) |
|---|---|---|---|
| haze: `dcp_guided` **omega** | 5.3 dB | 1.0 (16.5 dB) | omega is how much haze to remove. Low omega (0.5) leaves haze (11.3 dB); full removal is best on this synthetic haze. The original DCP paper uses 0.95 so far objects keep a little haze. |
| haze: `dcp_guided` **patch** | 4.8 dB | 15 (16.4 dB) | Tiny patches (3) wrongly remove bright scene objects; huge patches (101) cannot follow how haze changes across the picture. About 15 pixels balances these. |
| haze: `tarel` **strength** | 5.3 dB | 0.85 | Too weak leaves haze; at 1.0 it removes real scene and PSNR collapses (9.3 dB). A narrow sweet spot. |
| `clahe` **clip_limit** | up to 7.8 dB | 0.002 to 0.005 on water, glass, plastic; 0.1 on haze | The clip limit caps how hard local contrast is stretched; high limits amplify noise and veil texture. On haze the curve is irregular (6.8 dB at 0.005 but 10.6 at 0.01): with 6 pictures, treat it as noisy. |
| `gamma_auto` **target** | 4.0 to 5.6 dB | 0.4 (0.5 ties on plastic) | A mid-grey target (0.5) is not best; slightly dark wins on hazy, watery and milky pictures because brightening everything washes out contrast. |
| water: `udcp` and `wb_dcp` **patch** | 5.4 and 4.4 dB | 61 (largest tested) | Water wants *large* patches: with small ones the dark channel reads bright scene colours as veil. Still below doing nothing at medium strength. |
| water: `red_comp` **strength** | 2.9 dB | 2.0 (16.8 dB, highest tested) | More red restored keeps helping; the best value is at the edge of the range, so the true optimum is higher than tested. |
| glass: `top_hat` **size** | 9.7 dB | 0.4 (largest tested) | The window must be bigger than the biggest blotch of film; small windows destroy the picture (8.3 dB). Best is again at the edge: needs larger values. |
| glass: `homomorphic` **low_gain** | 7.0 dB | 0.9 | Cutting the slow waves hard (low gain 0.2) throws away real scene brightness. Gentle is better. |
| glass: `homomorphic` **high_gain** | 6.0 dB | 1.0 | Boosting fine detail (above 1) amplifies the film's own texture. |
| glass: `dcp_near_global` **patch_fraction** | 4.2 dB | 0.25 | A patch about a quarter of the picture gives a near-constant transmission, which suits a film on glass. |
| plastic: `wiener` **noise_to_signal** | 4.8 dB | 0.1 | Anything from 0.0001 to 0.1 is about the same (16.1 to 16.4 dB); at 1.0 it over-smooths and collapses (11.6 dB). |
| plastic: `wb_dcp` **omega** | 4.4 dB | 0.5 | On plastic DCP should remove *less*: full removal makes it worse (13.7 dB). |
| plastic: `unsharp` amount, `stretch` percentile | under 1.8 dB | any | Not sensitive. |

**Patterns across the media:**
1. *Gentle beats aggressive* wherever the degradation is not distance-dependent haze: low CLAHE clip limit, low homomorphic gains, low DCP omega on plastic.
2. *Patch or window size should match the size of the thing being removed*: about 15 px for haze, 61 or more for water, a quarter of the picture for a film.
3. Several best values sit at the *edge* of the tested range (water `red_comp` strength, glass `top_hat` size, water DCP patch), so the true optimum is beyond. These sweeps need wider ranges before a final value is quoted.
4. For several methods (CLAHE, gamma, water DCP patch, the homomorphic gains, plastic `wb_dcp` omega) the default setting used in the Stage 4 tables is 2 to 6 dB below the best tested value. For guided DCP on haze the defaults (patch 15, omega 0.95) are already near the best (16.4 against 16.5 dB).

## 4. How the methods cope as the degradation grows

Tables `severity_<medium>.csv`, figures `severity_<medium>.png`: PSNR at 0.25× to 3× the medium strength.

- **Haze:** guided DCP is best up to about 1× the medium strength (22.2 dB at 0.25×, 15.1 at 1×). By 1.5× **Tarel overtakes it** (13.9 against 12.2 dB) and at 3× it is far better (12.3 against 8.0 dB): DCP degrades fast as haze thickens, while Tarel's median-based veil estimate degrades slowly. Tarel is worse than no restoration only at the lightest haze (0.25×). *Note:* on the real photographs Tarel was below doing nothing (12.1 against 12.4 dB), so the thick-fog advantage is shown on synthetic haze only.
- **Water:** `von_kries` and `red_comp` improve on no restoration at every strength above the lightest, and the gain *holds* with strength (von Kries: +2.6 dB at 1×, +2.4 dB at 3×). UDCP and wb_dcp are below no restoration at *every* strength.
- **Glass:** `stretch` and `dcp_near_global` stay above no restoration at every strength, and the gap is large (stretch: +11 dB at 1×, +6 dB at 3×). Histogram equalisation and homomorphic filtering help only at 1× and above.
- **Plastic:** `stretch` gives large gains at every strength; `unsharp` is never below doing nothing but gains at most about 1.5 dB. Wiener and Richardson-Lucy gain at most 0.7 dB at 1× or above and lose to doing nothing at the lightest strength; `wb_dcp` and equalisation are below doing nothing at the two lightest strengths.

## 5. Can synthetic data stand in for real data? (Stage 5)

`ranking_agreement.csv`: how alike the method rankings are on synthetic and real pictures (Spearman: +1 identical order, 0 unrelated, -1 reversed).

| Medium (score) | All levels | Mild | Medium | Heavy | Best on synthetic | Best on real |
|---|---|---|---|---|---|---|
| Haze (PSNR) | **+0.53** | +0.31 | +0.36 | +0.63 | `dcp_guided` | `dcp_clahe` |
| Glass (PSNR) | **+0.53** | +0.82 | +0.31 | **-0.19** | `stretch` | none (nothing helps) |
| Water (UCIQE) | +0.85 | +0.88 | +0.83 | +0.85 | `udcp` | `udcp` |

- **Haze** agrees only moderately: the DCP methods are on top in both worlds, but the exact winner differs and the light-haze ranking is weak (+0.31).
- **Glass** agrees only moderately overall and **reverses at heavy severity**: the synthetic film does not behave like raindrops.
- **Water's high agreement is misleading.** It compares UCIQE with UCIQE; UCIQE ranks UDCP first on both but PSNR ranks it last on the synthetic pictures. The agreement shows the *metric* is consistent, not that the methods are good.
- **Conclusion for the report:** use synthetic data to sweep severity and parameters (where it is the only practical option), but base any "recommended pipeline" on real photographs, and report this gap openly.

## 6. What this suggests for the recommended pipelines (tentative)

| Medium | Starting recommendation | Evidence level |
|---|---|---|
| Haze | guided DCP, then CLAHE; consider Tarel only for thick synthetic-style fog | **Real photographs** (+1.28 dB, 29 of 36) for DCP then CLAHE |
| Water | colour correction first (`red_comp` then `von_kries`), then gentle sharpening; do not use plain DCP | Synthetic only (held-out, 18 of 18) and *model-favoured*; **needs real captures** |
| Glass | stretch or near-global DCP for a veil; nothing helps raindrops here | Veil: synthetic only. Raindrops: real, but only "do nothing" |
| Plastic | contrast stretch; deblur only for mild blur with the blur strength known | Synthetic only; **no real data** |

## 7. Files and how this was produced

| What | Where |
|---|---|
| Code, one stage each | `code/experiments/run_experiments.py` (4), `run_sweeps.py` (4b), `run_severity_sweep.py` (4c), `run_pipelines.py` (4d), `compare_real_vs_synthetic.py` (5); `run_stages.py` runs them in order |
| Methods | `code/restore/` (`dcp.py`, `medium_aware.py`, `baselines.py`, `extra.py`, `fusion.py`, registry in `methods.py`), settings in `configs/` |
| Tables and figures | `results/tables/`, `results/figures/` |
| Speed | The whole run took **440 s** with 16 workers (matched 170 s, real 103 s, sweeps 38 s, severity 32 s, pipelines 96 s). A benchmark of one stage gave **59 s serial against 9 s with 16 workers (6.6 times faster)**. Memory peaked at 7 GB of the 16 GB budget with no back-off. An interrupted run resumes from `results/checkpoints/`. |

## 9. Later findings (2026-11-04; the numbers and caveats are in `log.md`, F48 to F60)
- **Calibrated presets (F48 to F52).** The original haze and glass presets flattered DCP and stretch. With presets fitted to real pictures, DCP is barely better than doing nothing on synthetic haze (11.74 against 11.56 dB), nothing beats doing nothing on raindrops, and glass agrees with real data (rank correlation 0.92). The apparent fall in haze rank agreement (0.53 to 0.09) was small-sample noise: on 90 scenes it is +0.49 (original) against +0.36 (calibrated), and outdoor scenes (RESIDE) give +0.08 (F61). The before-calibration results are frozen in `results_original_presets/`.
- **Real underwater data (F55, F56).** TUB tank pairs (1,183): nothing beats doing nothing by more than 0.1 dB and DCP is 7.3 dB below it. UIEB ocean photos have no reference; by eye, colour correction is best.
- **Milky and inky water (F57).** Synthetic water made milky (matching the tank) reproduces the real verdict; ink (a black cloud) is helped by brightening methods and not by DCP.
- **Global against local (F53, F54).** Local application helps haze (tiles of about 4 to 6 per side; real histogram equalisation +1.95 dB), mixed for water, hurts glass. The best tile size differs by medium.
- **Subtraction (F58).** Local-glow subtraction alone ties DCP then CLAHE on real haze; no combination with subtraction beat the best chain without it.
- **Chain length (F59, F60).** The damage is invertible in 1 to 2 steps; chains of 4 and 5 steps do not help on synthetic or real pictures. Useful range: 2 to 3 steps for haze, 2 for tank water, 1 to 3 for plastic, none for glass.
- **Full write-up with figures:** `report/final/report.pdf`; equations: `docs/model-equations.md`.

## 8. Still to do
- Capture real water, glass (fog and dirt), plastic and stacked pictures with a clear reference: the main gap in the evidence.
- Re-run the sweeps over wider ranges where the best value sat at the edge (water `red_comp`, glass `top_hat`, water DCP patch).
- Stage 6: failure analysis with dark-channel statistics and transmission maps, and the stacked-layer experiment (`degrade/stack.py` is ready).
- Report each result with its caveat from the box at the top.

Back to [[Subjects/DIP/Projects/DIP-Project/DIP-Project|DIP-Project]] · [[Subjects/DIP/Projects/DIP-Project/docs/stages|stages]] · [[Subjects/DIP/Projects/DIP-Project/data/README|data]]
