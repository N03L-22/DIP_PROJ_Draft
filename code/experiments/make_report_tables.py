"""STAGE 7b - turn the result tables (CSV) into LaTeX tables for the report, so no number is copied by hand.

Every table the report shows is written to report/final/tables/<name>.tex and pulled in with \\input. Re-run this after any
experiment is re-run, then recompile the report.
Run from the project root:  .venv/bin/python code/experiments/make_report_tables.py
"""
import csv
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
TABLES = PROJECT / "results" / "tables"
OUT = PROJECT / "report" / "final" / "tables"
MEDIA = ("haze", "water", "glass", "plastic")


BEFORE = PROJECT / "results_original_presets" / "tables"  # the same tables from the run with the original presets


def read(name, folder=None):
    with open((folder or TABLES) / name) as handle:
        return list(csv.DictReader(handle))


def tex(text):
    return str(text).replace("_", r"\_").replace(">", r"$>$").replace("&", r"\&").replace("%", r"\%")


def write_table(name, header, rows, spec):
    """header: list of column titles; rows: list of lists of strings; spec: tabular column spec."""
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines))


def mean_by(rows, key, fields):
    """{key value: [mean of each field]} over the rows."""
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append([float(r[f]) for f in fields])
    return {k: np.mean(v, axis=0) for k, v in groups.items()}


def method_table(name, rows, top=6):
    """Top methods by mean PSNR next to 'none', with SSIM and CIEDE2000."""
    means = mean_by(rows, "method", ["psnr", "ssim", "ciede2000"])
    order = sorted(means, key=lambda m: -means[m][0])
    shown = order[:top] + (["none"] if "none" not in order[:top] else [])
    body = [[tex(m), f"{means[m][0]:.2f}", f"{means[m][0] - means['none'][0]:+.2f}", f"{means[m][1]:.3f}", f"{means[m][2]:.1f}"] for m in shown]
    write_table(name, ["method", "PSNR dB", "vs none", "SSIM", r"CIEDE2000 $\downarrow$"], body, "lrrrr")


def main():
    for medium in MEDIA:
        method_table(f"matched_{medium}", read(f"{medium}_matched.csv"))
    for medium in ("haze", "glass"):
        method_table(f"real_{medium}", read(f"{medium}_real.csv"))
    method_table("real_water_tub", read("water_tub.csv"), top=4)
    method_table("real_water_tub_hazemethods", read("water_tub_hazemethods.csv"), top=4)
    # real tank water against synthetic water: do the 17 water methods rank alike? (per turbidity band and overall)
    from scipy.stats import spearmanr
    body = []
    for band in ("all", "mild", "medium", "heavy"):
        real_rows = [r for r in read("water_tub.csv") if band == "all" or r["level"] == band]
        syn_rows = [r for r in read("water_matched.csv") if band == "all" or r["level"] == band]
        real_means, syn_means = mean_by(real_rows, "method", ["psnr"]), mean_by(syn_rows, "method", ["psnr"])
        names = list(real_means)
        body.append([band, str(len({r["id"] for r in real_rows})), f"{real_means['none'][0]:.2f}", f"{max(real_means, key=lambda m: real_means[m][0]).replace('_', chr(92) + '_')}",
                     f"{spearmanr([real_means[n][0] for n in names], [syn_means[n][0] for n in names])[0]:+.2f}"])
    write_table("water_tub_agreement", ["turbidity band", "pairs", "none (PSNR)", "best method on real", "rank agreement with synthetic"], body, "lrrlr")

    # held-out verification of the searched pipelines: best sequence, DCP, nothing
    body = []
    for medium in MEDIA:
        rows = read(f"pipelines_{medium}_verify.csv")
        means = mean_by(rows, "pipeline", ["psnr", "ssim"])
        best = max(means, key=lambda p: means[p][0])
        single = {p: v for p, v in means.items() if " > " not in p}
        dcp = single.get("dcp_guided")
        base = single.get("none")
        body.append([medium, tex(best), f"{means[best][0]:.2f}", f"{means[best][1]:.3f}",
                     "--" if dcp is None else f"{dcp[0]:.2f}", "--" if base is None else f"{base[0]:.2f}"])
    write_table("pipelines_verify", ["medium", "best sequence (held-out)", "PSNR", "SSIM", "DCP alone", "no restoration"], body, "llrrrr")

    # before (original presets) against after (calibrated presets): best method, DCP and doing nothing on the matched pairs, and the searched chain
    body = []
    for medium in ("haze", "glass"):
        for tag, folder in (("original", BEFORE), ("calibrated", TABLES)):
            means = mean_by(read(f"{medium}_matched.csv", folder), "method", ["psnr"])
            best = max(means, key=lambda m: means[m][0])
            chain = mean_by(read(f"pipelines_{medium}_verify.csv", folder), "pipeline", ["psnr"])
            best_chain = max(chain, key=lambda p: chain[p][0])
            agreement = next(r for r in read("ranking_agreement.csv", folder) if r["medium"] == medium and r["level"] == "all" and r["score"] == "psnr")
            dcp = means.get("dcp_guided", [float("nan")])[0] if medium == "haze" else float("nan")
            body.append([medium, tag, tex(best), f"{means[best][0]:.2f}", "--" if dcp != dcp else f"{dcp:.2f}", f"{means['none'][0]:.2f}",
                         tex(best_chain), f"{chain[best_chain][0]:.2f}", f"{float(agreement['spearman']):+.2f}"])
    write_table("before_after_presets", ["medium", "presets", "best single", "PSNR", "guided DCP", "none", "best chain (held-out)", "PSNR", "rank agreement"], body, "llllrrlrr")

    # signature of the degradation: real pictures against original and calibrated synthetic pictures, per severity (dark shift, contrast ratio, brightness shift, sharpness ratio)
    fields = ("dark_shift", "contrast_ratio", "brightness_shift", "sharpness_ratio")
    original = {}
    for r in read("model_signature.csv", BEFORE):
        if r["world"] == "synthetic":
            original.setdefault((r["medium"], r["group"]), []).append([float(r[f]) for f in fields])
    body = []
    for r in read("tuning_best.csv"):
        if r["medium"] == "glass" or (r["medium"] == "haze" and r["structure"] == "scene"):
            old = np.mean(original[(r["medium"], r["level"])], axis=0)
            for tag, values in (("real", [r["real_" + f] for f in fields]), ("synthetic, original", old), ("synthetic, calibrated", [r["synthetic_" + f] for f in fields])):
                body.append([r["medium"], r["level"], tag] + [f"{float(v):.2f}" for v in values])
    write_table("signature_compare", ["medium", "level", "pictures", "dark shift", "contrast ratio", "brightness shift", "sharpness ratio"], body, "lllrrrr")

    rows = [r for r in read("outdoor_haze_summary.csv") if (r["level"] == "all") != (r["world"] == "reside")]
    names = {"nyu_original": "indoor rooms, original haze (synthetic)", "nyu": "indoor rooms, calibrated haze (synthetic)", "reside": "outdoor, RESIDE-made haze (synthetic)", "diode": "outdoor, DIODE depth, our haze (synthetic)"}
    write_table("outdoor_haze", ["pictures", "n", "Spearman vs real haze", "best method", "PSNR", "guided DCP", "none"],
                [[names[r["world"]], r["pictures"], f"{float(r['spearman']):+.2f}", tex(r["best_method"]), r["best_psnr"], r["dcp_guided"], r["none"]] for r in rows], "lrrlrrr")

    rows = read("agreement_stability.csv")
    write_table("agreement_stability", ["comparison", "scenes / pairs", "first estimate (90\\% interval)", "with 90 synthetic scenes (90\\% interval)"],
                [[r["comparison"], (f"{r['synthetic_scenes_before']} / {r['real_pairs']}"), f"{float(r['point_before']):+.2f} ({r['ci_before']})",
                  (f"{float(r['point_now']):+.2f} ({r['ci_now']})" if r["point_now"] else "--")] for r in rows], "lrll")

    # chain-length analysis: ceilings (Stage 4g) and the curve of the best chain per length (Stage 4h)
    rows = read("chain_limits_summary.csv")
    f = lambda r, k, d=2: "--" if r[k] in ("nan", "") else f"{float(r[k]):.{d}f}"  # noqa: E731
    write_table("chain_limits", ["medium", "none", "1 step", "2 steps", "3 steps", "oracle LUT", "oracle affine", "true inverse", "predicted inverse"],
                [[r["medium"]] + [f(r, k) for k in ("none", "chain1", "chain2", "chain3", "oracle_lut", "oracle_affine", "true_inverse", "predicted_inverse")] for r in rows], "lrrrrrrrr")
    rows = read("longer_summary.csv")
    write_table("longer_chains", ["medium", "set", "0", "1", "2", "3", "4", "5", "length chosen on B"],
                [[r["medium"], s] + [f"{float(r[s + str(n)]):.2f}" for n in range(6)] + [r["chosen_length"] if s == "B" else ""] for r in rows for s in ("B", "C")], "llrrrrrrr")

    rows = read("longer_real_summary.csv")
    write_table("longer_chains_real", ["pictures", "set", "0", "1", "2", "3", "4", "5", "length chosen on B"],
                [[{"haze": "real haze", "water": "real tank water"}[r["medium"]], s] + [f"{float(r[s + str(n)]):.2f}" for n in range(6)] + [r["chosen_length"] if s == "B" else ""] for r in rows for s in ("B", "C")], "llrrrrrrr")

    # synthetic MILKY water (calibrated to the tank photos) and the INK what-if
    for cloud in ("milk", "ink"):
        means = mean_by([r for r in read("milky_test.csv") if r["cloud"] == cloud], "method", ["psnr"])
        order = sorted(means, key=lambda m: -means[m][0])[:4] + ["none"]
        write_table(f"milky_{cloud}", ["method", "PSNR dB", "vs none"], [[tex(m), f"{means[m][0]:.2f}", f"{means[m][0] - means['none'][0]:+.2f}"] for m in dict.fromkeys(order)], "lrr")
    rows = read("milky_best.csv")
    body = []
    for r in rows:
        body.append([r["level"], "real tank", "--", "--"] + [f"{float(r['real_' + f]):.2f}" for f in ("contrast_ratio", "brightness_shift", "sharpness_ratio")])
        body.append([r["level"], "synthetic milky", r["beta"], r["blur"]] + [f"{float(r['synthetic_' + f]):.2f}" for f in ("contrast_ratio", "brightness_shift", "sharpness_ratio")])
    write_table("milky_signature", ["band", "pictures", "beta", "blur", "contrast ratio", "brightness shift", "sharpness ratio"], body, "llrrrrr")
    rows = read("milky_agreement.csv")
    body = [[r["methods"], r["band"], r.get("spearman_original_vs_real") or "--", r["spearman_milky_vs_real"], tex(r["best_real"]), tex(r["best_milky"])] for r in rows]
    write_table("milky_agreement", ["method list", "band", "original water", "milky water", "best on real", "best on milky"], body, "llrrll")

    rows = read("ranking_agreement.csv")
    body = [[r["medium"], r["level"], r["score"], r["methods"], f"{float(r['spearman']):+.2f}", f"{float(r['kendall']):+.2f}", tex(r["best_on_synthetic"]), tex(r["best_on_real"])]
            for r in rows if r["score"] == "psnr" and r["level"] == "all"]
    write_table("ranking_agreement", ["medium", "level", "score", "methods", "Spearman", "Kendall", "best on synthetic", "best on real"], body, "llllrrll")

    rows = read("model_validation_summary.csv")
    body = [[r["source"], r["pairs"], f"{float(r['median_tau']):.1f}", f"{float(r['median_airlight']):.2f}", f"{float(r['psnr_no_model']):.1f}",
             f"{float(r['psnr_affine']):.1f}", f"{float(r['heldout_no_model']):.1f}", f"{float(r['heldout_calibrated']):.1f}"] for r in rows]
    write_table("model_validation", ["dataset", "pairs", r"median $\tau$", "median $A$", "none", "affine fit", "held-out none", "held-out ramp"], body, "lrrrrrrr")

    rows = read("scene_matched_summary.csv")
    body = [[r["medium"], r["pictures"], f"{float(r['spearman']):+.2f}", f"{float(r['kendall']):+.2f}", f"{float(r['gain_correlation_per_picture']):+.2f}",
             tex(r["best_on_real"]), tex(r["best_on_synthetic_twin"]), r["mean_abs_psnr_difference_dB"]] for r in rows]
    write_table("scene_matched", ["medium", "pictures", "Spearman", "Kendall", "gain corr.", "best on real", "best on twin", r"$|\Delta$PSNR$|$ dB"], body, "lrrrrllr")

    rows = read("frequency_subtraction_summary.csv")
    body = [[r["world"], tex(r["method"]), r["pictures"], r["psnr"], r["ssim"]] for r in rows]
    write_table("frequency_subtraction", ["data", "method", "pictures", "PSNR dB", "SSIM"], body, "llrrr")

    rows = read("theory_checks.csv")
    write_table("theory_checks", [tex(k) for k in rows[0]], [[tex(v) for v in r.values()] for r in rows], "l" * len(rows[0]))

    rows = read("stacked_summary.csv")
    write_table("stacked_summary", [tex(k) for k in rows[0]], [[tex(v[:8] if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v) for v in r.values()] for r in rows], "l" * len(rows[0]))
    print("wrote LaTeX tables to", OUT)


if __name__ == "__main__":
    main()
