"""Sanity checks for the filters, restoration methods and metrics. Each tests a fact that must hold if the code is right.
Run from the project root:  .venv/bin/python code/tests/test_restore.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.filters import box_filter, dark_channel, gaussian_blur, gaussian_blur_fft  # noqa: E402
from common.stats import luminance  # noqa: E402
from degrade import haze, plastic  # noqa: E402
from degrade.depth import ramp_depth  # noqa: E402
from metrics.full_reference import full_reference_scores  # noqa: E402
from metrics.no_reference import no_reference_scores  # noqa: E402
from metrics.ranking import ranking_agreement  # noqa: E402
from restore import baselines, extra, fusion  # noqa: E402
from restore.dcp import dcp_restore, guided_filter  # noqa: E402
from restore.methods import REGISTRY, build_method  # noqa: E402

rng = np.random.default_rng(0)
checks = []


def check(name, condition):
    checks.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)


# a clear test scene with real structure: blocks of different colours plus some noise
CLEAR = np.clip(np.kron(rng.random((6, 8, 3)), np.ones((16, 16, 1))) * 0.8 + 0.05 * rng.random((96, 128, 3)), 0, 1)
DEPTH = ramp_depth(96, 128)
HAZY = haze.apply(CLEAR, DEPTH, {"beta": 1.2})

# ---- common filters
window_average = np.stack([uniform_filter(CLEAR[..., c], size=7, mode="constant") for c in range(3)], -1)
window_count = uniform_filter(np.ones(CLEAR.shape[:2]), size=7, mode="constant")[:, :, None]
check("box_filter matches SciPy's window average", np.allclose(box_filter(CLEAR, 3), window_average / window_count))
check("FFT blur is close to the direct blur", np.abs(gaussian_blur(CLEAR, 4) - gaussian_blur_fft(CLEAR, 4)).max() < 5e-3)
check("haze raises the dark channel", dark_channel(HAZY).mean() > dark_channel(CLEAR).mean() + 0.2)

# ---- guided filter and DCP
flat = np.full((40, 50), 0.4)
check("guided filter leaves a constant picture unchanged", np.allclose(guided_filter(flat, flat, 5, 1e-3), 0.4))
guide = np.zeros((40, 50)); guide[:, 25:] = 1.0  # a sharp vertical edge
noisy_source = guide + 0.1 * rng.standard_normal(guide.shape)
smoothed = guided_filter(guide, noisy_source, 4, 1e-3)
check("guided filter keeps the edge of its guide but removes noise",
      abs(smoothed[:, 24].mean() - 0.0) < 0.1 and abs(smoothed[:, 26].mean() - 1.0) < 0.1 and smoothed[:, :20].std() < noisy_source[:, :20].std())
result = dcp_restore(HAZY)
check("DCP raises PSNR on a matched hazy picture", full_reference_scores(result["restored"], CLEAR)["psnr"] > full_reference_scores(HAZY, CLEAR)["psnr"] + 3)
check("DCP lowers the dark channel", dark_channel(result["restored"]).mean() < dark_channel(HAZY).mean())
check("DCP transmission stays within 0 and 1", result["transmission"].min() >= 0 and result["transmission"].max() <= 1)
check("airlight is a bright colour for a hazy picture", result["airlight"].min() > 0.6)

# ---- baselines
check("auto gamma lands on the target brightness", abs(luminance(baselines.auto_gamma(HAZY, 0.5)).mean() - 0.5) < 0.01)
check("histogram equalisation spreads the brightness (std rises on a dull picture)",
      luminance(baselines.histogram_equalise(HAZY)).std() > luminance(HAZY).std())
check("contrast stretch widens the range", np.ptp(luminance(baselines.contrast_stretch(HAZY))) > np.ptp(luminance(HAZY)))
cast = np.clip(CLEAR * np.array([1.0, 0.8, 0.4]), 0, 1)
balanced = baselines.gray_world(cast)
check("gray world reduces a colour cast", np.ptp(balanced.reshape(-1, 3).mean(axis=0)) < np.ptp(cast.reshape(-1, 3).mean(axis=0)))
blurred = plastic.apply(CLEAR, None, {"transmission": 1.0, "tint_red": 1, "tint_green": 1, "tint_blue": 1, "blur": 0.006, "airlight": 0.0})
sharper = baselines.wiener_deblur(blurred, blur_sigma=0.006, noise_to_signal=1e-3)
check("Wiener deblur is closer to the sharp picture than the blurred one", full_reference_scores(sharper, CLEAR)["psnr"] > full_reference_scores(blurred, CLEAR)["psnr"])

# ---- extra techniques (extra.py and fusion.py)
def spread(image):
    """How different the three colour averages are (0 = perfectly balanced)."""
    return float(np.ptp(image.reshape(-1, 3).mean(axis=0)))


for name, fixer in (("white_patch", extra.white_patch), ("shades_of_gray", extra.shades_of_gray), ("von_kries", extra.von_kries)):
    check(f"{name} reduces a colour cast", spread(fixer(cast)) < spread(cast))
bluegreen = np.clip(CLEAR * np.array([0.3, 1.0, 1.0]), 0, 1)
check("red-channel compensation brings red back", extra.red_channel_compensation(bluegreen)[..., 0].mean() > bluegreen[..., 0].mean() + 0.05)
chroma_before = HAZY / np.maximum(HAZY.sum(axis=2, keepdims=True), 1e-3)
equalised = extra.equalise_intensity_hsi(HAZY)
check("HSI-intensity equalisation raises contrast and keeps the colour proportions (where nothing clipped)",
      luminance(equalised).std() > luminance(HAZY).std() and np.abs((equalised / np.maximum(equalised.sum(axis=2, keepdims=True), 1e-3) - chroma_before))[equalised.max(axis=2) < 0.99].mean() < 0.02)
specified = luminance(extra.histogram_specification(HAZY, 0.5, 0.2))
check("histogram specification lands near the requested mean and spread", abs(specified.mean() - 0.5) < 0.05 and abs(specified.std() - 0.2) < 0.06)
check("equalise-to-average-histogram gives all colours the same distribution",
      np.abs(np.sort(extra.equalise_to_average_histogram(HAZY)[..., 0].ravel()) - np.sort(extra.equalise_to_average_histogram(HAZY)[..., 2].ravel())).max() < 1e-6)
check("log and power-law transforms brighten (gamma 0.6 below 1)", luminance(extra.log_transform(HAZY * 0.3)).mean() > luminance(HAZY * 0.3).mean() and extra.power_law(HAZY, 0.6).mean() > HAZY.mean())
gradient_film = np.clip(CLEAR * 0.6 + 0.35 * np.linspace(0, 1, 128)[None, :, None], 0, 1)
check("flat-field correction evens out a slow film (less left-right brightness slope)",
      abs(extra.flat_field(gradient_film)[:, -20:].mean() - extra.flat_field(gradient_film)[:, :20].mean()) < abs(gradient_film[:, -20:].mean() - gradient_film[:, :20].mean()))


def gradient_energy(image):
    """Total edge strength of a picture: larger means sharper."""
    grey = luminance(image)
    return float(np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean())


soft = gaussian_blur(CLEAR, 2.0)
check("unsharp masking and frequency-domain sharpening both raise edge strength",
      gradient_energy(extra.unsharp_mask(soft, 0.01, 1.5)) > gradient_energy(soft) and gradient_energy(extra.frequency_laplacian_sharpen(soft, 2.0)) > gradient_energy(soft))
check("top-hat output stays in range and removes a flat background",
      0 <= extra.top_hat_background(gradient_film).min() and extra.top_hat_background(gradient_film).max() <= 1)
for name, deblur in (("richardson_lucy", extra.richardson_lucy), ("constrained_least_squares", extra.constrained_least_squares)):
    check(f"{name} is closer to the sharp picture than the blurred one", full_reference_scores(deblur(blurred), CLEAR)["psnr"] > full_reference_scores(blurred, CLEAR)["psnr"])
salt = CLEAR.copy(); salt[rng.random(salt.shape[:2]) < 0.05] = 1.0
check("median cleanup removes salt noise", full_reference_scores(extra.median_cleanup(salt), CLEAR)["psnr"] > full_reference_scores(salt, CLEAR)["psnr"] + 2)
flat_noisy = np.clip(0.5 + 0.05 * rng.standard_normal((60, 80, 3)), 0, 1)
check("adaptive noise reduction smooths a flat noisy area", extra.adaptive_noise_reduction(flat_noisy, 7, 0.0025).std() < flat_noisy.std() * 0.8)
check("local-statistics enhancement only brightens (never darkens)", (extra.local_statistics_enhance(HAZY * 0.3) >= HAZY * 0.3 - 1e-12).all())
weights_sum = sum(fusion.mertens_weights(i) for i in (CLEAR, HAZY))
check("fusion weights are positive", weights_sum.min() > 0)
check("pyramid collapse reproduces the picture", np.abs(fusion.blend([CLEAR, CLEAR], levels=4) - CLEAR).max() < 0.15)
check("Tarel's veil removal raises PSNR on a hazy picture", full_reference_scores(fusion.tarel_veil_removal(HAZY), CLEAR)["psnr"] > full_reference_scores(HAZY, CLEAR)["psnr"] + 2)

# ---- the registry: every method returns a valid picture of the same shape
for name in REGISTRY:
    output = build_method({"name": name})(HAZY)
    check(f"method '{name}' returns a valid picture", output.shape == HAZY.shape and output.min() >= -1e-9 and output.max() <= 1 + 1e-9)

# ---- metrics
check("PSNR is infinite and SSIM is 1 for identical pictures", np.isinf(full_reference_scores(CLEAR, CLEAR)["psnr"]) and abs(full_reference_scores(CLEAR, CLEAR)["ssim"] - 1) < 1e-9)
check("haze lowers entropy and UCIQE and raises the dark channel (global contrast is left out: depth-varying haze adds a brightness gradient that can raise it)",
      all(no_reference_scores(HAZY)[k] < no_reference_scores(CLEAR)[k] for k in ("entropy", "uciqe")) and no_reference_scores(HAZY)["dark_channel"] > no_reference_scores(CLEAR)["dark_channel"] + 0.2)
check("ranking agreement is +1 for the same order and -1 for the opposite", ranking_agreement([1, 2, 3, 4], [5, 6, 7, 8])["spearman"] > 0.99 and ranking_agreement([1, 2, 3, 4], [4, 3, 2, 1])["kendall"] < -0.99)

failed = [name for name, ok in checks if not ok]
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
sys.exit(1 if failed else 0)
