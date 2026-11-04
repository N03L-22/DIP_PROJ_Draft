"""STAGE 3f - removing haze by SUBTRACTION and by frequency-domain filtering, from the theory of the haze model.

Theory (derived in docs/model-equations.md). Haze is  I = J t + A (1 - t)  with a veil V = A (1 - t) that varies slowly across the
picture. In the frequency domain, for a constant t, every non-zero frequency of J is scaled by t, and the veil adds only to zero
frequency. So undoing haze means (1) subtract the veil (its low frequencies) and (2) multiply the rest by 1 / t. Neither t nor the
veil is known, so these functions estimate them from two priors measured on CLEAR pictures: their average brightness and the
strength of their mid-frequency detail. Those priors are the only knowledge used; no clear version of the picture being restored is needed.
"""
import numpy as np

from common.filters import gaussian_blur_fft
from common.spectra import band_std, frequency_radius
from common.stats import luminance

# average brightness and mid-frequency detail strength of the CLEAR pictures of the calibration half of the real haze data (measured on the first, calibration, half of each dataset)
PRIOR_MEAN = 0.378       # mean luminance of the 94 clear calibration pictures
PRIOR_BAND_STD = 0.0854  # their detail strength in the band 0.04 to 0.25 cycles per pixel


def lowfreq_subtract(image, sigma=0.10, keep=0.0):
    """Subtract the low-frequency part of the picture (the veil), keep the rest, then stretch the result to the full range.

    low = a very wide blur of the picture (sigma is a fraction of the longer side). out = image - (1 - keep) * (low - average of low).
    keep = 0 removes all of the slow variation; keep = 1 changes nothing. This is a high-pass filter written as a subtraction.
    """
    low = gaussian_blur_fft(image, sigma * max(image.shape[:2]))
    flattened = image - (1.0 - keep) * (low - low.mean(axis=(0, 1)))
    lo, hi = np.percentile(flattened, [0.5, 99.5])
    return np.clip((flattened - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def blind_veil_inversion(image, prior_mean=PRIOR_MEAN, prior_band_std=PRIOR_BAND_STD, local_sigma=None):
    """Invert the haze model with estimated t and veil:  J = (I - V) / t.

    t is estimated from how weak the picture's mid-frequency detail is compared with a clear picture's (haze scales detail by t):
        t = (detail strength of this picture) / prior_band_std, limited to 0.1 .. 1
    The veil V is what is left of the average brightness after the scene's own brightness t * prior_mean is accounted for:
        V = average of I - t * prior_mean            (one number: constant veil), or
        V = wide blur of I - t * prior_mean          (a veil that changes across the picture; local_sigma is a fraction of the longer side)
    """
    t = float(np.clip(band_std(image) / prior_band_std, 0.1, 1.0))
    if local_sigma is None:
        veil = image.mean(axis=(0, 1)) - t * prior_mean
    else:
        veil = gaussian_blur_fft(image, local_sigma * max(image.shape[:2])) - t * prior_mean
    return np.clip((image - np.maximum(veil, 0.0)) / t, 0.0, 1.0)


def apply_radial_filter(image, edges, gains):
    """Multiply the picture's spectrum by a radial gain: `gains[i]` for frequencies in ring i. Used to apply a filter measured from data."""
    height, width = image.shape[:2]
    padded = np.pad(image, ((height // 2, height // 2), (width // 2, width // 2), (0, 0)), mode="symmetric")
    radius = frequency_radius(padded.shape[:2])
    ring = np.clip(np.digitize(radius, edges) - 1, 0, len(gains) - 1)
    gain = np.asarray(gains)[ring]
    result = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * gain[:, :, None], axes=(0, 1)).real
    return np.clip(result[height // 2: height // 2 + height, width // 2: width // 2 + width], 0.0, 1.0)
