"""Scores that compare a restored picture with the clear reference. Higher PSNR and SSIM are better, lower CIEDE2000 is better.

These are library calls (scikit-image), which the proposal allows for the standard metrics.
"""
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def full_reference_scores(restored, reference):
    """Return {"psnr": dB, "ssim": 0..1, "ciede2000": colour difference} for two float RGB pictures in [0, 1]."""
    return {
        "psnr": float(peak_signal_noise_ratio(reference, restored, data_range=1.0)),           # pixel error, in decibels
        "ssim": float(structural_similarity(reference, restored, channel_axis=2, data_range=1.0)),  # similarity of local structure
        "ciede2000": float(deltaE_ciede2000(rgb2lab(reference), rgb2lab(restored)).mean()),  # perceived colour difference
    }
