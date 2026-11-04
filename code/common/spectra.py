"""Frequency-domain helpers used by both the restoration methods and the theory checks: padded FFT, radial rings, band statistics.

Frequencies are in cycles per pixel (0 to about 0.7 at the corners). 'Radial' means we average over rings of equal distance from
the zero frequency, which describes how much of each spatial scale a picture contains, regardless of direction.
"""
import numpy as np

from common.stats import luminance


def padded_spectrum(grey):
    """FFT of a 2-D picture mirrored by half its size on every side (so the picture's borders do not leak into the spectrum). Returns (spectrum, pad)."""
    height, width = grey.shape
    padded = np.pad(grey, ((height // 2, height // 2), (width // 2, width // 2)), mode="symmetric")
    return np.fft.fft2(padded), (height // 2, width // 2)


def frequency_radius(shape):
    """The distance of every FFT position from zero frequency, in cycles per pixel (same layout as np.fft.fft2 output)."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.hypot(fy, fx)


def ring_edges(count=36, top=0.5):
    """Edges of `count` rings from 0 to `top` cycles per pixel (a sample of the spectrum out to the Nyquist limit of 0.5)."""
    return np.linspace(0.0, top, count + 1)


def ring_means(values, radius, edges):
    """Average `values` over each ring of `radius` between consecutive `edges`. Returns one number per ring (nan for an empty ring)."""
    which = np.digitize(radius.ravel(), edges) - 1
    flat = values.ravel()
    return np.array([flat[which == i].mean() if np.any(which == i) else np.nan for i in range(len(edges) - 1)])


def amplitude_ratio(degraded_grey, clear_grey, edges):
    """How much of each spatial scale survives: ring-averaged |degraded spectrum| divided by ring-averaged |clear spectrum|.

    A constant transmission t scales every non-zero frequency by t, so the ratio is flat at t; blur makes it fall with frequency;
    a veil adds energy at the lowest frequencies (the first rings).
    """
    spectrum_d, _ = padded_spectrum(degraded_grey)
    spectrum_c, _ = padded_spectrum(clear_grey)
    radius = frequency_radius(spectrum_d.shape)
    return ring_means(np.abs(spectrum_d), radius, edges) / np.maximum(ring_means(np.abs(spectrum_c), radius, edges), 1e-12)


def band_std(image, low=0.04, high=0.25):
    """Standard deviation of the picture's brightness restricted to the frequency band [low, high] cycles per pixel (the veil's low frequencies and the finest noise are left out)."""
    grey = luminance(image)
    spectrum, pad = padded_spectrum(grey)
    radius = frequency_radius(spectrum.shape)
    in_band = np.fft.ifft2(spectrum * ((radius >= low) & (radius <= high))).real
    return float(in_band[pad[0]: pad[0] + grey.shape[0], pad[1]: pad[1] + grey.shape[1]].std())
