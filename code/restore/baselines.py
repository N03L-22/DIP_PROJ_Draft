"""STAGE 3b - generic restoration baselines that are not tied to one medium.

Each takes a float RGB picture (height, width, 3) in [0, 1] and returns one of the same shape. Only formulas that are
short enough to read in one go are written by hand here; CLAHE is the one library call (the proposal allows it).
"""
import warnings

import numpy as np
from skimage.color import lab2rgb, rgb2lab
from skimage.exposure import equalize_adapthist, equalize_hist

from common.filters import gaussian_blur_fft
from common.stats import luminance


def lab_to_rgb(lab):
    """Convert CIELab back to RGB in [0, 1]. Equalising the lightness can push a colour just outside what RGB can show;
    those values are clipped, and the library's warning about that is switched off because it is expected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.clip(lab2rgb(lab), 0.0, 1.0)


def gray_world(image):
    """White balance by the gray-world assumption: the average colour of a scene should be grey, so scale each colour to match."""
    channel_means = image.reshape(-1, 3).mean(axis=0)
    gains = channel_means.mean() / np.maximum(channel_means, 1e-6)
    return np.clip(image * gains, 0.0, 1.0)


def clahe(image, clip_limit=0.01, tiles=8):
    """Local contrast boost (CLAHE) on the lightness only, so colours are not shifted. Library: scikit-image."""
    lab = rgb2lab(image)
    lightness = lab[:, :, 0] / 100.0
    kernel = max(8, max(image.shape[:2]) // tiles)  # tile size in pixels
    lab[:, :, 0] = 100.0 * equalize_adapthist(lightness, kernel_size=kernel, clip_limit=clip_limit)
    return lab_to_rgb(lab)


def retinex_msr(image, sigmas=(15, 80, 250)):
    """Multi-scale Retinex: look at each pixel relative to its blurred surroundings, at three scales, then average.

    log(I) - log(blur(I)) removes the slowly varying light and keeps detail. The result is stretched back to 0..1.
    The scales are given for a 512 px picture and grow or shrink with the picture size.
    """
    scale = max(image.shape[:2]) / 512.0
    log_image = np.log(image + 1e-3)
    total = np.zeros_like(image)
    for sigma in sigmas:
        total += log_image - np.log(gaussian_blur_fft(image, sigma * scale) + 1e-3)
    total /= len(sigmas)
    low, high = np.percentile(total, [1, 99], axis=(0, 1))  # per-colour stretch, ignoring the extreme 1%
    return np.clip((total - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def homomorphic(image, cutoff=0.05, low_gain=0.5, high_gain=1.5):
    """Homomorphic filtering (for uneven veils on glass): take the log, turn the slow waves down and the fast waves up, undo the log.

    The log turns light * detail into light + detail, so a frequency filter can treat them separately. `cutoff` is the
    fraction of the picture size where the filter changes from low_gain (slow light) to high_gain (fine detail).
    """
    height, width = image.shape[:2]
    padded = np.pad(np.log1p(image), ((height // 2, height // 2), (width // 2, width // 2), (0, 0)), mode="symmetric")
    big_h, big_w = padded.shape[:2]
    # distance of each frequency from the centre, as a fraction of the padded size
    fy = np.fft.fftfreq(big_h)[:, None]
    fx = np.fft.fftfreq(big_w)[None, :]
    distance = np.hypot(fy, fx)
    gain = (high_gain - low_gain) * (1.0 - np.exp(-(distance / cutoff) ** 2)) + low_gain
    filtered = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * gain[:, :, None], axes=(0, 1)).real
    result = np.expm1(filtered[height // 2: height // 2 + height, width // 2: width // 2 + width])
    low, high = np.percentile(result, [1, 99], axis=(0, 1))
    return np.clip((result - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def wiener_deblur(image, blur_sigma=0.006, noise_to_signal=0.01):
    """Undo a Gaussian blur with the Wiener filter (for plastic, which softens the picture).

    The blur multiplies the spectrum by H; the Wiener filter divides by it, but gives up where H is tiny:
        restored spectrum = conj(H) * G / (|H|^2 + K)        K = noise-to-signal power ratio
    blur_sigma is a fraction of the longer side, the same unit the plastic model uses.
    """
    height, width = image.shape[:2]
    sigma_pixels = blur_sigma * max(height, width)
    padded = np.pad(image, ((height // 2, height // 2), (width // 2, width // 2), (0, 0)), mode="symmetric")
    big_h, big_w = padded.shape[:2]
    # transfer function of a Gaussian blur: also a Gaussian, in frequency
    fy = np.fft.fftfreq(big_h)[:, None]
    fx = np.fft.fftfreq(big_w)[None, :]
    transfer = np.exp(-2.0 * (np.pi * sigma_pixels) ** 2 * (fy ** 2 + fx ** 2))
    wiener = transfer / (transfer ** 2 + noise_to_signal)  # H is real here, so conj(H) = H
    restored = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * wiener[:, :, None], axes=(0, 1)).real
    return np.clip(restored[height // 2: height // 2 + height, width // 2: width // 2 + width], 0.0, 1.0)


def histogram_equalise(image):
    """Global histogram equalisation on the lightness only: spread the brightness values evenly over the whole range.

    Cheap and strong, but it works on the whole picture at once, so it cannot tell fog from scene. Library: scikit-image.
    """
    lab = rgb2lab(image)
    lab[:, :, 0] = 100.0 * equalize_hist(lab[:, :, 0] / 100.0)
    return lab_to_rgb(lab)


def histogram_equalise_rgb(image):
    """Histogram equalisation on each of red, green and blue separately.

    This also evens out the colour balance (it can fix a colour cast) but each colour is stretched on its own, so
    colours can shift into unnatural hues. Kept to show that difference against the lightness-only version.
    """
    return np.stack([equalize_hist(image[:, :, c]) for c in range(3)], axis=-1)


def auto_gamma(image, target=0.5):
    """Brighten or darken with a power curve chosen so the average brightness lands on `target`: result = image ** gamma.

    gamma < 1 brightens (good for dark underwater pictures), gamma > 1 darkens (for washed-out fog). The right gamma has no
    simple formula (the average of a power is not the power of the average), so it is found by bisection: the average
    brightness falls steadily as gamma grows, so we keep halving the interval that must contain the answer.
    """
    image = np.clip(image, 0.0, 1.0)
    sample = image[::2, ::2]                      # every second pixel in each direction is plenty to find the right gamma
    low, high = 0.05, 20.0                        # the answer lies somewhere in here
    for _ in range(28):                           # 28 halvings narrow the interval far below any visible difference
        middle = np.sqrt(low * high)              # halve the interval on a log scale
        if luminance(sample ** middle).mean() > target:
            low = middle                          # still too bright: need a bigger gamma
        else:
            high = middle
    return image ** np.sqrt(low * high)


def contrast_stretch_per_channel(image, low_percentile=1.0, high_percentile=99.0):
    """Linear stretch with a SEPARATE range for each colour: the inverse of a per-colour affine map I_c = a_c J_c + b_c (a coloured glow or a tint).

    contrast_stretch below uses one shared range, so it cannot undo a different gain and offset per colour; this one can (and it also removes a colour cast).
    """
    low, high = (np.percentile(image, q, axis=(0, 1)) for q in (low_percentile, high_percentile))
    return np.clip((image - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def contrast_stretch(image, low_percentile=1.0, high_percentile=99.0):
    """Linear stretch: the darkest `low_percentile` % maps to black and the brightest `100 - high_percentile` % to white.

    One shared range for all colours, so colour balance is not changed.
    """
    low, high = np.percentile(luminance(image), [low_percentile, high_percentile])
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
