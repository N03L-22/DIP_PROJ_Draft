"""STAGE 3d - more classical techniques, written by hand. Most come straight from the course notes (lecture numbers in brackets).

Every function takes a float RGB picture (height, width, 3) in [0, 1] and returns one of the same shape, so any of them can
be a step in a pipeline. They are grouped by what they do:
  colour and tone      white patch, shades of gray, von Kries, red-channel compensation, HSI-intensity equalisation,
                       equalisation to the average histogram, histogram specification, log and power-law
  local contrast       local-statistics enhancement (Lec04), flat-field correction, top-hat background removal (Lec13)
  sharpening           unsharp masking / high-boost (Lec04), frequency-domain Laplacian sharpening (Lec12)
  deblurring           Richardson-Lucy, constrained least squares (Lec18)
  cleanup              median, adaptive local noise reduction (Lec17, Lec18)
"""
import numpy as np
from scipy.ndimage import grey_opening, median_filter
from scipy.special import ndtri  # inverse of the normal curve: used to build a target histogram

from common.filters import box_filter, gaussian_blur_fft
from common.stats import luminance


# ------------------------------------------------------------------ colour and tone
def apply_gains(image, gains):
    """Multiply the red, green and blue channels by one gain each and clip. Used by every white-balance method below."""
    return np.clip(image * np.asarray(gains)[None, None, :], 0.0, 1.0)


def white_patch(image, percentile=99.0):
    """White balance: assume the brightest pixels of each colour should be white, so scale each colour to make them 1 (Lec16, 'per-channel scaling')."""
    brightest = np.percentile(image.reshape(-1, 3), percentile, axis=0)
    return apply_gains(image, 1.0 / np.maximum(brightest, 1e-3))


def shades_of_gray(image, power=6.0):
    """White balance between gray-world (power 1) and white-patch (power infinity): estimate the light colour as the p-norm mean of each channel."""
    light = (image.reshape(-1, 3) ** power).mean(axis=0) ** (1.0 / power)
    return apply_gains(image, light.mean() / np.maximum(light, 1e-3))


# matrix that turns RGB into cone responses L, M, S (Hunt-Pointer-Estevez), used by the von Kries method of Lec16
RGB_TO_LMS = np.array([[0.4002, 0.7076, -0.0808], [-0.2263, 1.1653, 0.0457], [0.0, 0.0, 0.9182]])


def von_kries(image):
    """Von Kries white balance (Lec16): divide each cone response by the light's cone response, then go back to RGB.

    The light is estimated as the average colour of the picture (gray-world), the same assumption as gray_world, but the scaling
    is done on cone responses, which is closer to how the eye adapts.
    """
    lms = image @ RGB_TO_LMS.T
    light = lms.reshape(-1, 3).mean(axis=0)
    scaled = lms / np.maximum(light, 1e-6) * light.mean()  # divide by the light, keep overall brightness
    return np.clip(scaled @ np.linalg.inv(RGB_TO_LMS).T, 0.0, 1.0)


def red_channel_compensation(image, strength=1.0):
    """Underwater: red is absorbed first, so put some back, only where red is weak: R' = R + strength * (mean G - mean R) * (1 - R) * G.

    The (1 - R) factor leaves already-bright red alone, the G factor adds red only where there is scene signal (Ancuti et al., 2018).
    """
    red, green = image[:, :, 0], image[:, :, 1]
    boosted = image.copy()
    boosted[:, :, 0] = red + strength * (green.mean() - red.mean()) * (1.0 - red) * green
    return np.clip(boosted, 0.0, 1.0)


def equalise_values(values, target_cdf_function=None):
    """Histogram-match one array of values in [0, 1]. With no target the output histogram is flat (plain equalisation).

    Method: rank every value (its position in the sorted list, 0..1 = the source's cumulative histogram), then push that
    rank through the inverse of the wanted cumulative histogram. Used by three techniques below, so it is a shared function.
    """
    flat = values.ravel()
    ranks = np.empty(flat.size)
    ranks[np.argsort(flat, kind="stable")] = (np.arange(flat.size) + 0.5) / flat.size
    matched = ranks if target_cdf_function is None else target_cdf_function(ranks)
    return matched.reshape(values.shape)


def equalise_intensity_hsi(image):
    """Equalise only the intensity I = (R+G+B)/3 and rescale the colours to match (Lec16: 'equalise I of HSI only').

    Hue and saturation stay put, so colours are not shifted, unlike equalising each colour on its own.
    """
    intensity = image.mean(axis=2)
    new_intensity = equalise_values(intensity)
    return np.clip(image * (new_intensity / np.maximum(intensity, 1e-3))[:, :, None], 0.0, 1.0)


def equalise_to_average_histogram(image):
    """Equalise each colour towards the AVERAGE of the three colour histograms (Lec16, second strategy).

    All three channels end up with the same histogram, so no single channel is stretched more than the others.
    """
    average_sorted = np.sort(image.reshape(-1, 3), axis=0).mean(axis=1)  # the average of the three sorted channels
    positions = (np.arange(average_sorted.size) + 0.5) / average_sorted.size
    return np.stack([equalise_values(image[:, :, c], lambda r: np.interp(r, positions, average_sorted)) for c in range(3)], axis=-1)


def histogram_specification(image, target_mean=0.5, target_std=0.2):
    """Histogram specification (Lec03): reshape the brightness histogram into a bell curve of chosen mean and spread.

    Equalisation forces a FLAT histogram, which over-brightens dark pictures. Here we choose the target instead: most
    pixels near target_mean, fewer far from it. Works on the brightness; colour ratios are kept.
    """
    brightness = luminance(image)
    new_brightness = equalise_values(brightness, lambda r: np.clip(target_mean + target_std * ndtri(np.clip(r, 1e-6, 1 - 1e-6)), 0.0, 1.0))
    return np.clip(image * (new_brightness / np.maximum(brightness, 1e-3))[:, :, None], 0.0, 1.0)


def log_transform(image):
    """Log transform (Lec03): c * log(1 + I) expands the dark values and compresses the bright ones; c makes the maximum 1."""
    return np.log1p(np.clip(image, 0, 1)) / np.log(2.0)


def power_law(image, gamma=0.6):
    """Power-law (gamma) transform (Lec03): I ** gamma. gamma below 1 brightens, above 1 darkens. A fixed value, unlike auto_gamma."""
    return np.clip(image, 0, 1) ** gamma


# ------------------------------------------------------------------ local contrast
def local_statistics_enhance(image, window=0.15, k0=0.5, k1=0.02, k2=0.5, boost=3.0):
    """Enhance only where it is needed, using local mean and local spread (Lec04 'conditional enhancement', Gonzalez and Woods 3.3.4).

    A pixel is multiplied by `boost` when BOTH hold: its neighbourhood is darker than k0 x the picture's average, and the
    neighbourhood's contrast lies between k1 and k2 x the picture's contrast (dark and dull, but not noise and not already strong).
    `window` is a fraction of the longer side.
    """
    grey = luminance(image)
    radius = max(1, int(window * max(grey.shape) / 2))
    local_mean = box_filter(grey, radius)
    local_std = np.sqrt(np.maximum(box_filter(grey * grey, radius) - local_mean ** 2, 0.0))
    needs_help = (local_mean <= k0 * grey.mean()) & (local_std >= k1 * grey.std()) & (local_std <= k2 * grey.std())
    return np.clip(image * np.where(needs_help, boost, 1.0)[:, :, None], 0.0, 1.0)


def flat_field(image, size=0.25):
    """Flat-field correction: estimate the slowly changing film or lighting with a very wide blur, then divide it out.

    A milky film on glass changes slowly across the picture; dividing by the blur removes it and keeps the average brightness.
    `size` is the blur width as a fraction of the longer side.
    """
    background = gaussian_blur_fft(image, size * max(image.shape[:2]))
    return np.clip(image / np.maximum(background, 1e-3) * background.mean(axis=(0, 1)), 0.0, 1.0)


def top_hat_background(image, size=0.25):
    """Remove an uneven background with a morphological opening (Lec13): background = opening of the picture, result = picture - background.

    Opening (erode then dilate) with a large square removes everything smaller than the square, leaving the slowly varying
    background; subtracting it keeps the small bright detail (the 'white top-hat'). The result is stretched back to 0..1.
    """
    window = max(3, int(size * max(image.shape[:2])) | 1)
    background = grey_opening(image, size=(window, window, 1))
    detail = image - background
    return np.clip(detail / max(np.percentile(detail, 99.5), 1e-3), 0.0, 1.0)


# ------------------------------------------------------------------ sharpening
def unsharp_mask(image, sigma=0.008, amount=1.0):
    """Unsharp masking / high-boost (Lec04): add `amount` x (picture - its blurred copy) back to the picture.

    amount 1 is plain unsharp masking; larger amounts are high-boost filtering. sigma is a fraction of the longer side.
    """
    blurred = gaussian_blur_fft(image, sigma * max(image.shape[:2]))
    return np.clip(image + amount * (image - blurred), 0.0, 1.0)


def frequency_laplacian_sharpen(image, strength=1.0):
    """Frequency-domain Laplacian sharpening (Lec12): multiply the spectrum by 1 + strength * (D / D_max)^2, D = distance from the centre.

    The Laplacian in frequency is -4 pi^2 D^2, so 'picture minus Laplacian' is a gain that grows with the square of the
    frequency. We scale D so the highest frequency gets a gain of 1 + strength.
    """
    height, width = image.shape[:2]
    padded = np.pad(image, ((height // 2, height // 2), (width // 2, width // 2), (0, 0)), mode="symmetric")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    gain = 1.0 + strength * (np.hypot(fy, fx) / 0.5) ** 2
    result = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * gain[:, :, None], axes=(0, 1)).real
    return np.clip(result[height // 2: height // 2 + height, width // 2: width // 2 + width], 0.0, 1.0)


# ------------------------------------------------------------------ deblurring
def richardson_lucy(image, blur_sigma=0.006, iterations=12):
    """Richardson-Lucy deconvolution: an iterative deblur that keeps values positive and copes with noise better than a plain inverse.

    Each round asks "if my current estimate were blurred, how would it differ from the picture?" and multiplies the estimate
    by the blurred ratio. A Gaussian blur is its own mirror image, so the same blur is used both ways. blur_sigma is a
    fraction of the longer side, as in the plastic model.
    """
    sigma = blur_sigma * max(image.shape[:2])
    observed = np.maximum(image, 1e-4)
    estimate = observed.copy()
    for _ in range(iterations):
        ratio = observed / np.maximum(gaussian_blur_fft(estimate, sigma), 1e-4)
        estimate = estimate * gaussian_blur_fft(ratio, sigma)
    return np.clip(estimate, 0.0, 1.0)


def constrained_least_squares(image, blur_sigma=0.006, smoothness=0.01):
    """Constrained least squares deblurring (Lec18 family): restored spectrum = conj(H) G / (|H|^2 + gamma |P|^2).

    Like Wiener, but the stabiliser is a smoothness penalty (P = the Laplacian) instead of a noise-to-signal guess, so large
    `smoothness` gives a smoother, less noisy result. blur_sigma is a fraction of the longer side.
    """
    height, width = image.shape[:2]
    sigma = blur_sigma * max(height, width)
    padded = np.pad(image, ((height // 2, height // 2), (width // 2, width // 2), (0, 0)), mode="symmetric")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    blur = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))      # transfer function H of the Gaussian blur
    laplacian = (2.0 * np.pi) ** 2 * (fy ** 2 + fx ** 2)                    # |P|: the Laplacian in frequency
    filter_gain = blur / (blur ** 2 + smoothness * laplacian ** 2)
    restored = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * filter_gain[:, :, None], axes=(0, 1)).real
    return np.clip(restored[height // 2: height // 2 + height, width // 2: width // 2 + width], 0.0, 1.0)


# ------------------------------------------------------------------ cleanup (noise that enhancement amplifies)
def median_cleanup(image, size=3):
    """Median filter (Lec17): replace each pixel by the middle value of its neighbourhood. Removes specks and keeps edges better than a mean."""
    return median_filter(image, size=(size, size, 1), mode="nearest")


def adaptive_noise_reduction(image, window=7, noise_variance=0.002):
    """Adaptive local noise reduction (Lec18): f = g - (noise variance / local variance) * (g - local mean).

    Flat areas (local variance about the noise variance) become the local mean (smoothed); edges (local variance much larger)
    are left almost untouched. The ratio is limited to 1 so a very flat patch is never over-corrected.
    """
    radius = window // 2
    local_mean = box_filter(image, radius)
    local_variance = np.maximum(box_filter(image * image, radius) - local_mean ** 2, 1e-8)
    ratio = np.minimum(noise_variance / local_variance, 1.0)
    return np.clip(image - ratio * (image - local_mean), 0.0, 1.0)
