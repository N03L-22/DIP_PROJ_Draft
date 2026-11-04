"""Image filters used by more than one stage: Gaussian blur, box average and patch minimum.

Every function works on a 2-D array (one channel) or a 3-D array (height, width, 3, filtered channel by
channel). Borders are mirrored, so the picture does not darken or brighten at its edges.
"""
import numpy as np
from scipy.ndimage import minimum_filter


def gaussian_kernel(sigma):
    """1-D Gaussian weights that add up to 1, reaching 3 sigma to either side."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def _blur_along_axis(array, kernel, axis):
    """Blur one axis: add up copies of the picture shifted by each kernel position, times that position's weight."""
    radius = len(kernel) // 2
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="symmetric")  # mirror the border: d c b a | a b c d | d c b a
    length = array.shape[axis]
    result = np.zeros_like(array)
    for tap, weight in enumerate(kernel):
        index = [slice(None)] * array.ndim
        index[axis] = slice(tap, tap + length)
        result += weight * padded[tuple(index)]
    return result


def gaussian_blur(array, sigma):
    """Gaussian-blur rows then columns. sigma below 0.05 returns an unchanged copy."""
    if sigma < 0.05:
        return array.copy()
    kernel = gaussian_kernel(sigma)
    return _blur_along_axis(_blur_along_axis(array, kernel, 0), kernel, 1)


def gaussian_blur_fft(array, sigma):
    """The same Gaussian blur as gaussian_blur, but through the Fourier transform: its speed does not depend on sigma.

    A Gaussian blur multiplies the spectrum by another Gaussian, exp(-2 pi^2 sigma^2 f^2), so a wide blur costs the same
    as a narrow one. Use this for large sigma (wider than about 10 pixels); use gaussian_blur for small ones.
    """
    if sigma < 0.05:
        return array.copy()
    height, width = array.shape[:2]
    pad = int(np.ceil(3.0 * sigma))  # mirror the border so the transform does not wrap the picture around
    pad_spec = [(pad, pad), (pad, pad)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(array, pad_spec, mode="symmetric")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    transfer = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    if array.ndim == 3:
        transfer = transfer[:, :, None]
    blurred = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * transfer, axes=(0, 1)).real
    return blurred[pad: pad + height, pad: pad + width]


def _window_sums(values, radius):
    """Sum of the (2*radius + 1) square window around every pixel of a 2-D array, using an integral image.

    An integral image stores, at each position, the sum of everything above and to the left. Any window sum is then
    four lookups: bottom-right minus top-right minus bottom-left plus top-left.
    """
    window = 2 * radius + 1
    padded = np.pad(values, radius)  # zeros around the picture, so windows that stick out add nothing
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    return integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window] + integral[:-window, :-window]


def box_filter(array, radius):
    """Average over the (2*radius + 1) square window around each pixel (the guided filter needs many of these).

    Near the border the window sticks out of the picture, so we divide by how many pixels were really inside it.
    """
    pixels_inside = _window_sums(np.ones(array.shape[:2]), radius)
    if array.ndim == 2:
        return _window_sums(array, radius) / pixels_inside
    channels = [_window_sums(array[:, :, c], radius) for c in range(array.shape[2])]
    return np.stack(channels, axis=-1) / pixels_inside[:, :, None]


def patch_minimum(array, patch):
    """Smallest value inside each patch x patch window (the building block of the dark channel)."""
    return minimum_filter(array, size=patch, mode="nearest")


def dark_channel(image, patch=15):
    """The dark channel: for each pixel, the smallest colour value anywhere in the patch around it.

    In a clear outdoor picture almost every patch holds some pixel with one colour near 0 (a shadow, a dark object,
    something saturated). Haze adds light everywhere, so the dark channel rises. That rise is what DCP measures.
    """
    darkest_colour = image.min(axis=2)  # smallest of red, green, blue at each pixel
    return patch_minimum(darkest_colour, patch)
