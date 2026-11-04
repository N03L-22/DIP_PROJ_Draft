"""STAGE 3a - the Dark Channel Prior (He, Sun and Tang, 2009), written from scratch.

The idea in four steps. A hazy picture follows   I = J * t + A * (1 - t)
    J  the clear scene we want        t  how much scene light survives (1 = none lost, 0 = all lost)
    A  the colour of the haze (airlight)
1. Estimate A from the haziest-looking pixels.
2. Estimate t from the dark channel: in a clear picture the dark channel of J is about 0, so the dark channel of I / A
   is about 1 - t.
3. Smooth t with a guided filter so it follows the picture's edges (optional).
4. Solve for J:   J = (I - A) / max(t, t0) + A      (t0 stops us dividing by a tiny t and amplifying noise)
"""
import numpy as np

from common.filters import box_filter, dark_channel
from common.stats import luminance


def estimate_airlight(image, dark, top_fraction=0.001):
    """Step 1: the haze colour A.

    Take the brightest 0.1% of the dark channel (the pixels where haze is thickest), and among those pick the
    pixel that is brightest in the picture itself. Returns one colour (3 numbers).
    """
    count = max(1, int(top_fraction * dark.size))
    haziest = np.argsort(dark.ravel())[-count:]                 # positions of the haziest-looking pixels
    brightness = image.reshape(-1, 3).sum(axis=1)[haziest]
    return image.reshape(-1, 3)[haziest[np.argmax(brightness)]]


def raw_transmission(image, airlight, patch=15, omega=0.95):
    """Step 2: t = 1 - omega * darkchannel(I / A). omega < 1 keeps a little haze so far objects still look far away."""
    return 1.0 - omega * dark_channel(image / airlight, patch)


def guided_filter(guide, source, radius, eps):
    """Step 3: smooth `source` (the rough t) so that it follows the edges of `guide` (the grey picture).

    (He, Sun and Tang, 2013.) Inside each window the output is a straight-line function of the guide, a * guide + b,
    chosen to stay as close to the source as possible. eps keeps flat windows from over-fitting.
    """
    mean_guide = box_filter(guide, radius)
    mean_source = box_filter(source, radius)
    covariance = box_filter(guide * source, radius) - mean_guide * mean_source
    variance = box_filter(guide * guide, radius) - mean_guide ** 2
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    return box_filter(a, radius) * guide + box_filter(b, radius)


def recover(image, transmission, airlight, t0=0.1):
    """Step 4: undo the haze, J = (I - A) / max(t, t0) + A, clipped to the valid range."""
    t = np.maximum(transmission, t0)[:, :, None]
    return np.clip((image - airlight) / t + airlight, 0.0, 1.0)


def dcp_restore(image, patch=15, omega=0.95, t0=0.1, refine=True, airlight=None):
    """Run all four steps. Returns a dict with the restored picture and the things worth plotting.

    refine=False keeps the raw blocky transmission (plain DCP); refine=True smooths it (DCP + guided filter).
    Pass `airlight` to use a given colour (or per-pixel colour map) instead of estimating one.
    """
    dark = dark_channel(image, patch)
    if airlight is None:
        airlight = estimate_airlight(image, dark)
    transmission = raw_transmission(image, airlight, patch, omega)
    if refine:
        radius = 4 * patch  # the window of the guided filter is much larger than the dark-channel patch
        transmission = np.clip(guided_filter(luminance(image), transmission, radius, eps=1e-3), 0.0, 1.0)
    return {"restored": recover(image, transmission, airlight, t0), "transmission": transmission,
            "airlight": airlight, "dark_channel": dark}
