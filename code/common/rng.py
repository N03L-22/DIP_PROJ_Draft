"""A tiny random-number generator that gives the SAME numbers in Python and in JavaScript.

The demo page (demo/synth.js) and these Python models must draw identical noise, droplets and
smudges so their outputs can be compared. NumPy's generator cannot be reproduced in a browser, so
both sides use mulberry32, a 32-bit generator that is short enough to copy exactly.
"""
import numpy as np

_MASK = 0xFFFFFFFF


def mulberry32_array(seed, count):
    """Return `count` uniform numbers in [0, 1) from mulberry32 seeded with `seed`.

    The generator's internal counter just adds a constant each step, so draw number k can be
    computed directly. That lets us make a whole noise image at once instead of looping.
    """
    k = np.arange(1, count + 1, dtype=np.uint64)
    # counter after k steps, wrapped to 32 bits
    a = ((np.uint64(seed & _MASK) + k * np.uint64(0x6D2B79F5)) & np.uint64(_MASK)).astype(np.uint32)
    # NumPy unsigned arithmetic wraps around at 32 bits, which is exactly what the JavaScript
    # version does with Math.imul and the >>> 0 conversions.
    with np.errstate(over="ignore"):
        t = (a ^ (a >> np.uint32(15))) * (a | np.uint32(1))
        t = (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61))) ^ t
        out = (t ^ (t >> np.uint32(14))).astype(np.float64)
    return out / 4294967296.0
