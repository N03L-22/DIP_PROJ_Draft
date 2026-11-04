"""Sanity checks for the forward models. Each one tests a fact that must hold if the model is right.
Run from the project root:  .venv/bin/python code/tests/test_degrade.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from degrade import glass, haze, plastic, water  # noqa: E402
from degrade.depth import layered_depth, ramp_depth  # noqa: E402
from common.filters import gaussian_blur
from degrade.ops import smooth_noise_field  # noqa: E402
from degrade.stack import apply_stack  # noqa: E402

rng = np.random.default_rng(0)
CLEAN = rng.random((60, 80, 3))
DEPTH = ramp_depth(60, 80)
checks = []


def check(name, condition):
    checks.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)


# haze: no fog density means no change; thicker fog means a brighter, flatter picture
check("haze with beta = 0 leaves the picture unchanged", np.allclose(haze.apply(CLEAN, DEPTH, {"beta": 0.0}), CLEAN))
thin, thick = haze.apply(CLEAN, DEPTH, {"beta": 0.5}), haze.apply(CLEAN, DEPTH, {"beta": 2.0})
check("thicker haze lowers contrast", thick.std() < thin.std() < CLEAN.std())
check("haze is stronger far away (top) than near (bottom)", haze.transmission(DEPTH, haze.DEFAULTS)[0].mean() < haze.transmission(DEPTH, haze.DEFAULTS)[-1].mean())

# water: red must fade faster than blue with distance
far = np.full((60, 80), 1.0)
t = water.transmission(far, water.DEFAULTS)[0, 0]
check("underwater: red survives less than green, green less than blue", t[0] < t[1] < t[2])
check("water with every beta = 0 leaves the picture unchanged",
      np.allclose(water.apply(CLEAN, DEPTH, {"beta_red": 0, "beta_green": 0, "beta_blue": 0}), CLEAN))

# plastic: with transmission 1, no blur and no tint it is invisible; with a tint it changes colour balance
check("plastic with t = 1, no blur, no tint changes nothing",
      np.allclose(plastic.apply(CLEAN, None, {"transmission": 1.0, "blur": 0.0, "tint_red": 1, "tint_green": 1, "tint_blue": 1}), CLEAN))
yellow = plastic.apply(CLEAN, None, {"transmission": 1.0, "blur": 0.0})
check("yellowed plastic lowers blue more than red", (yellow[..., 2].mean() / CLEAN[..., 2].mean()) < (yellow[..., 0].mean() / CLEAN[..., 0].mean()))

# glass: nothing switched on means clear glass; same seed gives the same glass, a new seed a different one
clear_glass = {"veil": 0, "smudge": 0, "droplets": 0}
check("clear glass changes nothing", np.allclose(glass.apply(CLEAN, None, clear_glass), CLEAN))
check("glass is repeatable for the same seed", np.array_equal(glass.apply(CLEAN, None, {"seed": 3}), glass.apply(CLEAN, None, {"seed": 3})))
check("glass differs for a different seed", not np.allclose(glass.apply(CLEAN, None, {"seed": 3}), glass.apply(CLEAN, None, {"seed": 4})))
mask = glass.droplet_mask(60, 80, {**glass.DEFAULTS, "droplets": 10})
check("droplet mask stays within 0 and 1 and is not empty", mask.min() >= 0 and mask.max() <= 1 and mask.max() > 0.9)

# the proposal's two-layer equation  I2 = J*t1*t2 + A1*t2*(1 - t1) + A2*(1 - t2)
flat = np.full((60, 80), 0.5)  # constant depth, so t1 is one number
t1 = float(np.exp(-1.2 * 2.0 * 0.5)); a1 = 0.92; t2 = 0.7; a2 = 0.88
formula = CLEAN * t1 * t2 + a1 * t2 * (1 - t1) + a2 * (1 - t2)
stack = apply_stack(CLEAN, flat, [{"type": "haze"}, {"type": "plastic", "params": {"blur": 0.0, "tint_red": 1, "tint_green": 1, "tint_blue": 1}}])
check("haze then plastic matches the proposal's two-layer equation", np.allclose(stack, np.clip(formula, 0, 1)))
check("a disabled layer is skipped", np.allclose(apply_stack(CLEAN, DEPTH, [{"type": "haze", "enabled": False}]), CLEAN))
check("layer order matters (haze then water differs from water then haze)",
      not np.allclose(apply_stack(CLEAN, DEPTH, [{"type": "haze"}, {"type": "water"}]), apply_stack(CLEAN, DEPTH, [{"type": "water"}, {"type": "haze"}])))

# helpers
check("blur keeps the average brightness", abs(gaussian_blur(CLEAN, 3.0).mean() - CLEAN.mean()) < 1e-3)
field = smooth_noise_field(40, 50, 5.0, 1)
check("smooth noise field spans exactly 0 to 1", abs(field.min()) < 1e-12 and abs(field.max() - 1) < 1e-12)
check("layered depth has the requested levels", set(np.unique(layered_depth(60, 80))) == {1.0, 0.55, 0.15})

failed = [name for name, ok in checks if not ok]
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
sys.exit(1 if failed else 0)
