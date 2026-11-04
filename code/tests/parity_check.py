"""Check that the Python models (code/degrade) and the demo page's JavaScript (demo/synth.js) agree.

It builds one test scene, runs the same stack of all four effects through both, and reports the largest
difference. A difference below 1/255 is invisible; the models should agree far better than that.
Run from the project root:  .venv/bin/python code/tests/parity_check.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from degrade.depth import ramp_depth  # noqa: E402
from degrade.stack import apply_stack, save_stack  # noqa: E402

HEIGHT, WIDTH = 90, 120

# A stack that uses every effect, with non-default settings so the comparison is meaningful.
STACK = [
    {"type": "water", "enabled": True, "params": {"beta_red": 1.1, "scene_depth": 2.5}},
    {"type": "haze", "enabled": True, "params": {"beta": 0.8, "airlight": 0.8}},
    {"type": "glass", "enabled": True, "params": {"seed": 11, "droplets": 12, "veil": 0.4, "smudge": 0.5}},
    {"type": "plastic", "enabled": True, "params": {"transmission": 0.6, "blur": 0.01}},
]


def test_scene():
    """Smooth colour waves in [0, 1] (the same formula is in parity_node.js)."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    return np.stack([0.5 + 0.5 * np.sin(x * 0.05 + c) * np.cos(y * 0.04) for c in range(3)], axis=-1)


def main():
    scene, depth = test_scene(), ramp_depth(HEIGHT, WIDTH)
    python_result = apply_stack(scene, depth, STACK)
    with tempfile.TemporaryDirectory() as folder:
        stack_file = Path(folder) / "stack.json"
        save_stack(stack_file, STACK)
        node_output = subprocess.run(["node", str(HERE / "parity_node.js"), str(stack_file), str(HEIGHT), str(WIDTH)],
                                     capture_output=True, text=True, check=True).stdout
    js_result = np.array(json.loads(node_output)).reshape(HEIGHT, WIDTH, 3)
    worst = np.abs(python_result - js_result).max()
    print(f"largest difference between Python and JavaScript: {worst:.2e}  (one 8-bit step = {1 / 255:.2e})")
    print("PASS" if worst < 1e-6 else "FAIL")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
