"""Chain several degradations. The first layer is nearest the scene, the last is nearest the camera.

A stack is a list of layers, each {"type": "haze" | "water" | "glass" | "plastic", "enabled": bool, "params": {...}}.
Applying a stack feeds the output of each layer into the next one:

    two layers:  I2 = J*t1*t2 + A1*t2*(1 - t1) + A2*(1 - t2)     (the proposal's stacked equation)

The same JSON stack runs in the demo page (demo/synth.js) and gives matching results there.
"""
import json

import numpy as np

from degrade import glass, haze, plastic, water

MODELS = {"haze": haze, "water": water, "glass": glass, "plastic": plastic}


def apply_stack(clean, depth_map, layers):
    """Run each enabled layer in order and return the final image, clipped to [0, 1] once at the end."""
    image = clean
    for layer in layers:
        if layer.get("enabled", True):
            image = MODELS[layer["type"]].apply(image, depth_map, layer.get("params"))
    return np.clip(image, 0.0, 1.0)


def load_stack(path):
    """Read a stack saved from the demo page (or by save_stack)."""
    with open(path) as handle:
        return json.load(handle)["layers"]


def save_stack(path, layers):
    """Write a stack as JSON so it can be loaded in the demo page."""
    with open(path, "w") as handle:
        json.dump({"layers": layers}, handle, indent=2)
