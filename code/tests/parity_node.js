// Runs the demo's JavaScript synthesiser on the test scene and prints the result as JSON.
// Called by parity_check.py. Usage: node parity_node.js <stack.json> <height> <width>
const fs = require('fs');
const Synth = require('../../demo/synth.js');
const [stackPath, h, w] = [process.argv[2], +process.argv[3], +process.argv[4]];
const layers = JSON.parse(fs.readFileSync(stackPath, 'utf8')).layers;
// the same test scene as the Python side: smooth colour waves, 0..1
const scene = new Float64Array(h * w * 3);
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++)
  scene[(y * w + x) * 3 + c] = 0.5 + 0.5 * Math.sin(x * 0.05 + c) * Math.cos(y * 0.04);
const out = Synth.applyStack(scene, Synth.rampDepth(h, w, true), h, w, layers);
process.stdout.write(JSON.stringify(Array.from(out)));
