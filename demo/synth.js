/* Forward degradation models for the demo page, a line-by-line port of code/degrade/*.py.
 *
 * Images are flat Float64Arrays of height*width*3 numbers in [0, 1], row by row, red-green-blue
 * interleaved (the same memory order as the Python arrays). A depth map is height*width numbers,
 * 0 = near, 1 = far. Every model computes   I = J * t + A * (1 - t).
 *
 * Works in a browser (global `Synth`) and in Node (`require`), so the Python and JavaScript
 * results can be compared by code/tests/parity_check.py.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Synth = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- seeded random numbers (identical to code/common/rng.py) ---------- */
  function randomArray(seed, count) {
    const out = new Float64Array(count);
    const start = seed >>> 0;
    for (let k = 1; k <= count; k++) {
      // the generator's counter just adds a constant each step, so draw k is computed directly
      const a = (start + k * 0x6D2B79F5) % 4294967296;
      let t = Math.imul((a | 0) ^ (a >>> 15), 1 | a);
      t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
      out[k - 1] = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    return out;
  }

  /* ---------- blur, smooth noise, mixing (code/degrade/ops.py) ---------- */
  function gaussianKernel(sigma) {
    const radius = Math.max(1, Math.ceil(3 * sigma));
    const weights = new Float64Array(2 * radius + 1);
    let sum = 0;
    for (let i = -radius; i <= radius; i++) { const w = Math.exp(-0.5 * (i / sigma) * (i / sigma)); weights[i + radius] = w; sum += w; }
    for (let i = 0; i < weights.length; i++) weights[i] /= sum;
    return weights;
  }

  // mirror an out-of-range index back into [0, n) (same as NumPy's 'symmetric' padding)
  function mirrorIndex(index, n) {
    while (index < 0 || index >= n) index = index < 0 ? -index - 1 : 2 * n - index - 1;
    return index;
  }

  /* Gaussian blur of an (height, width, channels) array: rows first, then columns. */
  function gaussianBlur(data, height, width, channels, sigma) {
    if (sigma < 0.05) return Float64Array.from(data);
    const kernel = gaussianKernel(sigma), radius = (kernel.length - 1) / 2;
    const rowsDone = new Float64Array(data.length), result = new Float64Array(data.length);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < channels; c++) {
      let acc = 0;
      for (let tap = 0; tap < kernel.length; tap++) { let yy = y + tap - radius; if (yy < 0 || yy >= height) yy = mirrorIndex(yy, height); acc += kernel[tap] * data[(yy * width + x) * channels + c]; }
      rowsDone[(y * width + x) * channels + c] = acc;
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < channels; c++) {
      let acc = 0;
      for (let tap = 0; tap < kernel.length; tap++) { let xx = x + tap - radius; if (xx < 0 || xx >= width) xx = mirrorIndex(xx, width); acc += kernel[tap] * rowsDone[(y * width + xx) * channels + c]; }
      result[(y * width + x) * channels + c] = acc;
    }
    return result;
  }

  /* A smooth random field in [0, 1] (blobs about `sigma` pixels across). */
  const noiseCache = new Map();
  function smoothNoiseField(height, width, sigma, seed) {
    const key = [height, width, sigma, seed].join('|');
    if (noiseCache.has(key)) return noiseCache.get(key);
    const smooth = gaussianBlur(randomArray(seed, height * width), height, width, 1, sigma);
    let low = Infinity, high = -Infinity;
    for (let i = 0; i < smooth.length; i++) { if (smooth[i] < low) low = smooth[i]; if (smooth[i] > high) high = smooth[i]; }
    const span = Math.max(high - low, 1e-12);
    for (let i = 0; i < smooth.length; i++) smooth[i] = (smooth[i] - low) / span;
    if (noiseCache.size > 8) noiseCache.delete(noiseCache.keys().next().value);
    noiseCache.set(key, smooth);
    return smooth;
  }

  /* base * (1 - weight) + other * weight, weight being one number per pixel. */
  function mix(base, other, weight, height, width) {
    const out = new Float64Array(base.length);
    for (let p = 0; p < height * width; p++) for (let c = 0; c < 3; c++) {
      const i = p * 3 + c; out[i] = base[i] * (1 - weight[p]) + other[i] * weight[p];
    }
    return out;
  }

  /* ---------- default settings (the same numbers as the Python DEFAULTS) ---------- */
  const DEFAULTS = {
    haze: { beta: 1.2, scene_depth: 2.0, airlight: 0.92 },
    water: { beta_red: 0.9, beta_green: 0.35, beta_blue: 0.15, scene_depth: 3.0, veil_red: 0.10, veil_green: 0.42, veil_blue: 0.50 },
    glass: { veil: 0.35, veil_size: 0.05, smudge: 0.30, smudge_size: 0.03, smudge_blur: 0.010, droplets: 25,
             droplet_min: 0.010, droplet_max: 0.040, droplet_blur: 0.006, droplet_strength: 0.80, airlight: 0.85, seed: 7 },
    plastic: { transmission: 0.70, tint_red: 1.0, tint_green: 0.95, tint_blue: 0.80, blur: 0.006, airlight: 0.88 },
  };

  /* ---------- the four models; each returns a new image ---------- */
  function applyHaze(clean, depth, height, width, params) {
    const s = Object.assign({}, DEFAULTS.haze, params), out = new Float64Array(clean.length);
    for (let p = 0; p < height * width; p++) {
      const t = Math.exp(-s.beta * s.scene_depth * depth[p]);
      for (let c = 0; c < 3; c++) out[p * 3 + c] = clean[p * 3 + c] * t + s.airlight * (1 - t);
    }
    return out;
  }

  function applyWater(clean, depth, height, width, params) {
    const s = Object.assign({}, DEFAULTS.water, params), out = new Float64Array(clean.length);
    const betas = [s.beta_red, s.beta_green, s.beta_blue], veil = [s.veil_red, s.veil_green, s.veil_blue];
    for (let p = 0; p < height * width; p++) for (let c = 0; c < 3; c++) {
      const t = Math.exp(-betas[c] * s.scene_depth * depth[p]);
      out[p * 3 + c] = clean[p * 3 + c] * t + veil[c] * (1 - t);
    }
    return out;
  }

  function applyPlastic(clean, depth, height, width, params) {
    const s = Object.assign({}, DEFAULTS.plastic, params), out = new Float64Array(clean.length);
    const soft = gaussianBlur(clean, height, width, 3, s.blur * Math.max(height, width));
    const tint = [s.tint_red, s.tint_green, s.tint_blue];
    for (let p = 0; p < height * width; p++) for (let c = 0; c < 3; c++)
      out[p * 3 + c] = soft[p * 3 + c] * tint[c] * s.transmission + s.airlight * (1 - s.transmission);
    return out;
  }

  /* Glass step 1: the uneven milky film. */
  function glassVeil(image, height, width, s) {
    const field = smoothNoiseField(height, width, s.veil_size * Math.max(height, width), s.seed * 3 + 1), out = new Float64Array(image.length);
    for (let p = 0; p < height * width; p++) {
      const t = 1 - s.veil * (0.4 + 0.6 * field[p]);
      for (let c = 0; c < 3; c++) out[p * 3 + c] = image[p * 3 + c] * t + s.airlight * (1 - t);
    }
    return out;
  }

  /* Glass step 2: smudge patches that scatter the scene. */
  function glassSmudges(image, height, width, s) {
    const longer = Math.max(height, width);
    const field = smoothNoiseField(height, width, s.smudge_size * longer, s.seed * 3 + 2);
    const weight = new Float64Array(height * width);
    for (let p = 0; p < weight.length; p++) weight[p] = s.smudge * Math.min(1, Math.max(0, (field[p] - 0.5) * 2));
    const scattered = gaussianBlur(image, height, width, 3, s.smudge_blur * longer);
    for (let i = 0; i < scattered.length; i++) scattered[i] = scattered[i] * 0.8 + s.airlight * 0.2;
    return mix(image, scattered, weight, height, width);
  }

  /* Alpha map of round soft-edged droplets (1 inside a droplet, 0 outside). */
  function dropletMask(height, width, s) {
    const count = Math.floor(s.droplets), mask = new Float64Array(height * width);
    if (count <= 0) return mask;
    const longer = Math.max(height, width), draws = randomArray(s.seed * 3 + 3, 3 * count);
    for (let d = 0; d < count; d++) {
      const cx = draws[3 * d] * width, cy = draws[3 * d + 1] * height;
      const radius = (s.droplet_min + draws[3 * d + 2] * (s.droplet_max - s.droplet_min)) * longer;
      const edge = Math.max(1, 0.2 * radius);
      // only pixels within `radius` of the centre can be non-zero, so scan just that box
      const x0 = Math.max(0, Math.floor(cx - radius) - 1), x1 = Math.min(width - 1, Math.ceil(cx + radius) + 1);
      const y0 = Math.max(0, Math.floor(cy - radius) - 1), y1 = Math.min(height - 1, Math.ceil(cy + radius) + 1);
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const a = Math.min(1, Math.max(0, (radius - Math.hypot(x - cx, y - cy)) / edge));
        if (a > mask[y * width + x]) mask[y * width + x] = a;
      }
    }
    return mask;
  }

  /* Glass step 3: droplets blur and slightly darken what is behind them. */
  function glassDroplets(image, height, width, s) {
    const seen = gaussianBlur(image, height, width, 3, s.droplet_blur * Math.max(height, width));
    for (let i = 0; i < seen.length; i++) seen[i] = seen[i] * 0.9 + s.airlight * 0.1;
    const weight = dropletMask(height, width, s);
    for (let p = 0; p < weight.length; p++) weight[p] *= s.droplet_strength;
    return mix(image, seen, weight, height, width);
  }

  function applyGlass(clean, depth, height, width, params) {
    const s = Object.assign({}, DEFAULTS.glass, params);
    return glassDroplets(glassSmudges(glassVeil(clean, height, width, s), height, width, s), height, width, s);
  }

  const MODELS = { haze: applyHaze, water: applyWater, glass: applyGlass, plastic: applyPlastic };

  /* Run each enabled layer in order (first = nearest the scene); clip to [0, 1] once at the end. */
  function applyStack(clean, depth, height, width, layers) {
    let image = clean;
    for (const layer of layers) {
      if (layer.enabled === false) continue;
      image = MODELS[layer.type](image, depth, height, width, layer.params);
    }
    const out = new Float64Array(image.length);
    for (let i = 0; i < image.length; i++) out[i] = Math.min(1, Math.max(0, image[i]));
    return out;
  }

  /* ---------- depth maps for pictures without measured depth (code/degrade/depth.py) ---------- */
  function rampDepth(height, width, farAtTop) {
    const depth = new Float64Array(height * width);
    for (let y = 0; y < height; y++) {
      const row = height > 1 ? y / (height - 1) : 0;
      for (let x = 0; x < width; x++) depth[y * width + x] = farAtTop === false ? row : 1 - row;
    }
    return depth;
  }

  function layeredDepth(height, width, cuts, levels) {
    cuts = cuts || [0.4, 0.7]; levels = levels || [1.0, 0.55, 0.15];
    const depth = new Float64Array(height * width), edges = [0].concat(cuts, [1]);
    for (let band = 0; band < levels.length; band++) {
      const top = Math.round(edges[band] * height), bottom = Math.round(edges[band + 1] * height);
      for (let y = top; y < bottom; y++) for (let x = 0; x < width; x++) depth[y * width + x] = levels[band];
    }
    return depth;
  }

  return { randomArray, gaussianBlur, smoothNoiseField, DEFAULTS, MODELS, applyStack, rampDepth, layeredDepth,
           applyHaze, applyWater, applyGlass, applyPlastic };
});
