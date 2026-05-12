// Image processing — pure TS, runs in browser. No deps.
// Pipeline: RGB→YCbCr → unsharp on Y (no color shift, no darkening) →
// back to RGB → tone (exposure, shadows, highlights, gamma, BC, temp/tint, sat/vibrance).

export type Settings = {
  sharpAmount: number;   // 0..3
  sharpRadius: number;   // 0.5..4
  sharpThreshold: number;// 0..30 (skip tiny diffs → less noise)
  sharpPasses: number;   // 1..3 (iterative deblur)
  exposure: number;      // -2..2 stops
  brightness: number;    // -1..1
  contrast: number;      // -1..1
  shadows: number;       // -1..1 (lift dark, prevents darkening)
  highlights: number;    // -1..1 (recover bright)
  gamma: number;         // 0.3..3 (1 = neutral)
  temperature: number;   // -1..1 (cool..warm)
  tint: number;          // -1..1 (green..magenta)
  saturation: number;    // -1..1
  vibrance: number;      // -1..1
  clarity: number;       // -1..1 (local contrast on Y)
};

export const defaults: Settings = {
  sharpAmount: 1.2, sharpRadius: 1.2, sharpThreshold: 3, sharpPasses: 1,
  exposure: 0, brightness: 0, contrast: 0,
  shadows: 0.15, highlights: -0.05,
  gamma: 1, temperature: 0, tint: 0,
  saturation: 0, vibrance: 0.1, clarity: 0,
};

// Separable Gaussian blur on a single Float32 channel.
function gaussianBlur1D(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const r = Math.max(1, Math.round(radius * 2));
  const sigma = Math.max(0.5, radius);
  const kernel = new Float32Array(r * 2 + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + r] = v; sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);

  // horizontal
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        let xx = x + k; if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
        acc += src[y * w + xx] * kernel[k + r];
      }
      tmp[y * w + x] = acc;
    }
  }
  // vertical
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        let yy = y + k; if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
        acc += tmp[yy * w + x] * kernel[k + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// Rec.601 YCbCr
function rgbToYCbCr(data: Uint8ClampedArray, w: number, h: number) {
  const n = w * h;
  const Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    Y[i]  =  0.299 * r + 0.587 * g + 0.114 * b;
    Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    Cr[i] =  0.5 * r - 0.418688 * g - 0.081312 * b + 128;
  }
  return { Y, Cb, Cr };
}

function yCbCrToRgb(Y: Float32Array, Cb: Float32Array, Cr: Float32Array, out: Uint8ClampedArray) {
  const n = Y.length;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y = Y[i], cb = Cb[i] - 128, cr = Cr[i] - 128;
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    out[p]     = r < 0 ? 0 : r > 255 ? 255 : r;
    out[p + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    out[p + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
    // alpha untouched
  }
}

// Unsharp mask on Y channel only — preserves color & avoids overall darkening.
// Y' = Y + amount * (Y - blur(Y)). Threshold ignores tiny diffs (denoise-friendly).
function unsharpY(Y: Float32Array, w: number, h: number, amount: number, radius: number, threshold: number) {
  const blur = gaussianBlur1D(Y, w, h, radius);
  for (let i = 0; i < Y.length; i++) {
    const diff = Y[i] - blur[i];
    if (Math.abs(diff) < threshold) continue;
    let v = Y[i] + amount * diff;
    if (v < 0) v = 0; else if (v > 255) v = 255;
    Y[i] = v;
  }
}

// Local contrast (clarity) — wider radius unsharp on Y, softer.
function clarityY(Y: Float32Array, w: number, h: number, amount: number) {
  if (Math.abs(amount) < 0.001) return;
  const blur = gaussianBlur1D(Y, w, h, 20);
  for (let i = 0; i < Y.length; i++) {
    let v = Y[i] + amount * (Y[i] - blur[i]);
    if (v < 0) v = 0; else if (v > 255) v = 255;
    Y[i] = v;
  }
}

// Smooth weight curve for highlights/shadows split. x in [0,1].
const shadowWeight = (x: number) => Math.pow(1 - x, 2);
const highlightWeight = (x: number) => Math.pow(x, 2);

export function applyPipeline(src: ImageData, s: Settings): ImageData {
  const w = src.width, h = src.height;
  const out = new ImageData(new Uint8ClampedArray(src.data), w, h);

  // 1) YCbCr split
  const { Y, Cb, Cr } = rgbToYCbCr(out.data, w, h);

  // 2) Iterative unsharp on Y (deblur, no color shift)
  const passes = Math.max(1, Math.min(3, Math.round(s.sharpPasses)));
  if (s.sharpAmount > 0.001) {
    for (let p = 0; p < passes; p++) {
      const amt = s.sharpAmount / Math.sqrt(p + 1); // diminishing returns
      unsharpY(Y, w, h, amt, s.sharpRadius, s.sharpThreshold);
    }
  }

  // 3) Clarity (local contrast)
  clarityY(Y, w, h, s.clarity);

  // 4) Back to RGB
  yCbCrToRgb(Y, Cb, Cr, out.data);

  // 5) Tone & color in RGB (float-precision per pixel)
  const expMul = Math.pow(2, s.exposure);
  const bright = s.brightness * 255;
  const contrast = s.contrast; // -1..1, applied around 0.5
  const gamma = s.gamma <= 0 ? 1 : s.gamma;
  const invGamma = 1 / gamma;

  // temperature: shift R up / B down (warm) or opposite
  const tempR = 1 + s.temperature * 0.20;
  const tempB = 1 - s.temperature * 0.20;
  // tint: green<->magenta on G channel
  const tintG = 1 - s.tint * 0.15;

  const sat = 1 + s.saturation;     // 0..2
  const vib = s.vibrance;           // -1..1

  const data = out.data;
  for (let i = 0; i < data.length; i += 4) {
    let r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;

    // exposure (linear-ish multiply)
    r *= expMul; g *= expMul; b *= expMul;

    // shadows / highlights — luminance-weighted lifts (prevents global darkening)
    if (s.shadows !== 0 || s.highlights !== 0) {
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const lumC = lum < 0 ? 0 : lum > 1 ? 1 : lum;
      const sW = shadowWeight(lumC) * s.shadows;
      const hW = highlightWeight(lumC) * s.highlights;
      const add = sW * 0.5 + hW * 0.5;
      r += add; g += add; b += add;
    }

    // temperature/tint
    r *= tempR; b *= tempB; g *= tintG;

    // brightness (additive)
    if (bright) { r += bright / 255; g += bright / 255; b += bright / 255; }

    // contrast around 0.5
    if (contrast) {
      r = (r - 0.5) * (1 + contrast) + 0.5;
      g = (g - 0.5) * (1 + contrast) + 0.5;
      b = (b - 0.5) * (1 + contrast) + 0.5;
    }

    // gamma (midtones)
    if (gamma !== 1) {
      r = r <= 0 ? 0 : Math.pow(r, invGamma);
      g = g <= 0 ? 0 : Math.pow(g, invGamma);
      b = b <= 0 ? 0 : Math.pow(b, invGamma);
    }

    // saturation & vibrance (vibrance protects already-saturated pixels)
    if (sat !== 1 || vib !== 0) {
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sCur = mx === 0 ? 0 : (mx - mn) / mx;
      const vibMul = 1 + vib * (1 - sCur);
      const effSat = sat * vibMul;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      r = lum + (r - lum) * effSat;
      g = lum + (g - lum) * effSat;
      b = lum + (b - lum) * effSat;
    }

    // clamp & write
    r = r * 255; g = g * 255; b = b * 255;
    data[i]     = r < 0 ? 0 : r > 255 ? 255 : r;
    data[i + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    data[i + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
  }

  return out;
}
