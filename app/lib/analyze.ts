// Image analysis → auto-tune AI sliders.
// Stats: resolution, blur (Laplacian variance), noise (Sobel-suppressed std-dev),
// luminance mean/stddev, dark/bright fraction.
// All metrics computed on a downscaled Y-channel sample for speed (< 5ms typically).

import type { AiOpts } from "./ai";
import type { NeuralOpts } from "./neural";

export type ImageStats = {
  width: number;
  height: number;
  longSide: number;
  pixelCount: number;
  blurScore: number;      // 0..1   higher = blurrier
  laplacianVar: number;   // raw metric (for display)
  noiseScore: number;     // 0..1   higher = noisier
  meanLum: number;        // 0..255
  stdLum: number;         // 0..255 (contrast proxy)
  darkFrac: number;       // 0..1   fraction of pixels with Y < 64
  brightFrac: number;     // 0..1   fraction with Y > 192
};

// Down-sample to ~512 longest side for analysis. Returns Y channel + dims.
function sampleY(src: ImageData, target = 512) {
  const sw = src.width, sh = src.height;
  const long = Math.max(sw, sh);
  const k = long > target ? target / long : 1;
  const dw = Math.max(8, Math.round(sw * k));
  const dh = Math.max(8, Math.round(sh * k));

  if (k === 1) {
    const Y = new Float32Array(sw * sh);
    const d = src.data;
    for (let i = 0, p = 0; i < Y.length; i++, p += 4) {
      Y[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    }
    return { Y, w: sw, h: sh };
  }

  // canvas downscale (bilinear) then read pixels
  const tmp = document.createElement("canvas");
  tmp.width = sw; tmp.height = sh;
  tmp.getContext("2d")!.putImageData(src, 0, 0);
  const out = document.createElement("canvas");
  out.width = dw; out.height = dh;
  const ctx = out.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "medium";
  ctx.drawImage(tmp, 0, 0, dw, dh);
  const id = ctx.getImageData(0, 0, dw, dh);
  const Y = new Float32Array(dw * dh);
  const d = id.data;
  for (let i = 0, p = 0; i < Y.length; i++, p += 4) {
    Y[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }
  return { Y, w: dw, h: dh };
}

export function analyzeImage(src: ImageData): ImageStats {
  const { Y, w, h } = sampleY(src, 512);
  const n = Y.length;

  // mean & std-dev of luminance
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Y[i];
  const mean = sum / n;
  let varSum = 0;
  let dark = 0, bright = 0;
  for (let i = 0; i < n; i++) {
    const d = Y[i] - mean;
    varSum += d * d;
    if (Y[i] < 64) dark++;
    if (Y[i] > 192) bright++;
  }
  const std = Math.sqrt(varSum / n);

  // Laplacian variance — classic blur metric.
  // Sharp images have many strong edges → high variance.
  // For each interior pixel: lap = -4*c + N + S + E + W
  let lapMean = 0, lapVar = 0;
  let lapN = 0;
  // first pass: mean
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = -4 * Y[i] + Y[i - w] + Y[i + w] + Y[i - 1] + Y[i + 1];
      lapMean += lap; lapN++;
    }
  }
  lapMean /= Math.max(1, lapN);
  // second pass: variance
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = -4 * Y[i] + Y[i - w] + Y[i + w] + Y[i - 1] + Y[i + 1];
      const d = lap - lapMean;
      lapVar += d * d;
    }
  }
  lapVar /= Math.max(1, lapN);

  // Map laplacian variance to blur score.
  // Empirical thresholds:
  //   var > 500 → sharp (score ~0)
  //   var = 100 → mildly blurry (score ~0.6)
  //   var <  20 → severely blurry (score ~0.95)
  const blurScore = Math.max(0, Math.min(1, 1 - lapVar / 500));

  // Noise estimate: std of high-frequency residual in flat regions.
  // Approx: residual = Y - mean(3x3 neighborhood); look at low-gradient pixels.
  let nSum = 0, nCount = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = Y[i + 1] - Y[i - 1];
      const gy = Y[i + w] - Y[i - w];
      const gMag = Math.abs(gx) + Math.abs(gy);
      if (gMag > 30) continue; // skip edges
      const avg = (Y[i - w - 1] + Y[i - w] + Y[i - w + 1] +
                   Y[i - 1] + Y[i] + Y[i + 1] +
                   Y[i + w - 1] + Y[i + w] + Y[i + w + 1]) / 9;
      const r = Y[i] - avg;
      nSum += r * r; nCount++;
    }
  }
  const noiseStd = nCount > 0 ? Math.sqrt(nSum / nCount) : 0;
  // noise score: 0..1, std > 8 = noisy
  const noiseScore = Math.max(0, Math.min(1, noiseStd / 8));

  return {
    width: src.width,
    height: src.height,
    longSide: Math.max(src.width, src.height),
    pixelCount: src.width * src.height,
    blurScore,
    laplacianVar: lapVar,
    noiseScore,
    meanLum: mean,
    stdLum: std,
    darkFrac: dark / n,
    brightFrac: bright / n,
  };
}

// ---------- mappers ----------

export function autoTuneAi(stats: ImageStats): AiOpts {
  const b = stats.blurScore;       // 0..1
  const noise = stats.noiseScore;  // 0..1
  const lowLight = stats.meanLum < 90 || stats.darkFrac > 0.4;
  const lowContrast = stats.stdLum < 35;
  const small = stats.longSide < 700;

  // deblur scales with blur severity. Very blurry → max.
  const deblur = b < 0.15 ? 0.0 : Math.min(1, 0.3 + b * 0.85);

  // sharpen: cascade strong on blurry, milder on sharp images
  const sharpen = b < 0.15 ? 0.5 : Math.min(1, 0.7 + b * 0.35);

  // denoise: high if noise high, but reduce when blur is heavy (don't smooth recovered detail)
  let denoise = noise * 0.85;
  if (b > 0.5) denoise *= 0.5;
  denoise = Math.max(0.1, Math.min(0.9, denoise));

  // clarity: more on low-contrast / blurry
  const clarity = Math.min(0.7, 0.3 + (b * 0.4) + (lowContrast ? 0.2 : 0));

  // auto-exposure: kick in for dark or low-contrast scenes
  const autoExposure = lowLight ? 0.55 : lowContrast ? 0.4 : 0.2;

  // vibrance: gentle uniform boost
  const vibrance = 0.2;

  // upscale: only for small images (avoids huge outputs)
  const upscale = small;

  return { upscale, deblur, denoise, sharpen, clarity, autoExposure, vibrance };
}

export function autoTuneNeural(stats: ImageStats): NeuralOpts {
  const long = stats.longSide;
  const hasWebGPU = typeof navigator !== "undefined" && !!(navigator as any).gpu;

  // model is fixed (Swin2SR real-world 4x). Tune only inference params.
  // maxInput: smaller for wasm to fit RAM, bigger for WebGPU.
  let maxInput: number;
  if (hasWebGPU) maxInput = Math.min(long, 768);
  else           maxInput = Math.min(long, 384);

  const tileSize = long <= 256 ? 0 : hasWebGPU ? 192 : 128;
  const overlap = tileSize > 0 ? Math.max(8, Math.round(tileSize * 0.12)) : 0;

  return {
    faceGlow: true,
    autoColor: true,
    hiDetail: true,
    maxInput,
    tileSize,
    overlap,
  };
}
