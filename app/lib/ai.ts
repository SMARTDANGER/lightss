// Strong photo enhancement — Remini-like pipeline. Pure TS, browser only.
// Stages: bilateral denoise (Y) → Lanczos 2x upscale (optional) → CAS sharpen (Y)
//        → CLAHE auto-exposure (Y) → clarity (Y) → vibrance/sat polish (RGB).
// All luma operations preserve color. Color polish is conservative.

export type AiOpts = {
  upscale: boolean;       // 2x Lanczos
  deblur: number;         // 0..1   Richardson-Lucy strength (iterations + sigma)
  denoise: number;        // 0..1   bilateral strength
  sharpen: number;        // 0..1   CAS strength
  clarity: number;        // 0..1   wide-radius local contrast on Y
  autoExposure: number;   // 0..1   CLAHE blend
  vibrance: number;       // 0..1   skin-friendly saturation boost
};

export const aiDefaults: AiOpts = {
  upscale: true,
  deblur: 0.6,
  denoise: 0.25,
  sharpen: 0.9,
  clarity: 0.5,
  autoExposure: 0.35,
  vibrance: 0.25,
};

// ---------- color space ----------
function splitYCbCr(data: Uint8ClampedArray) {
  const n = data.length >> 2;
  const Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    Y[i]  =  0.299 * r + 0.587 * g + 0.114 * b;
    Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    Cr[i] =  0.5 * r - 0.418688 * g - 0.081312 * b + 128;
  }
  return { Y, Cb, Cr };
}

function mergeYCbCr(Y: Float32Array, Cb: Float32Array, Cr: Float32Array, out: Uint8ClampedArray) {
  const n = Y.length;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y = Y[i], cb = Cb[i] - 128, cr = Cr[i] - 128;
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    out[p]     = r < 0 ? 0 : r > 255 ? 255 : r;
    out[p + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    out[p + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
    out[p + 3] = 255;
  }
}

// ---------- bilateral filter (denoise, edge-preserving) ----------
// 3x3 window with spatial+range gaussian on Y only. ~2.5x faster than 5x5
// with negligible visual difference for typical denoise needs.
// Inner loop uses precomputed range lookup table for exp().
const _rangeLUT = new Float32Array(512);
let _lutSigmaR = -1;
function getRangeLUT(sigmaR: number) {
  if (sigmaR === _lutSigmaR) return _rangeLUT;
  const sR2 = 2 * sigmaR * sigmaR;
  for (let i = 0; i < 512; i++) _rangeLUT[i] = Math.exp(-(i * i) / sR2);
  _lutSigmaR = sigmaR;
  return _rangeLUT;
}

export function bilateralY(Y: Float32Array, w: number, h: number, sigmaS: number, sigmaR: number): Float32Array {
  const r = 1;
  const out = new Float32Array(Y.length);
  const sS2 = 2 * sigmaS * sigmaS;
  // 3x3 spatial kernel
  const sp = new Float32Array(9);
  for (let dy = -r; dy <= r; dy++)
    for (let dx = -r; dx <= r; dx++)
      sp[(dy + r) * 3 + (dx + r)] = Math.exp(-(dx * dx + dy * dy) / sS2);
  const rangeLUT = getRangeLUT(sigmaR);

  const sp0 = sp[0], sp1 = sp[1], sp2 = sp[2];
  const sp3 = sp[3], sp4 = sp[4], sp5 = sp[5];
  const sp6 = sp[6], sp7 = sp[7], sp8 = sp[8];

  for (let y = 0; y < h; y++) {
    const r0 = (y - 1 < 0 ? 0 : y - 1) * w;
    const r1 = y * w;
    const r2 = (y + 1 >= h ? h - 1 : y + 1) * w;
    for (let x = 0; x < w; x++) {
      const idx = r1 + x;
      const cv = Y[idx];
      const xL = x - 1 < 0 ? 0 : x - 1;
      const xR = x + 1 >= w ? w - 1 : x + 1;

      const v0 = Y[r0 + xL], v1 = Y[r0 + x], v2 = Y[r0 + xR];
      const v3 = Y[r1 + xL],               v5 = Y[r1 + xR];
      const v6 = Y[r2 + xL], v7 = Y[r2 + x], v8 = Y[r2 + xR];

      let d;
      d = v0 - cv; if (d < 0) d = -d; const w0 = sp0 * (d < 512 ? rangeLUT[d | 0] : 0);
      d = v1 - cv; if (d < 0) d = -d; const w1 = sp1 * (d < 512 ? rangeLUT[d | 0] : 0);
      d = v2 - cv; if (d < 0) d = -d; const w2 = sp2 * (d < 512 ? rangeLUT[d | 0] : 0);
      d = v3 - cv; if (d < 0) d = -d; const w3 = sp3 * (d < 512 ? rangeLUT[d | 0] : 0);
      const w4 = sp4; // center, dv=0 → range weight = 1
      d = v5 - cv; if (d < 0) d = -d; const w5 = sp5 * (d < 512 ? rangeLUT[d | 0] : 0);
      d = v6 - cv; if (d < 0) d = -d; const w6 = sp6 * (d < 512 ? rangeLUT[d | 0] : 0);
      d = v7 - cv; if (d < 0) d = -d; const w7 = sp7 * (d < 512 ? rangeLUT[d | 0] : 0);
      d = v8 - cv; if (d < 0) d = -d; const w8 = sp8 * (d < 512 ? rangeLUT[d | 0] : 0);

      const wsum = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8;
      const sum = v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3 + cv * w4 + v5 * w5 + v6 * w6 + v7 * w7 + v8 * w8;
      out[idx] = wsum > 0 ? sum / wsum : cv;
    }
  }
  return out;
}

// ---------- Lanczos-3 separable 2x upscale on Uint8ClampedArray RGBA ----------
function lanczosKernel(x: number, a: number): number {
  if (x === 0) return 1;
  if (x <= -a || x >= a) return 0;
  const pix = Math.PI * x;
  return (a * Math.sin(pix) * Math.sin(pix / a)) / (pix * pix);
}

export function lanczos2x(src: ImageData): ImageData {
  const a = 3;
  const sw = src.width, sh = src.height;
  const dw = sw * 2, dh = sh * 2;
  const sd = src.data;

  // horizontal pass → temp float buffer (dw * sh * 4)
  const tmp = new Float32Array(dw * sh * 4);
  // precompute x weights
  const xWeights: Float32Array[] = new Array(dw);
  const xIndices: Int32Array[] = new Array(dw);
  for (let x = 0; x < dw; x++) {
    const cx = (x + 0.5) / 2 - 0.5; // src coord
    const left = Math.floor(cx) - a + 1;
    const right = Math.floor(cx) + a;
    const ws = new Float32Array(right - left + 1);
    const ids = new Int32Array(right - left + 1);
    let sum = 0;
    for (let i = left; i <= right; i++) {
      const w = lanczosKernel(cx - i, a);
      ws[i - left] = w; sum += w;
      let xi = i; if (xi < 0) xi = 0; else if (xi >= sw) xi = sw - 1;
      ids[i - left] = xi;
    }
    for (let i = 0; i < ws.length; i++) ws[i] /= sum;
    xWeights[x] = ws; xIndices[x] = ids;
  }
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      const ws = xWeights[x], ids = xIndices[x];
      let r = 0, g = 0, b = 0, al = 0;
      for (let k = 0; k < ws.length; k++) {
        const p = (y * sw + ids[k]) * 4;
        const w = ws[k];
        r += sd[p] * w; g += sd[p + 1] * w; b += sd[p + 2] * w; al += sd[p + 3] * w;
      }
      const o = (y * dw + x) * 4;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b; tmp[o + 3] = al;
    }
  }

  // vertical pass → final
  const out = new Uint8ClampedArray(dw * dh * 4);
  const yWeights: Float32Array[] = new Array(dh);
  const yIndices: Int32Array[] = new Array(dh);
  for (let y = 0; y < dh; y++) {
    const cy = (y + 0.5) / 2 - 0.5;
    const top = Math.floor(cy) - a + 1;
    const bot = Math.floor(cy) + a;
    const ws = new Float32Array(bot - top + 1);
    const ids = new Int32Array(bot - top + 1);
    let sum = 0;
    for (let i = top; i <= bot; i++) {
      const w = lanczosKernel(cy - i, a);
      ws[i - top] = w; sum += w;
      let yi = i; if (yi < 0) yi = 0; else if (yi >= sh) yi = sh - 1;
      ids[i - top] = yi;
    }
    for (let i = 0; i < ws.length; i++) ws[i] /= sum;
    yWeights[y] = ws; yIndices[y] = ids;
  }
  for (let y = 0; y < dh; y++) {
    const ws = yWeights[y], ids = yIndices[y];
    for (let x = 0; x < dw; x++) {
      let r = 0, g = 0, b = 0, al = 0;
      for (let k = 0; k < ws.length; k++) {
        const p = (ids[k] * dw + x) * 4;
        const w = ws[k];
        r += tmp[p] * w; g += tmp[p + 1] * w; b += tmp[p + 2] * w; al += tmp[p + 3] * w;
      }
      const o = (y * dw + x) * 4;
      out[o]     = r < 0 ? 0 : r > 255 ? 255 : r;
      out[o + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
      out[o + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
      out[o + 3] = al < 0 ? 0 : al > 255 ? 255 : al;
    }
  }
  return new ImageData(out, dw, dh);
}

// ---------- CAS (Contrast Adaptive Sharpening) on Y channel ----------
// Inspired by AMD FidelityFX CAS. Sharpens flat regions more, edges less.
export function casY(Y: Float32Array, w: number, h: number, strength: number) {
  if (strength <= 0.001) return;
  const src = new Float32Array(Y);
  const peak = -1 / (8 - 3 * Math.min(1, strength)); // dynamic neighbor weight
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const a = src[i - w - 1], b = src[i - w], c = src[i - w + 1];
      const d = src[i - 1],     e = src[i],     f = src[i + 1];
      const g = src[i + w - 1], hh = src[i + w], ii = src[i + w + 1];

      const mn4 = Math.min(d, e, f, b, hh);
      const mx4 = Math.max(d, e, f, b, hh);
      const mn = Math.min(mn4, a, c, g, ii);
      const mx = Math.max(mx4, a, c, g, ii);

      // amplitude: low contrast → 1, high contrast → 0
      const rcp = mx > 0.001 ? 1 / mx : 0;
      const ampMin = Math.min(mn, 255 - mx) * rcp;
      const amp = Math.sqrt(Math.max(0, Math.min(1, ampMin / 255 + 0.001)));
      const wgt = amp * peak * strength;
      const rcpW = 1 / (1 + 4 * wgt);
      const out = (e + wgt * (b + d + f + hh)) * rcpW;
      Y[i] = out < 0 ? 0 : out > 255 ? 255 : out;
    }
  }
}

// ---------- CLAHE (Contrast Limited Adaptive Histogram Equalization) on Y ----------
// Tiled 8x8 grid with bilinear blending. Clip prevents noise blow-up.
export function claheY(Y: Float32Array, w: number, h: number, blend: number, clipLimit = 2.5, tilesX = 8, tilesY = 8) {
  if (blend <= 0.001) return;
  const tw = Math.ceil(w / tilesX), th = Math.ceil(h / tilesY);
  const bins = 256;
  const maps: Uint8Array[] = []; // tilesX*tilesY maps each 256 bins

  // build CDF per tile
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Float32Array(bins);
      const x0 = tx * tw, y0 = ty * th;
      const x1 = Math.min(w, x0 + tw), y1 = Math.min(h, y0 + th);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const v = Y[y * w + x] | 0;
          hist[v < 0 ? 0 : v > 255 ? 255 : v]++;
          count++;
        }
      }
      // clip
      const clip = (clipLimit * count) / bins;
      let excess = 0;
      for (let i = 0; i < bins; i++) if (hist[i] > clip) { excess += hist[i] - clip; hist[i] = clip; }
      const add = excess / bins;
      for (let i = 0; i < bins; i++) hist[i] += add;
      // CDF → map
      const map = new Uint8Array(bins);
      let acc = 0;
      const scale = 255 / Math.max(1, count);
      for (let i = 0; i < bins; i++) { acc += hist[i]; map[i] = Math.round(acc * scale); }
      maps.push(map);
    }
  }

  // apply with bilinear blend between tile maps
  for (let y = 0; y < h; y++) {
    const fy = y / th - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(fy)));
    const ty1 = Math.max(0, Math.min(tilesY - 1, ty0 + 1));
    const dy = fy - ty0;
    for (let x = 0; x < w; x++) {
      const fx = x / tw - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(fx)));
      const tx1 = Math.max(0, Math.min(tilesX - 1, tx0 + 1));
      const dx = fx - tx0;
      const v = Y[y * w + x] | 0;
      const vc = v < 0 ? 0 : v > 255 ? 255 : v;
      const m00 = maps[ty0 * tilesX + tx0][vc];
      const m10 = maps[ty0 * tilesX + tx1][vc];
      const m01 = maps[ty1 * tilesX + tx0][vc];
      const m11 = maps[ty1 * tilesX + tx1][vc];
      const top = m00 * (1 - dx) + m10 * dx;
      const bot = m01 * (1 - dx) + m11 * dx;
      const eq = top * (1 - dy) + bot * dy;
      Y[y * w + x] = Y[y * w + x] * (1 - blend) + eq * blend;
    }
  }
}

// ---------- separable gaussian with kernel cache ----------
const _kernelCache = new Map<string, { r: number; k: Float32Array }>();
function gaussKernel(radius: number) {
  const key = radius.toFixed(3);
  const hit = _kernelCache.get(key);
  if (hit) return hit;
  const r = Math.max(1, Math.round(radius * 2));
  const sigma = Math.max(0.5, radius);
  const k = new Float32Array(r * 2 + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k[i + r] = v; s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const entry = { r, k };
  _kernelCache.set(key, entry);
  return entry;
}

function gauss1D(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const { r, k } = gaussKernel(radius);
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  // horizontal
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let xx = x + i;
        if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
        acc += src[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  // vertical
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let yy = y + i;
        if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// ---------- Richardson-Lucy deconvolution (real deblur) ----------
// Inverts a known Gaussian PSF iteratively. For unknown blur we assume
// symmetric Gaussian with given sigma. 10-20 iterations recover heavy blur.
// PSF is symmetric → flipped = original → reuse the same gauss1D for both
// the forward blur and the back-projection.
export function richardsonLucyY(
  Y: Float32Array, w: number, h: number, sigma: number, iterations: number
) {
  if (iterations < 1) return;
  const observed = new Float32Array(Y);
  // initial estimate = observed
  const estimate = new Float32Array(Y);
  const ratio = new Float32Array(Y.length);

  for (let it = 0; it < iterations; it++) {
    const reblurred = gauss1D(estimate, w, h, sigma);
    for (let i = 0; i < Y.length; i++) {
      const denom = reblurred[i];
      ratio[i] = denom > 0.5 ? observed[i] / denom : 1;
      if (ratio[i] > 4) ratio[i] = 4;   // damping to prevent ringing
      if (ratio[i] < 0.25) ratio[i] = 0.25;
    }
    const correction = gauss1D(ratio, w, h, sigma);
    for (let i = 0; i < Y.length; i++) {
      let v = estimate[i] * correction[i];
      if (v < 0) v = 0; else if (v > 255) v = 255;
      estimate[i] = v;
    }
  }

  for (let i = 0; i < Y.length; i++) Y[i] = estimate[i];
}

// ---------- clarity (wide-radius local contrast) ----------
function clarityYpass(Y: Float32Array, w: number, h: number, amount: number) {
  if (amount <= 0.001) return;
  const radius = Math.min(40, Math.max(8, Math.round(Math.min(w, h) * 0.02)));
  const blur = gauss1D(Y, w, h, radius);
  for (let i = 0; i < Y.length; i++) {
    let v = Y[i] + amount * (Y[i] - blur[i]);
    if (v < 0) v = 0; else if (v > 255) v = 255;
    Y[i] = v;
  }
}

// ---------- vibrance/saturation polish in RGB ----------
function polishRgb(data: Uint8ClampedArray, vibrance: number) {
  if (vibrance <= 0.001) return;
  for (let p = 0; p < data.length; p += 4) {
    const r = data[p] / 255, g = data[p + 1] / 255, b = data[p + 2] / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sCur = mx === 0 ? 0 : (mx - mn) / mx;
    const mul = 1 + vibrance * (1 - sCur) * 0.8;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const nr = lum + (r - lum) * mul;
    const ng = lum + (g - lum) * mul;
    const nb = lum + (b - lum) * mul;
    data[p]     = nr < 0 ? 0 : nr * 255 > 255 ? 255 : nr * 255;
    data[p + 1] = ng < 0 ? 0 : ng * 255 > 255 ? 255 : ng * 255;
    data[p + 2] = nb < 0 ? 0 : nb * 255 > 255 ? 255 : nb * 255;
  }
}

// ---------- orchestrator ----------
export type Progress = (stage: string, pct: number) => void;

export async function aiEnhance(src: ImageData, opts: AiOpts, onProg?: Progress): Promise<ImageData> {
  const yieldUI = () => new Promise<void>((res) => setTimeout(res, 0));

  let cur = src;

  if (opts.upscale) {
    onProg?.("2x büyütme (Lanczos)", 5);
    await yieldUI();
    cur = lanczos2x(cur);
  }

  onProg?.("luma çıkarımı", 30);
  await yieldUI();
  const w = cur.width, h = cur.height;
  const out = new ImageData(new Uint8ClampedArray(cur.data), w, h);
  let { Y, Cb, Cr } = splitYCbCr(out.data);

  if (opts.denoise > 0.001) {
    onProg?.("bilateral denoise", 40);
    await yieldUI();
    const sigmaS = 2.0;
    const sigmaR = 8 + opts.denoise * 22;   // 8..30
    Y = bilateralY(Y, w, h, sigmaS, sigmaR);
  }

  if (opts.deblur > 0.001) {
    // sigma: 1..3.5 (heavier blur assumed for higher strength)
    // iterations: 4..24 (more = stronger recovery + more ringing risk)
    const sigma = 0.8 + opts.deblur * 2.7;
    const iters = Math.round(4 + opts.deblur * 20);
    onProg?.(`deblur (RL ${iters}× σ${sigma.toFixed(1)})`, 55);
    await yieldUI();
    richardsonLucyY(Y, w, h, sigma, iters);
  }

  if (opts.sharpen > 0.001) {
    onProg?.("CAS sharpen", 72);
    await yieldUI();
    // cascade: 3 passes with decreasing strength → stronger snap on blurry source
    casY(Y, w, h, opts.sharpen);
    if (opts.sharpen > 0.4) casY(Y, w, h, opts.sharpen * 0.6);
    if (opts.sharpen > 0.7) casY(Y, w, h, opts.sharpen * 0.35);
  }

  if (opts.autoExposure > 0.001) {
    onProg?.("CLAHE auto-expo", 80);
    await yieldUI();
    claheY(Y, w, h, opts.autoExposure * 0.6);
  }

  if (opts.clarity > 0.001) {
    onProg?.("clarity", 88);
    await yieldUI();
    clarityYpass(Y, w, h, opts.clarity * 0.7);
  }

  onProg?.("renk birleştirme", 94);
  await yieldUI();
  mergeYCbCr(Y, Cb, Cr, out.data);

  if (opts.vibrance > 0.001) {
    onProg?.("vibrance polish", 97);
    await yieldUI();
    polishRgb(out.data, opts.vibrance);
  }

  onProg?.("bitti", 100);
  return out;
}
