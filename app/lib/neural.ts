// Remini-style enhancement: single model (Swin2SR real-world 4x BSRGAN)
// + post-polish (skin smooth + CAS sharpen + CLAHE + warm vibrance).
// Loaded at runtime from esm.sh — not bundled. ONNX inference runs in Web Worker
// (env.backends.onnx.wasm.proxy = true).

"use client";

import { bilateralY, casY, claheY, richardsonLucyY } from "./ai";

export const MODEL = {
  id: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
  label: "Real-world 4x BSRGAN",
  scale: 4,
  approxSizeMB: 24,
  desc: "Swin2SR real-world. Bulanık portre + telefon fotoğrafı için optimize.",
} as const;

export type NeuralOpts = {
  faceGlow: boolean;   // warm + vibrance + skin smooth
  autoColor: boolean;  // CLAHE auto-exposure
  hiDetail: boolean;   // CAS cascade sharpen + clarity
  maxInput: number;    // px on longest side — downscale before inference
  tileSize: number;    // 0 = no tiling
  overlap: number;
};

export const neuralDefaults: NeuralOpts = {
  faceGlow: true,
  autoColor: true,
  hiDetail: true,
  maxInput: 512,
  tileSize: 128,
  overlap: 16,
};

let _trans: any = null;
let _pipe: any = null;
let _loading: Promise<any> | null = null;

const TRANSFORMERS_CDN = "https://esm.sh/@huggingface/transformers@3.0.2";

async function loadTransformers(onLog?: (msg: string) => void) {
  if (_trans) return _trans;
  onLog?.("CDN fetch (transformers.js)…");
  console.log("[neural] importing transformers.js from", TRANSFORMERS_CDN);
  // @ts-ignore — runtime ESM fetch from CDN
  _trans = await import(/* webpackIgnore: true */ TRANSFORMERS_CDN);
  console.log("[neural] transformers loaded");
  _trans.env.allowLocalModels = false;
  _trans.env.allowRemoteModels = true;
  try {
    _trans.env.backends.onnx.wasm.proxy = true;
    _trans.env.backends.onnx.wasm.numThreads = 1;
  } catch (e) { console.warn("[neural] wasm env setup warn:", e); }
  return _trans;
}

function pickDevice(): "webgpu" | "wasm" {
  if (typeof navigator !== "undefined" && (navigator as any).gpu) return "webgpu";
  return "wasm";
}

async function getPipe(onProg?: (stage: string, pct: number) => void) {
  if (_pipe) { console.log("[neural] pipe cached"); return _pipe; }
  if (_loading) return _loading;

  _loading = (async () => {
    const t = await loadTransformers((m) => onProg?.(m, 2));
    const device = pickDevice();
    console.log("[neural] device:", device);
    onProg?.(`pipeline init (${device})`, 5);

    const baseOpts: any = {
      progress_callback: (info: any) => {
        if (info.status === "download") {
          onProg?.(`indiriliyor: ${info.file ?? "model"}`, 8);
        } else if (info.status === "progress" && info.total) {
          const pct = (info.loaded / info.total) * 100;
          const mb = (info.loaded / (1024 * 1024)).toFixed(1);
          const tot = (info.total / (1024 * 1024)).toFixed(1);
          onProg?.(`indiriliyor: ${info.file ?? "model"} ${mb}/${tot}MB`, 8 + pct * 0.4);
        } else if (info.status === "done") {
          onProg?.("model derleniyor (ONNX session)", 50);
        } else if (info.status === "ready") {
          onProg?.("hazır", 55);
        }
      },
    };
    let pipe;
    try {
      pipe = await t.pipeline("image-to-image", MODEL.id, {
        ...baseOpts,
        device,
        dtype: device === "webgpu" ? "fp16" : "fp32",
      });
    } catch (err) {
      console.warn("[neural] device init failed, fallback to wasm", err);
      onProg?.("wasm fallback", 30);
      pipe = await t.pipeline("image-to-image", MODEL.id, baseOpts);
    }
    _pipe = pipe;
    _loading = null;
    return pipe;
  })();
  return _loading;
}

function imageDataToRaw(src: ImageData, RawImage: any) {
  const n = src.width * src.height;
  const rgb = new Uint8ClampedArray(n * 3);
  const d = src.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    rgb[i * 3]     = d[p];
    rgb[i * 3 + 1] = d[p + 1];
    rgb[i * 3 + 2] = d[p + 2];
  }
  return new RawImage(rgb, src.width, src.height, 3);
}

function rawToImageData(raw: any): ImageData {
  const w = raw.width, h = raw.height;
  const ch = raw.channels;
  const out = new ImageData(w, h);
  const od = out.data, rd = raw.data;
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    od[p]     = rd[i * ch];
    od[p + 1] = rd[i * ch + 1];
    od[p + 2] = rd[i * ch + 2];
    od[p + 3] = 255;
  }
  return out;
}

function downscaleViaCanvas(src: ImageData, factor: number): ImageData {
  const sw = src.width, sh = src.height;
  const dw = Math.max(1, Math.round(sw * factor));
  const dh = Math.max(1, Math.round(sh * factor));
  const tmp = document.createElement("canvas");
  tmp.width = sw; tmp.height = sh;
  tmp.getContext("2d")!.putImageData(src, 0, 0);
  const out = document.createElement("canvas");
  out.width = dw; out.height = dh;
  const ctx = out.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmp, 0, 0, dw, dh);
  return ctx.getImageData(0, 0, dw, dh);
}

function extractTile(src: ImageData, x0: number, y0: number, tw: number, th: number): ImageData {
  const out = new ImageData(tw, th);
  const od = out.data, sd = src.data, sw = src.width;
  for (let y = 0; y < th; y++) {
    const sy = y0 + y;
    for (let x = 0; x < tw; x++) {
      const sx = x0 + x;
      const si = (sy * sw + sx) * 4;
      const di = (y * tw + x) * 4;
      od[di]     = sd[si];
      od[di + 1] = sd[si + 1];
      od[di + 2] = sd[si + 2];
      od[di + 3] = 255;
    }
  }
  return out;
}

function blendTile(dst: ImageData, tile: ImageData, dx0: number, dy0: number, overlapPx: number) {
  const dd = dst.data, td = tile.data;
  const dw = dst.width, dh = dst.height;
  const tw = tile.width, th = tile.height;
  for (let y = 0; y < th; y++) {
    const yy = dy0 + y;
    if (yy < 0 || yy >= dh) continue;
    const wy = overlapPx > 0 ? Math.min(1, Math.min(y, th - 1 - y) / overlapPx) : 1;
    for (let x = 0; x < tw; x++) {
      const xx = dx0 + x;
      if (xx < 0 || xx >= dw) continue;
      const wx = overlapPx > 0 ? Math.min(1, Math.min(x, tw - 1 - x) / overlapPx) : 1;
      const w = wy * wx;
      const di = (yy * dw + xx) * 4;
      const ti = (y * tw + x) * 4;
      const cur = dd[di + 3] / 255;
      const totalW = cur + w;
      if (totalW <= 0) continue;
      dd[di]     = (dd[di]     * cur + td[ti]     * w) / totalW;
      dd[di + 1] = (dd[di + 1] * cur + td[ti + 1] * w) / totalW;
      dd[di + 2] = (dd[di + 2] * cur + td[ti + 2] * w) / totalW;
      dd[di + 3] = Math.min(255, totalW * 255);
    }
  }
}

// ---------- Remini-style post-polish ----------
// Runs on neural SR output. Performs:
//  - skin-smooth bilateral on Y (sigmaR=14)
//  - CAS sharpen cascade (3 passes)
//  - CLAHE auto-exposure (0.35 blend) when autoColor
//  - warm temperature shift + vibrance when faceGlow
function reminiPolish(src: ImageData, opts: NeuralOpts): ImageData {
  const w = src.width, h = src.height;
  const out = new ImageData(new Uint8ClampedArray(src.data), w, h);
  const data = out.data;

  // RGB → Y for luma ops
  const n = w * h;
  const Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    Y[i]  =  0.299 * r + 0.587 * g + 0.114 * b;
    Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    Cr[i] =  0.5 * r - 0.418688 * g - 0.081312 * b + 128;
  }

  // skin smooth: very mild bilateral, only on Y
  if (opts.faceGlow) {
    const smoothed = bilateralY(Y, w, h, 2.0, 14);
    for (let i = 0; i < n; i++) Y[i] = Y[i] * 0.4 + smoothed[i] * 0.6;
  }

  if (opts.hiDetail) {
    casY(Y, w, h, 0.7);
    casY(Y, w, h, 0.4);
  }

  if (opts.autoColor) {
    claheY(Y, w, h, 0.32);
  }

  // back to RGB
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y = Y[i], cb = Cb[i] - 128, cr = Cr[i] - 128;
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    data[p]     = r < 0 ? 0 : r > 255 ? 255 : r;
    data[p + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    data[p + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
  }

  // Remini face glow: warm shift + vibrance + slight contrast
  if (opts.faceGlow) {
    const warmR = 1.06, warmB = 0.96;     // subtle warm
    const vibrance = 0.35;
    for (let p = 0; p < data.length; p += 4) {
      let r = data[p] / 255 * warmR;
      let g = data[p + 1] / 255;
      let b = data[p + 2] / 255 * warmB;

      // contrast around 0.5
      r = (r - 0.5) * 1.06 + 0.5;
      g = (g - 0.5) * 1.06 + 0.5;
      b = (b - 0.5) * 1.06 + 0.5;

      // vibrance — protect already-saturated pixels
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sCur = mx === 0 ? 0 : (mx - mn) / mx;
      const mul = 1 + vibrance * (1 - sCur) * 0.85;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      r = lum + (r - lum) * mul;
      g = lum + (g - lum) * mul;
      b = lum + (b - lum) * mul;

      r *= 255; g *= 255; b *= 255;
      data[p]     = r < 0 ? 0 : r > 255 ? 255 : r;
      data[p + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
      data[p + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
    }
  }

  return out;
}

export type NeuralProgress = (stage: string, pct: number) => void;

export async function neuralEnhance(
  src: ImageData,
  opts: NeuralOpts,
  onProg?: NeuralProgress
): Promise<ImageData> {
  console.log("[neural] start, src:", src.width, "x", src.height);
  onProg?.("başlatılıyor", 1);
  const t = await loadTransformers((m) => onProg?.(m, 2));
  const pipe = await getPipe((stage, pct) => onProg?.(stage, pct));
  onProg?.("girdi hazırlanıyor", 56);

  // downscale if too big
  let working = src;
  const longSide = Math.max(src.width, src.height);
  if (longSide > opts.maxInput) {
    const k = opts.maxInput / longSide;
    working = downscaleViaCanvas(src, k);
  }

  const sw = working.width, sh = working.height;
  const scale = MODEL.scale;
  let neuralOut: ImageData;

  if (opts.tileSize <= 0 || (sw <= opts.tileSize && sh <= opts.tileSize)) {
    console.log("[neural] single-pass", sw, "x", sh);
    onProg?.(`neural çıkarım ${sw}×${sh}`, 60);
    await new Promise((r) => setTimeout(r, 0));
    const tileImg = imageDataToRaw(working, t.RawImage);
    onProg?.("model çalışıyor…", 65);
    const t0 = performance.now();
    const result = await pipe(tileImg);
    console.log("[neural] inference done in", ((performance.now() - t0) / 1000).toFixed(1), "s");
    neuralOut = rawToImageData(Array.isArray(result) ? result[0] : result);
  } else {
    const ts = opts.tileSize;
    const ov = opts.overlap;
    const step = Math.max(1, ts - ov);
    const tilesX = Math.max(1, Math.ceil((sw - ov) / step));
    const tilesY = Math.max(1, Math.ceil((sh - ov) / step));
    const total = tilesX * tilesY;
    const dw = sw * scale, dh = sh * scale;
    neuralOut = new ImageData(dw, dh);
    let done = 0;

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        let x0 = tx * step, y0 = ty * step;
        if (x0 + ts > sw) x0 = Math.max(0, sw - ts);
        if (y0 + ts > sh) y0 = Math.max(0, sh - ts);
        const tw = Math.min(ts, sw - x0);
        const th = Math.min(ts, sh - y0);
        const tile = extractTile(working, x0, y0, tw, th);
        const raw = imageDataToRaw(tile, t.RawImage);
        const result = await pipe(raw);
        const rawOut = Array.isArray(result) ? result[0] : result;
        const tileOut = rawToImageData(rawOut);
        blendTile(neuralOut, tileOut, x0 * scale, y0 * scale, ov * scale);
        done++;
        onProg?.(`tile ${done}/${total}`, 55 + (done / total) * 38);
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    const dOut = neuralOut.data;
    for (let i = 3; i < dOut.length; i += 4) dOut[i] = 255;
  }

  onProg?.("Remini polish", 95);
  await new Promise((r) => setTimeout(r, 0));
  const polished = reminiPolish(neuralOut, opts);
  onProg?.("bitti", 100);
  return polished;
}
