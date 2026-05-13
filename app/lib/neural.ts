// Direct onnxruntime-web inference (no transformers.js).
// Why direct: transformers.js v3 ships ort.webgpu.bundle.min.mjs which loads
// the jsep WASM (`ort-wasm-simd-threaded.jsep.wasm`). When WebGPU isn't bound
// the jsep wasm throws `Gc[i] is not a function` mid-inference. Going direct
// lets us pick `ort.wasm.min.mjs` — pure wasm, no jsep hooks, reliable.
//
// Model: Swin2SR real-world 4x BSRGAN, fetched from HuggingFace Hub directly.
// Pre/post processing: RGBA → CHW float32 in [0,1], pad to multiple of 8.

"use client";

import { bilateralY, casY, claheY } from "./ai";

export const MODEL = {
  id: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
  label: "Real-world 4x BSRGAN",
  scale: 4,
  approxSizeMB: 24,
  desc: "Swin2SR real-world. Bulanık portre + telefon fotoğrafı için optimize.",
  url: "https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/onnx/model.onnx",
} as const;

export type NeuralOpts = {
  faceGlow: boolean;
  autoColor: boolean;
  hiDetail: boolean;
  maxInput: number;
  tileSize: number;
  overlap: number;
};

export const neuralDefaults: NeuralOpts = {
  faceGlow: true,
  autoColor: true,
  hiDetail: true,
  maxInput: 384,
  tileSize: 128,
  overlap: 16,
};

// ---------- ORT loader ----------
const ORT_VERSION = "1.20.0";
const ORT_ESM_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;
const ORT_WASM_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let _ort: any = null;
let _session: any = null;
let _loading: Promise<any> | null = null;

async function loadOrt(onLog?: (msg: string) => void): Promise<any> {
  if (_ort) return _ort;
  onLog?.("CDN fetch (onnxruntime-web wasm bundle)");
  console.log("[neural] importing ORT from", ORT_ESM_URL);
  // @ts-ignore — runtime ESM fetch (skip webpack bundling)
  _ort = await import(/* webpackIgnore: true */ ORT_ESM_URL);
  console.log("[neural] ORT loaded:", Object.keys(_ort));
  _ort.env.wasm.wasmPaths = ORT_WASM_BASE;
  _ort.env.wasm.numThreads = 1;
  _ort.env.wasm.simd = true;
  _ort.env.logLevel = "warning";
  return _ort;
}

// ---------- IndexedDB model cache ----------
const DB_NAME = "lightss-models";
const DB_STORE = "onnx";

async function idbGet(key: string): Promise<ArrayBuffer | null> {
  return new Promise((resolve) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
    req.onerror = () => resolve(null);
    req.onsuccess = () => {
      const tx = req.result.transaction(DB_STORE, "readonly");
      const get = tx.objectStore(DB_STORE).get(key);
      get.onsuccess = () => resolve(get.result ?? null);
      get.onerror = () => resolve(null);
    };
  });
}

async function idbPut(key: string, buf: ArrayBuffer): Promise<void> {
  return new Promise((resolve) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
    req.onerror = () => resolve();
    req.onsuccess = () => {
      const tx = req.result.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).put(buf, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    };
  });
}

async function fetchModel(url: string, onProg: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  const cached = await idbGet(url);
  if (cached) {
    console.log("[neural] model from IndexedDB cache:", cached.byteLength, "bytes");
    onProg(cached.byteLength, cached.byteLength);
    return cached;
  }
  console.log("[neural] fetching model from", url);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model fetch ${resp.status}`);
  const total = +(resp.headers.get("content-length") || "0");
  if (!resp.body) {
    const buf = await resp.arrayBuffer();
    onProg(buf.byteLength, buf.byteLength);
    await idbPut(url, buf);
    return buf;
  }
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProg(received, total || received);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  await idbPut(url, buf.buffer);
  return buf.buffer;
}

async function getSession(onProg?: (stage: string, pct: number) => void): Promise<{ session: any; ort: any }> {
  if (_session && _ort) { console.log("[neural] session cached"); return { session: _session, ort: _ort }; }
  if (_loading) return _loading;

  _loading = (async () => {
    const ort = await loadOrt((m) => onProg?.(m, 2));
    onProg?.("model indiriliyor", 8);
    const modelBuf = await fetchModel(MODEL.url, (loaded, total) => {
      const mb = (loaded / (1024 * 1024)).toFixed(1);
      const tot = (total / (1024 * 1024)).toFixed(1);
      const pct = total > 0 ? (loaded / total) * 100 : 0;
      onProg?.(`indiriliyor: model.onnx ${mb}/${tot}MB`, 8 + pct * 0.4);
    });
    onProg?.("ONNX session derleniyor", 52);
    console.log("[neural] creating InferenceSession from", modelBuf.byteLength, "bytes");
    const session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    console.log("[neural] session inputs:", session.inputNames, "outputs:", session.outputNames);
    _session = session;
    _loading = null;
    return { session, ort };
  })();
  return _loading;
}

// ---------- helpers ----------
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

// ---------- ONNX inference for a single tile ----------
async function runTile(ort: any, session: any, tile: ImageData): Promise<ImageData> {
  const w = tile.width, h = tile.height;
  // Swin2SR window size is 8. Pad to multiple of 8.
  const win = 8;
  const padW = Math.ceil(w / win) * win;
  const padH = Math.ceil(h / win) * win;

  // RGBA → CHW float32 [0,1], replicate-edge pad
  const inp = new Float32Array(3 * padH * padW);
  const d = tile.data;
  for (let y = 0; y < padH; y++) {
    const sy = y < h ? y : h - 1;
    for (let x = 0; x < padW; x++) {
      const sx = x < w ? x : w - 1;
      const si = (sy * w + sx) * 4;
      inp[0 * padH * padW + y * padW + x] = d[si] / 255;
      inp[1 * padH * padW + y * padW + x] = d[si + 1] / 255;
      inp[2 * padH * padW + y * padW + x] = d[si + 2] / 255;
    }
  }

  const inputTensor = new ort.Tensor("float32", inp, [1, 3, padH, padW]);
  const inputName = session.inputNames[0];
  const outputs = await session.run({ [inputName]: inputTensor });
  const outputName = session.outputNames[0];
  const outT = outputs[outputName];
  const [, , oH, oW] = outT.dims as number[];
  const od = outT.data as Float32Array;

  const scale = oH / padH; // expected 4
  const cropW = w * scale;
  const cropH = h * scale;

  const result = new ImageData(cropW, cropH);
  const rd = result.data;
  const planeSize = oH * oW;
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const dst = (y * cropW + x) * 4;
      const srcIdx = y * oW + x;
      let r = od[0 * planeSize + srcIdx] * 255;
      let g = od[1 * planeSize + srcIdx] * 255;
      let b = od[2 * planeSize + srcIdx] * 255;
      rd[dst]     = r < 0 ? 0 : r > 255 ? 255 : r;
      rd[dst + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
      rd[dst + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
      rd[dst + 3] = 255;
    }
  }
  return result;
}

// ---------- Remini-style post-polish ----------
function reminiPolish(src: ImageData, opts: NeuralOpts): ImageData {
  const w = src.width, h = src.height;
  const out = new ImageData(new Uint8ClampedArray(src.data), w, h);
  const data = out.data;
  const n = w * h;
  const Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    Y[i]  =  0.299 * r + 0.587 * g + 0.114 * b;
    Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    Cr[i] =  0.5 * r - 0.418688 * g - 0.081312 * b + 128;
  }
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
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y = Y[i], cb = Cb[i] - 128, cr = Cr[i] - 128;
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    data[p]     = r < 0 ? 0 : r > 255 ? 255 : r;
    data[p + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    data[p + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
  }
  if (opts.faceGlow) {
    const warmR = 1.06, warmB = 0.96;
    const vibrance = 0.35;
    for (let p = 0; p < data.length; p += 4) {
      let r = data[p] / 255 * warmR;
      let g = data[p + 1] / 255;
      let b = data[p + 2] / 255 * warmB;
      r = (r - 0.5) * 1.06 + 0.5;
      g = (g - 0.5) * 1.06 + 0.5;
      b = (b - 0.5) * 1.06 + 0.5;
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
  const { session, ort } = await getSession((stage, pct) => onProg?.(stage, pct));
  onProg?.("girdi hazırlanıyor", 56);

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
    const t0 = performance.now();
    neuralOut = await runTile(ort, session, working);
    console.log("[neural] inference done in", ((performance.now() - t0) / 1000).toFixed(1), "s");
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
        const tOut = await runTile(ort, session, tile);
        blendTile(neuralOut, tOut, x0 * scale, y0 * scale, ov * scale);
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
