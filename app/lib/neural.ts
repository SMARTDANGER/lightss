// Neural super-resolution via @huggingface/transformers (ONNX Runtime Web).
// Lazy-loads model on first call. Browser-only. Tiled inference for big images.

"use client";

export type NeuralModelId =
  | "Xenova/swin2SR-classical-sr-x2-64"
  | "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
  | "Xenova/swin2SR-compressed-sr-x4-48";

export type NeuralModelInfo = {
  id: NeuralModelId;
  label: string;
  scale: number;
  approxSizeMB: number;
  desc: string;
};

export const NEURAL_MODELS: NeuralModelInfo[] = [
  {
    id: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    label: "Real-world 4x (önerilen)",
    scale: 4,
    approxSizeMB: 24,
    desc: "Bulanık telefon fotoğrafları için BSRGAN eğitimli, en güçlü.",
  },
  {
    id: "Xenova/swin2SR-compressed-sr-x4-48",
    label: "Sıkıştırılmış 4x",
    scale: 4,
    approxSizeMB: 12,
    desc: "JPEG artefakt + düşük kalite fotoğraflar için.",
  },
  {
    id: "Xenova/swin2SR-classical-sr-x2-64",
    label: "Klasik 2x (hızlı)",
    scale: 2,
    approxSizeMB: 13,
    desc: "Hızlı, hafif. Genel amaç.",
  },
];

let _trans: any = null;
const _pipes = new Map<NeuralModelId, any>();
const _loading = new Map<NeuralModelId, Promise<any>>();

// CDN URL of @huggingface/transformers ESM build. Loaded at runtime by browser
// (NOT bundled by webpack) so we can use a heavy ONNX runtime without bloating
// the Vercel deploy. webpackIgnore tells Next/webpack to leave this import alone.
const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

async function loadTransformers() {
  if (_trans) return _trans;
  // @ts-ignore — runtime ESM fetch from CDN
  _trans = await import(/* webpackIgnore: true */ TRANSFORMERS_CDN);
  _trans.env.allowLocalModels = false;
  _trans.env.allowRemoteModels = true;
  return _trans;
}

export async function getNeuralPipe(modelId: NeuralModelId, onProg?: (pct: number) => void) {
  const cached = _pipes.get(modelId);
  if (cached) return cached;
  const loading = _loading.get(modelId);
  if (loading) return loading;

  const p = (async () => {
    const t = await loadTransformers();
    const pipe = await t.pipeline("image-to-image", modelId, {
      progress_callback: (info: any) => {
        if (info.status === "progress" && info.total) {
          const pct = (info.loaded / info.total) * 100;
          onProg?.(Math.max(0, Math.min(100, pct)));
        }
      },
    });
    _pipes.set(modelId, pipe);
    _loading.delete(modelId);
    return pipe;
  })();
  _loading.set(modelId, p);
  return p;
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
  const od = out.data;
  const rd = raw.data;
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
  const od = out.data;
  const sd = src.data;
  const sw = src.width;
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

// Place tile result back into output with linear feathering on overlap region.
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
      // existing pixel weight + new pixel weight blend
      const cur = dd[di + 3] / 255; // store current accumulated alpha as weight
      const totalW = cur + w;
      if (totalW <= 0) continue;
      dd[di]     = (dd[di]     * cur + td[ti]     * w) / totalW;
      dd[di + 1] = (dd[di + 1] * cur + td[ti + 1] * w) / totalW;
      dd[di + 2] = (dd[di + 2] * cur + td[ti + 2] * w) / totalW;
      dd[di + 3] = Math.min(255, totalW * 255);
    }
  }
}

export type NeuralProgress = (stage: string, pct: number) => void;

export type NeuralOpts = {
  modelId: NeuralModelId;
  maxInput: number; // px on longest side; downscale before inference if larger
  tileSize: number; // 0 = no tiling, run whole image (after maxInput cap)
  overlap: number;  // tile overlap in src px
};

export const neuralDefaults: NeuralOpts = {
  modelId: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
  maxInput: 768,
  tileSize: 192,
  overlap: 24,
};

export async function neuralEnhance(
  src: ImageData,
  opts: NeuralOpts,
  onProg?: NeuralProgress
): Promise<ImageData> {
  const model = NEURAL_MODELS.find((m) => m.id === opts.modelId) ?? NEURAL_MODELS[0];

  onProg?.("model indiriliyor", 0);
  const t = await loadTransformers();
  const pipe = await getNeuralPipe(opts.modelId, (p) =>
    onProg?.("model indiriliyor", p * 0.5)
  );
  onProg?.("hazırlanıyor", 52);

  // optional downscale to cap memory
  let working = src;
  const longSide = Math.max(src.width, src.height);
  if (longSide > opts.maxInput) {
    const k = opts.maxInput / longSide;
    working = downscaleViaCanvas(src, k);
  }

  const sw = working.width, sh = working.height;
  const scale = model.scale;

  if (opts.tileSize <= 0 || (sw <= opts.tileSize && sh <= opts.tileSize)) {
    onProg?.("neural çıkarım", 70);
    const tileImg = imageDataToRaw(working, t.RawImage);
    const result = await pipe(tileImg);
    onProg?.("bitti", 100);
    return rawToImageData(Array.isArray(result) ? result[0] : result);
  }

  // tiled inference
  const ts = opts.tileSize;
  const ov = opts.overlap;
  const step = Math.max(1, ts - ov);
  const tilesX = Math.max(1, Math.ceil((sw - ov) / step));
  const tilesY = Math.max(1, Math.ceil((sh - ov) / step));
  const total = tilesX * tilesY;
  let done = 0;

  // out is initialized with alpha=0 (used as accumulated weight in blendTile)
  const dw = sw * scale, dh = sh * scale;
  const out = new ImageData(dw, dh);

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      let x0 = tx * step;
      let y0 = ty * step;
      if (x0 + ts > sw) x0 = Math.max(0, sw - ts);
      if (y0 + ts > sh) y0 = Math.max(0, sh - ts);
      const tw = Math.min(ts, sw - x0);
      const th = Math.min(ts, sh - y0);

      const tile = extractTile(working, x0, y0, tw, th);
      const raw = imageDataToRaw(tile, t.RawImage);
      const result = await pipe(raw);
      const rawOut = Array.isArray(result) ? result[0] : result;
      const tileOut = rawToImageData(rawOut);
      blendTile(out, tileOut, x0 * scale, y0 * scale, ov * scale);

      done++;
      const pct = 55 + (done / total) * 43;
      onProg?.(`tile ${done}/${total}`, pct);
      // yield UI
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // fill any alpha=0 (shouldn't happen with full coverage); force opaque
  const dOut = out.data;
  for (let i = 3; i < dOut.length; i += 4) dOut[i] = 255;

  onProg?.("bitti", 100);
  return out;
}
