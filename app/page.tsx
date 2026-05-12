"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { applyPipeline, defaults, Settings } from "./lib/process";

type Range = { key: keyof Settings; label: string; min: number; max: number; step: number };

const SHARP: Range[] = [
  { key: "sharpAmount", label: "Netlik (Sharpen)", min: 0, max: 3, step: 0.05 },
  { key: "sharpRadius", label: "Yarıçap",          min: 0.5, max: 4, step: 0.1 },
  { key: "sharpThreshold", label: "Eşik (gürültü koru)", min: 0, max: 30, step: 1 },
  { key: "sharpPasses", label: "Pas sayısı",        min: 1, max: 3, step: 1 },
  { key: "clarity", label: "Clarity (lokal kontrast)", min: -1, max: 1, step: 0.02 },
];

const LIGHT: Range[] = [
  { key: "exposure", label: "Pozlama (stop)", min: -2, max: 2, step: 0.05 },
  { key: "brightness", label: "Parlaklık", min: -1, max: 1, step: 0.02 },
  { key: "contrast", label: "Kontrast", min: -1, max: 1, step: 0.02 },
  { key: "shadows", label: "Gölgeleri Aç", min: -1, max: 1, step: 0.02 },
  { key: "highlights", label: "Parlakları Düşür", min: -1, max: 1, step: 0.02 },
  { key: "gamma", label: "Gamma (orta tonlar)", min: 0.3, max: 3, step: 0.02 },
];

const COLOR: Range[] = [
  { key: "temperature", label: "Sıcaklık (mavi↔sarı)", min: -1, max: 1, step: 0.02 },
  { key: "tint", label: "Renk Tonu (yeşil↔mor)", min: -1, max: 1, step: 0.02 },
  { key: "saturation", label: "Doygunluk", min: -1, max: 1, step: 0.02 },
  { key: "vibrance", label: "Canlılık (Vibrance)", min: -1, max: 1, step: 0.02 },
];

export default function Page() {
  const [settings, setSettings] = useState<Settings>(defaults);
  const [filename, setFilename] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [hasImage, setHasImage] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceRef = useRef<ImageData | null>(null);
  const rafRef = useRef<number | null>(null);

  const onFile = useCallback((file: File) => {
    setFilename(file.name);
    const img = new Image();
    img.onload = () => {
      const max = 2400;
      let w = img.width, h = img.height;
      if (w > max || h > max) {
        const k = Math.min(max / w, max / h);
        w = Math.round(w * k); h = Math.round(h * k);
      }
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, w, h);
      sourceRef.current = ctx.getImageData(0, 0, w, h);
      const out = canvasRef.current!;
      out.width = w; out.height = h;
      setHasImage(true);
    };
    img.src = URL.createObjectURL(file);
  }, []);

  const schedule = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      if (!sourceRef.current || !canvasRef.current) return;
      setBusy(true);
      const out = applyPipeline(sourceRef.current, settings);
      canvasRef.current.getContext("2d")!.putImageData(out, 0, 0);
      setBusy(false);
    });
  }, [settings]);

  useEffect(() => { if (hasImage) schedule(); }, [settings, hasImage, schedule]);

  const update = (k: keyof Settings, v: number) =>
    setSettings((s) => ({ ...s, [k]: v }));

  const download = () => {
    if (!canvasRef.current) return;
    canvasRef.current.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = (filename.replace(/\.[^.]+$/, "") || "image") + "_lightss.png";
      a.click();
    }, "image/png");
  };

  const reset = () => setSettings(defaults);

  const renderSlider = (r: Range) => {
    const v = settings[r.key];
    return (
      <div className="row" key={r.key}>
        <label>
          <span>{r.label}</span>
          <span>{typeof v === "number" ? v.toFixed(2) : v}</span>
        </label>
        <input
          type="range"
          min={r.min} max={r.max} step={r.step}
          value={v}
          onChange={(e) => update(r.key, parseFloat(e.target.value))}
        />
      </div>
    );
  };

  return (
    <div className="app">
      <div className="canvasWrap">
        {!hasImage && <div className="hint">Resim yükle → netleştir, ışığı ayarla, indir.</div>}
        <canvas ref={canvasRef} style={{ display: hasImage ? "block" : "none" }} />
      </div>

      <div className="panel">
        <div className="header">
          <h2>Lightss</h2>
          <span className="tag">{busy ? "işleniyor…" : "hazır"}</span>
        </div>

        <div className="btns">
          <label className="file">
            <input
              type="file" accept="image/*"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
            />
            Resim Seç
          </label>
          <button onClick={reset} disabled={!hasImage}>Sıfırla</button>
          <button className="primary" onClick={download} disabled={!hasImage}>İndir PNG</button>
        </div>

        <h3>Netleştirme</h3>
        {SHARP.map(renderSlider)}
        <div className="hint">
          Algoritma: YCbCr luminance kanalında iteratif Unsharp Mask. Renk kaymaz, genel parlaklık düşmez.
        </div>

        <h3>Işık</h3>
        {LIGHT.map(renderSlider)}
        <div className="hint">
          Gölgeleri Aç → karartmadan detay açar. Pozlama linear çarpan, gamma orta tonlar.
        </div>

        <h3>Renk</h3>
        {COLOR.map(renderSlider)}

        <div className="hint">
          Tüm işlem tarayıcıda. Görsel sunucuya gitmez. Vercel'de statik olarak çalışır.
        </div>
      </div>
    </div>
  );
}
