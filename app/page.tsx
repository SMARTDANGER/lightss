"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { applyPipeline, defaults, Settings } from "./lib/process";
import {
  neuralDefaults,
  neuralEnhance,
  NeuralOpts,
  MODEL,
} from "./lib/neural";
import { analyzeImage, autoTuneAi, autoTuneNeural, ImageStats } from "./lib/analyze";
import CompareSlider from "./components/CompareSlider";

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
  const [neuralOpts, setNeuralOpts] = useState<NeuralOpts>(neuralDefaults);
  const [filename, setFilename] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [hasImage, setHasImage] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiStage, setAiStage] = useState<string>("");
  const [aiPct, setAiPct] = useState<number>(0);
  const [aiError, setAiError] = useState<string>("");
  const [aiElapsed, setAiElapsed] = useState<number>(0);
  const [stats, setStats] = useState<ImageStats | null>(null);
  const [autoTune, setAutoTune] = useState(true);
  const [beforeData, setBeforeData] = useState<ImageData | null>(null);
  const [afterData, setAfterData] = useState<ImageData | null>(null);
  const [showCompare, setShowCompare] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceRef = useRef<ImageData | null>(null);
  const originalRef = useRef<ImageData | null>(null);
  const rafRef = useRef<number | null>(null);

  const onFile = useCallback((file: File) => {
    setFilename(file.name);
    setShowCompare(false);
    setBeforeData(null);
    setAfterData(null);
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
      const id = ctx.getImageData(0, 0, w, h);
      sourceRef.current = id;
      originalRef.current = new ImageData(new Uint8ClampedArray(id.data), w, h);
      const out = canvasRef.current!;
      out.width = w; out.height = h;
      try {
        const t0 = performance.now();
        const s = analyzeImage(id);
        console.log("[analyze] in", (performance.now() - t0).toFixed(1), "ms", s);
        setStats(s);
        if (autoTune) {
          // also tune classic settings via algorithmic mapper for any post-AI manual tweak baseline
          autoTuneAi(s);
          setNeuralOpts(autoTuneNeural(s));
        }
      } catch (err) {
        console.warn("[analyze] failed", err);
      }
      setHasImage(true);
    };
    img.src = URL.createObjectURL(file);
  }, [autoTune]);

  const reTune = () => {
    if (!originalRef.current) return;
    const s = analyzeImage(originalRef.current);
    setStats(s);
    setNeuralOpts(autoTuneNeural(s));
  };

  const schedule = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      if (!sourceRef.current || !canvasRef.current) return;
      setBusy(true);
      const out = applyPipeline(sourceRef.current, settings);
      const cnv = canvasRef.current;
      if (cnv.width !== out.width || cnv.height !== out.height) {
        cnv.width = out.width; cnv.height = out.height;
      }
      cnv.getContext("2d")!.putImageData(out, 0, 0);
      setBusy(false);
    });
  }, [settings]);

  useEffect(() => { if (hasImage && !showCompare) schedule(); }, [settings, hasImage, schedule, showCompare]);

  const runAi = async () => {
    if (!originalRef.current) return;
    setAiBusy(true); setAiPct(0); setAiStage("başlatılıyor"); setAiError("");
    const t0 = performance.now();
    const elapsedTimer = window.setInterval(() => {
      setAiElapsed(Math.round((performance.now() - t0) / 1000));
    }, 250);
    try {
      const result = await neuralEnhance(originalRef.current, neuralOpts, (stage, pct) => {
        setAiStage(stage); setAiPct(Math.round(pct));
      });
      sourceRef.current = result;
      setBeforeData(originalRef.current);
      setAfterData(result);
      setShowCompare(true);
      const cnv = canvasRef.current!;
      cnv.width = result.width; cnv.height = result.height;
      cnv.getContext("2d")!.putImageData(result, 0, 0);
      setAiPct(100); setAiStage("bitti");
    } catch (e: any) {
      console.error("[runAi] failed:", e);
      const msg = e?.message || e?.toString() || "bilinmeyen hata";
      setAiStage("HATA");
      setAiError(msg);
    } finally {
      window.clearInterval(elapsedTimer);
      setAiBusy(false);
    }
  };

  const revertToOriginal = () => {
    if (!originalRef.current) return;
    sourceRef.current = new ImageData(
      new Uint8ClampedArray(originalRef.current.data),
      originalRef.current.width,
      originalRef.current.height
    );
    setShowCompare(false);
    setBeforeData(null);
    setAfterData(null);
    setSettings(defaults);
    schedule();
  };

  const update = (k: keyof Settings, v: number) => setSettings((s) => ({ ...s, [k]: v }));
  const toggleNeural = (k: "faceGlow" | "autoColor" | "hiDetail") =>
    setNeuralOpts((s) => ({ ...s, [k]: !s[k] }));

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
        {!hasImage && <div className="hint">Resim yükle → AI Netleştir → karşılaştır → indir.</div>}
        <canvas
          ref={canvasRef}
          style={{ display: hasImage && !showCompare ? "block" : "none" }}
        />
        {hasImage && showCompare && beforeData && afterData && (
          <CompareSlider beforeData={beforeData} afterData={afterData} />
        )}
        {aiBusy && (
          <div className="aiOverlay">
            <div className="spinner" />
            <div className="overlayStage">{aiStage}</div>
            <div className="overlayPct">%{aiPct}</div>
            <div className="overlayBar"><div style={{ width: `${aiPct}%` }} /></div>
            <div className="overlayMeta">{aiElapsed}s · konsol (F12) detay</div>
          </div>
        )}
        {!aiBusy && aiError && (
          <div className="aiError">
            <div><strong>Hata</strong></div>
            <div className="errMsg">{aiError}</div>
            <button onClick={() => setAiError("")}>Kapat</button>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="header">
          <h2>Lightss</h2>
          <span className="tag">{aiBusy ? "AI işliyor…" : busy ? "işleniyor…" : "hazır"}</span>
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

        {stats && (
          <div className="stats">
            <div className="statsHead">
              <span>📊 Otomatik analiz</span>
              <button className="reTune" onClick={reTune} disabled={!hasImage}>Yeniden ayarla</button>
            </div>
            <div className="statsGrid">
              <div><span>Boyut</span><b>{stats.width}×{stats.height}</b></div>
              <div><span>Bulanıklık</span><b className={stats.blurScore > 0.5 ? "bad" : stats.blurScore > 0.2 ? "med" : "good"}>
                {(stats.blurScore * 100).toFixed(0)}%
              </b></div>
              <div><span>Gürültü</span><b className={stats.noiseScore > 0.5 ? "bad" : stats.noiseScore > 0.2 ? "med" : "good"}>
                {(stats.noiseScore * 100).toFixed(0)}%
              </b></div>
              <div><span>Parlaklık</span><b>{stats.meanLum.toFixed(0)}/255</b></div>
            </div>
            <label className="check">
              <input type="checkbox" checked={autoTune} onChange={(e) => setAutoTune(e.target.checked)} />
              <span>Yeni resimde otomatik ayarla</span>
            </label>
          </div>
        )}

        <button
          className="aiBtn"
          onClick={runAi}
          disabled={!hasImage || aiBusy}
          title={`${MODEL.label} (${MODEL.scale}x, ~${MODEL.approxSizeMB}MB)`}
        >
          {aiBusy ? `${aiStage} — %${aiPct}` : "AI NETLEŞTİR"}
        </button>
        {aiBusy && (
          <div className="progress"><div style={{ width: `${aiPct}%` }} /></div>
        )}

        <div className="reminiToggles">
          <button
            className={`pill ${neuralOpts.faceGlow ? "on" : ""}`}
            onClick={() => toggleNeural("faceGlow")}
            disabled={aiBusy}
            title="Sıcak ton + vibrance + cilt yumuşat"
          >✨ Yüz Glow</button>
          <button
            className={`pill ${neuralOpts.autoColor ? "on" : ""}`}
            onClick={() => toggleNeural("autoColor")}
            disabled={aiBusy}
            title="CLAHE tile-bazlı otomatik pozlama"
          >🎨 Oto Renk</button>
          <button
            className={`pill ${neuralOpts.hiDetail ? "on" : ""}`}
            onClick={() => toggleNeural("hiDetail")}
            disabled={aiBusy}
            title="CAS cascade sharpen — keskin detay"
          >🔍 Yüksek Detay</button>
        </div>
        <div className="modelInfo">
          <span>Model: <b>{MODEL.label}</b> · {MODEL.scale}x · ~{MODEL.approxSizeMB}MB</span>
          <span className="dim">{MODEL.desc}</span>
        </div>

        {hasImage && (beforeData && afterData) && (
          <div className="btns">
            <button onClick={() => setShowCompare((s) => !s)}>
              {showCompare ? "Karşılaştırmayı Gizle" : "Önce/Sonra Karşılaştır"}
            </button>
            <button onClick={revertToOriginal}>Orijinale Dön</button>
          </div>
        )}

        <button
          className="advToggle"
          onClick={() => setShowAdvanced((s) => !s)}
        >
          {showAdvanced ? "▼" : "▶"} Manuel İnce Ayar
        </button>

        {showAdvanced && (
          <>
            <h3>Netleştirme</h3>
            {SHARP.map(renderSlider)}

            <h3>Işık</h3>
            {LIGHT.map(renderSlider)}

            <h3>Renk</h3>
            {COLOR.map(renderSlider)}
          </>
        )}

        <div className="hint">
          Tüm işlem tarayıcıda. Model HuggingFace CDN'inden indirilir, IndexedDB'de cache'lenir.
          Görsel sunucuya gitmez.
        </div>
      </div>
    </div>
  );
}
