"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { applyPipeline, defaults, Settings } from "./lib/process";
import { aiDefaults, aiEnhance, AiOpts } from "./lib/ai";
import {
  neuralDefaults,
  neuralEnhance,
  NEURAL_MODELS,
  NeuralOpts,
  NeuralModelId,
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

type Mode = "algorithmic" | "neural";

// Light polish after neural SR: small CAS-like sharpen + vibrance via existing pipeline.
const POLISH: Settings = {
  ...defaults,
  sharpAmount: 0.6, sharpRadius: 1.0, sharpThreshold: 2, sharpPasses: 1,
  clarity: 0.15, shadows: 0.08, highlights: -0.03, vibrance: 0.15,
};

export default function Page() {
  const [settings, setSettings] = useState<Settings>(defaults);
  const [aiOpts, setAiOpts] = useState<AiOpts>(aiDefaults);
  const [neuralOpts, setNeuralOpts] = useState<NeuralOpts>(neuralDefaults);
  const [mode, setMode] = useState<Mode>("algorithmic");
  const [neuralPolish, setNeuralPolish] = useState(true);
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
      // analyze + auto-tune
      try {
        const t0 = performance.now();
        const s = analyzeImage(id);
        console.log("[analyze] in", (performance.now() - t0).toFixed(1), "ms", s);
        setStats(s);
        if (autoTune) {
          setAiOpts(autoTuneAi(s));
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
    setAiOpts(autoTuneAi(s));
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
      let result: ImageData;
      if (mode === "neural") {
        result = await neuralEnhance(originalRef.current, neuralOpts, (stage, pct) => {
          setAiStage(stage); setAiPct(Math.round(pct));
        });
        if (neuralPolish) {
          setAiStage("post-polish"); setAiPct(99);
          result = applyPipeline(result, POLISH);
        }
      } else {
        result = await aiEnhance(originalRef.current, aiOpts, (stage, pct) => {
          setAiStage(stage); setAiPct(Math.round(pct));
        });
      }
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
  const updateAi = (k: keyof AiOpts, v: number | boolean) => setAiOpts((s) => ({ ...s, [k]: v }));
  const updateNeural = <K extends keyof NeuralOpts>(k: K, v: NeuralOpts[K]) =>
    setNeuralOpts((s) => ({ ...s, [k]: v }));

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

  const selectedModel = NEURAL_MODELS.find((m) => m.id === neuralOpts.modelId) ?? NEURAL_MODELS[0];

  return (
    <div className="app">
      <div className="canvasWrap">
        {!hasImage && <div className="hint">Resim yükle → AI ile netleştir → karşılaştır → indir.</div>}
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
            <div className="overlayMeta">{aiElapsed}s geçti · konsol (F12) detay</div>
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
                {(stats.blurScore * 100).toFixed(0)}% (L={stats.laplacianVar.toFixed(0)})
              </b></div>
              <div><span>Gürültü</span><b className={stats.noiseScore > 0.5 ? "bad" : stats.noiseScore > 0.2 ? "med" : "good"}>
                {(stats.noiseScore * 100).toFixed(0)}%
              </b></div>
              <div><span>Parlaklık</span><b>{stats.meanLum.toFixed(0)}/255</b></div>
              <div><span>Kontrast</span><b>{stats.stdLum.toFixed(0)}</b></div>
              <div><span>Karanlık</span><b>{(stats.darkFrac * 100).toFixed(0)}%</b></div>
            </div>
            <label className="check">
              <input type="checkbox" checked={autoTune} onChange={(e) => setAutoTune(e.target.checked)} />
              <span>Yeni resimde otomatik ayarla</span>
            </label>
          </div>
        )}

        <div className="modeTabs">
          <button
            className={mode === "algorithmic" ? "tab active" : "tab"}
            onClick={() => setMode("algorithmic")}
          >Deblur + Sharpen</button>
          <button
            className={mode === "neural" ? "tab active" : "tab"}
            onClick={() => setMode("neural")}
          >Neural SR (2x/4x)</button>
        </div>
        <div className="hint">
          {mode === "algorithmic"
            ? "Bulanık fotoğraflar için: Richardson-Lucy deconvolution + cascade CAS sharpen. Gerçek deblur."
            : "Düşük çözünürlük için: Swin2SR super-resolution modeli. Detay ekler ama blur'u açmaz."}
        </div>

        <button
          className="aiBtn"
          onClick={runAi}
          disabled={!hasImage || aiBusy}
          title={mode === "neural"
            ? "Swin2SR ONNX modeli tarayıcıda çalışır. İlk seferde model indirilir."
            : "Bilateral + Lanczos + CAS + CLAHE + Clarity + Vibrance"}
        >
          {aiBusy
            ? `${aiStage} — %${aiPct}`
            : mode === "neural" ? "AI NETLEŞTİR (NEURAL)" : "AI NETLEŞTİR"}
        </button>
        {aiBusy && (
          <div className="progress"><div style={{ width: `${aiPct}%` }} /></div>
        )}

        {hasImage && (beforeData && afterData) && (
          <div className="btns">
            <button onClick={() => setShowCompare((s) => !s)}>
              {showCompare ? "Karşılaştırmayı Gizle" : "Karşılaştır"}
            </button>
            <button onClick={revertToOriginal}>Orijinale Dön</button>
          </div>
        )}

        {mode === "neural" ? (
          <>
            <h3>Neural Ayarları</h3>
            <div className="row">
              <label><span>Model</span><span className="dim">{selectedModel.scale}x · ~{selectedModel.approxSizeMB}MB</span></label>
              <select
                value={neuralOpts.modelId}
                onChange={(e) => updateNeural("modelId", e.target.value as NeuralModelId)}
                disabled={aiBusy}
              >
                {NEURAL_MODELS.map((m) => (
                  <option key={m.id} value={m.id}>{m.label}</option>
                ))}
              </select>
              <div className="hint">{selectedModel.desc}</div>
            </div>

            <div className="row">
              <label>
                <span>Maks. girdi (uzun kenar)</span>
                <span>{neuralOpts.maxInput} px</span>
              </label>
              <input
                type="range" min={256} max={1536} step={64}
                value={neuralOpts.maxInput}
                onChange={(e) => updateNeural("maxInput", parseInt(e.target.value))}
                disabled={aiBusy}
              />
              <div className="hint">Büyük girdi = daha çok detay ama daha çok RAM ve süre. 512-768 tatlı nokta.</div>
            </div>

            <div className="row">
              <label>
                <span>Tile boyutu</span>
                <span>{neuralOpts.tileSize > 0 ? `${neuralOpts.tileSize} px` : "tek seferde"}</span>
              </label>
              <input
                type="range" min={0} max={384} step={32}
                value={neuralOpts.tileSize}
                onChange={(e) => updateNeural("tileSize", parseInt(e.target.value))}
                disabled={aiBusy}
              />
              <div className="hint">0 = tek seferde (küçük resim). Büyükler için 128-256.</div>
            </div>

            <div className="row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={neuralPolish}
                  onChange={(e) => setNeuralPolish(e.target.checked)}
                  disabled={aiBusy}
                />
                <span>Sonrası polish (hafif sharpen + vibrance)</span>
              </label>
            </div>
          </>
        ) : (
          <>
            <h3>Algoritmik Ayarlar</h3>
            <div className="row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={aiOpts.upscale}
                  onChange={(e) => updateAi("upscale", e.target.checked)}
                />
                <span>2x Büyüt (Lanczos)</span>
              </label>
            </div>
            {([
              { key: "deblur", label: "Deblur (Richardson-Lucy)" },
              { key: "denoise", label: "Gürültü temizle (Bilateral)" },
              { key: "sharpen", label: "Sharpen (CAS, cascade)" },
              { key: "clarity", label: "Clarity" },
              { key: "autoExposure", label: "Oto pozlama (CLAHE)" },
              { key: "vibrance", label: "Vibrance" },
            ] as { key: keyof AiOpts; label: string }[]).map((r) => {
              const v = aiOpts[r.key] as number;
              return (
                <div className="row" key={r.key}>
                  <label>
                    <span>{r.label}</span>
                    <span>{v.toFixed(2)}</span>
                  </label>
                  <input
                    type="range" min={0} max={1} step={0.02}
                    value={v}
                    onChange={(e) => updateAi(r.key, parseFloat(e.target.value))}
                  />
                </div>
              );
            })}
          </>
        )}

        <h3>Netleştirme (manuel)</h3>
        {SHARP.map(renderSlider)}

        <h3>Işık</h3>
        {LIGHT.map(renderSlider)}

        <h3>Renk</h3>
        {COLOR.map(renderSlider)}

        <div className="hint">
          Tüm işlem tarayıcıda. Neural model dosyası HuggingFace CDN'inden indirilir, tarayıcı önbelleğine alınır.
          Görsel sunucuya gitmez.
        </div>
      </div>
    </div>
  );
}
