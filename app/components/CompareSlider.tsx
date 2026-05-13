"use client";
import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  beforeData: ImageData | null;
  afterData: ImageData | null;
};

export default function CompareSlider({ beforeData, afterData }: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const beforeRef = useRef<HTMLCanvasElement | null>(null);
  const afterRef = useRef<HTMLCanvasElement | null>(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  useEffect(() => {
    if (!beforeData || !afterData) return;
    const b = beforeRef.current, a = afterRef.current;
    if (!b || !a) return;
    // upscaled after may differ from before — paint each at native size, CSS scales
    b.width = beforeData.width; b.height = beforeData.height;
    a.width = afterData.width; a.height = afterData.height;
    b.getContext("2d")!.putImageData(beforeData, 0, 0);
    a.getContext("2d")!.putImageData(afterData, 0, 0);
  }, [beforeData, afterData]);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = wrapRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const p = ((clientX - r.left) / r.width) * 100;
    setPos(Math.max(0, Math.min(100, p)));
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => { if (dragging.current) updateFromClientX(e.clientX); };
    const onUp = () => { dragging.current = false; };
    const onTouch = (e: TouchEvent) => { if (dragging.current && e.touches[0]) updateFromClientX(e.touches[0].clientX); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onTouch, { passive: true });
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("touchend", onUp);
    };
  }, [updateFromClientX]);

  if (!beforeData || !afterData) return null;

  return (
    <div
      className="compare"
      ref={wrapRef}
      onClick={(e) => updateFromClientX(e.clientX)}
      style={{ aspectRatio: `${afterData.width} / ${afterData.height}` }}
    >
      <canvas ref={afterRef} className="cmpCanvas" />
      <canvas
        ref={beforeRef}
        className="cmpCanvas cmpBefore"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      />
      <div className="cmpLabel cmpLeft" style={{ opacity: pos > 8 ? 1 : 0 }}>ÖNCE</div>
      <div className="cmpLabel cmpRight" style={{ opacity: pos < 92 ? 1 : 0 }}>SONRA</div>
      <div
        className="cmpHandle"
        style={{ left: `${pos}%` }}
        onMouseDown={(e) => { dragging.current = true; updateFromClientX(e.clientX); e.preventDefault(); }}
        onTouchStart={(e) => { dragging.current = true; if (e.touches[0]) updateFromClientX(e.touches[0].clientX); }}
      >
        <div className="cmpLine" />
        <div className="cmpThumb">⇔</div>
      </div>
    </div>
  );
}
