# Lightss

Tarayıcıda foto netleştir + ışık/renk ayarla. Server'a yüklenmez. Vercel uyumlu.

## Algoritma (deblur, karartmadan)

- RGB → **YCbCr** ayrıştırma
- Sadece **Y (luminance)** kanalına iteratif **Unsharp Mask**: `Y' = Y + amount * (Y - GaussianBlur(Y))`, eşik filtresi ile gürültü korunur, çoklu pas (1-3) ile güçlü deblur
- Cb/Cr dokunulmaz → renk kayması yok
- Tone: exposure (linear), highlights/shadows (luminance ağırlıklı, karartmaz), gamma, kontrast, sıcaklık/tint, doygunluk + vibrance, clarity (lokal kontrast)

## Çalıştır

```bash
npm install
npm run dev
```

## Vercel'e deploy

```bash
npm i -g vercel
vercel
```

Veya GitHub'a push et → Vercel'de Import.
