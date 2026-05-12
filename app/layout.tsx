import "./globals.css";

export const metadata = {
  title: "Lightss — Foto Netleştir & Işık Ayarla",
  description: "Bulanık fotoğrafı netleştir, ışığı ayarla. Tarayıcıda çalışır.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
