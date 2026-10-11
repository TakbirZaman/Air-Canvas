import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  viewportFit: "cover",
  themeColor: "#050508",
};

export const metadata: Metadata = {
  title: "Air Canvas — Draw in the Air",
  description: "Draw in the air with your index finger, tracked live via webcam using MediaPipe HandLandmarker. Pinch to draw, 🖐 to erase, ✌️ to undo, ✊ to clear. Touch fallback included. Crafted by Takbir Zaman Bhuiyan — https://www.linkedin.com/in/takbir-zaman-bhuiyan/",
  authors: [{ name: "Takbir Zaman Bhuiyan", url: "https://www.linkedin.com/in/takbir-zaman-bhuiyan/" }],
  creator: "Takbir Zaman Bhuiyan",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "Air Canvas" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
