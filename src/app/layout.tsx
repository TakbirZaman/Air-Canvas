import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Air Canvas — Draw in the Air",
  description: "Draw in the air with your index finger, tracked live via webcam using MediaPipe HandLandmarker. Pinch to draw, ✌️ to undo, ✊ to clear.",
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
