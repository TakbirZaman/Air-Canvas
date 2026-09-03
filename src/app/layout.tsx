import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Air Canvas",
  description: "Draw in the air using hand-tracking, powered by MediaPipe.",
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
