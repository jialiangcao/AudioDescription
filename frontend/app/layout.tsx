import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "adesc — audio description",
  description: "Generate audio-description narration for video.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
