import type { Metadata } from "next";
import { Bricolage_Grotesque, IBM_Plex_Mono, Public_Sans } from "next/font/google";
import AuthGate from "./components/AuthGate";
import "./globals.css";

// Three roles: a grotesque with real character for the product voice, a
// workhorse for reading (Public Sans comes out of the US design system's
// accessibility work, which is the point of this app), and a mono for
// timecodes and counts, which must line up.
const display = Bricolage_Grotesque({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

const body = Public_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "BuddyWatch",
  description:
    "Add spoken description to any video, then ask questions about what happens in it.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${body.variable} ${mono.variable}`}
    >
      <body>
        <AuthGate>{children}</AuthGate>
      </body>
    </html>
  );
}
