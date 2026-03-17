import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ocularis",
  description: "Autonomous web agent monitor",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-zinc-950 text-zinc-100 min-h-screen font-mono antialiased">
        <header className="border-b border-zinc-800 px-6 py-3 flex items-center gap-3">
          <span className="text-lg font-semibold tracking-tight text-white">Ocularis</span>
          <span className="text-xs text-zinc-500">agent monitor</span>
        </header>
        <main className="max-w-screen-xl mx-auto px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
