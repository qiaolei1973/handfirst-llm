import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "v5: 多维输入 — 矩阵",
  description: "权重矩阵、3D 曲面、多维特征",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
