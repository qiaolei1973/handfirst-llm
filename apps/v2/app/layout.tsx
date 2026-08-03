import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '🔬 v2 · MSE + SGD',
  description: 'SGD + mini-batch — 数据空间里的直线怎么转，loss 曲面上的点怎么滑向谷底',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
