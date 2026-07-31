import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '🔬 梯度下降几何解剖',
  description: '看每一步：数据空间里的直线怎么转，loss 曲面上的点怎么滑向谷底',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
