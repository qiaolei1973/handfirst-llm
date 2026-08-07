import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'v4：优化曲线',
  description: '标准化、Adam 优化器、训练/验证分离、Early Stopping',
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
