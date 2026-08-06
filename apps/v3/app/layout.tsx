import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'v3：进入非线性世界',
  description: '从直线到曲线——ReLU 激活函数、隐藏层、反向传播',
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
