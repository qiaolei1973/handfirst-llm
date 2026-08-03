import 'highlight.js/styles/github.css';
import './docs.css';

export const metadata = {
  title: {
    template: '%s · 手摸手机器学习',
    default: '手摸手机器学习',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
