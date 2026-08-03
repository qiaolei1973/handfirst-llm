import 'highlight.js/styles/github.css';
import './docs.css';

export const metadata = {
  title: {
    template: '%s · HandFirst ML',
    default: 'HandFirst ML',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
