import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { marked, Renderer } from 'marked';
import hljs from 'highlight.js';

const CONTENT_DIR = join(process.cwd(), 'content');

// ---- 配置 marked + highlight.js ----
const renderer = new Renderer();
renderer.code = function (token: { text: string; lang?: string }) {
  const lang = token.lang || '';
  const language = hljs.getLanguage(lang) ? lang : undefined;
  const highlighted = language
    ? hljs.highlight(token.text, { language }).value
    : hljs.highlightAuto(token.text).value;
  return `<pre><code class="hljs ${language || ''}">${highlighted}</code></pre>`;
};

marked.use({ gfm: true, renderer });

// ---- MD → HTML（同步）----
export function renderMarkdown(content: string): string {
  return marked.parse(content) as string;
}

// ---- 读取 content 目录下的 .md 文件 ----
export function readContentFile(slug: string[]): { content: string; exists: boolean } {
  const slugPath = slug.length === 0 ? ['index'] : slug;
  // 先试直接文件: content/v1.md
  const filePath = join(CONTENT_DIR, ...slugPath) + '.md';
  // 再试文件夹 index: content/v1/index.md
  const indexInDir = join(CONTENT_DIR, ...slugPath, 'index.md');

  for (const p of [filePath, indexInDir]) {
    try {
      return { content: readFileSync(p, 'utf-8'), exists: true };
    } catch { /* 下一个 */ }
  }
  return { content: '', exists: false };
}

// ---- 扫描目录获取导航 ----
export interface NavItem {
  title: string;
  href: string;
  children?: NavItem[];
}

export function getNavigation(): NavItem[] {
  const items: NavItem[] = [];

  try {
    readFileSync(join(CONTENT_DIR, 'index.md'));
    items.push({ title: '总览', href: '/' });
  } catch { /* skip */ }

  try {
    const dirs = readdirSync(CONTENT_DIR)
      .filter((d) => {
        try { return statSync(join(CONTENT_DIR, d)).isDirectory(); } catch { return false; }
      })
      .sort((a, b) => {
        const ai = DIR_ORDER.indexOf(a);
        const bi = DIR_ORDER.indexOf(b);
        if (ai === -1 && bi === -1) return a.localeCompare(b);
        if (ai === -1) return 1;
        if (bi === -1) return -1;
        return ai - bi;
      });

    for (const dir of dirs) {
      try {
        readFileSync(join(CONTENT_DIR, dir, 'index.md'));
        const title = DIR_TITLES[dir] || dir;
        items.push({
          title,
          href: `/${dir}`,
          children: getSubPages(dir),
        });
      } catch { /* skip */ }
    }
  } catch { /* skip */ }

  return items;
}

const DIR_ORDER = ['v1', 'v2', 'v3', 'v4', 'recap'];

const DIR_TITLES: Record<string, string> = {
  v1: 'v1：猜一条直线',
  v2: 'v2：进入机器学习的世界',
  v3: 'v3：画曲线',
  v4: 'v4：优化曲线',
  recap: '复习：机器学习是什么',
};

function getSubPages(dir: string): NavItem[] | undefined {
  const pages: NavItem[] = [];
  try {
    const files = readdirSync(join(CONTENT_DIR, dir))
      .filter((f) => f.endsWith('.md') && f !== 'index.md')
      .sort();
    for (const file of files) {
      const slug = file.replace(/\.md$/, '');
      pages.push({ title: fileNameToTitle(slug), href: `/${dir}/${slug}` });
    }
  } catch { /* skip */ }
  return pages.length > 0 ? pages : undefined;
}

function fileNameToTitle(name: string): string {
  return name.replace(/-/g, ' ');
}
