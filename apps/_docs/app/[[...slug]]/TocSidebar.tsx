'use client';

import { useEffect, useState } from 'react';
import type { TocItem } from '@/lib/markdown';

/**
 * 右侧章节大纲：h2/h3 列表，点击平滑滚到对应标题，滚动时高亮当前位置。
 *
 * scrollspy 用「最后一个越过顶部线的标题」判定（标题在文档里自上而下，
 * getBoundingClientRect().top 单调递增），比 IntersectionObserver 的分段式
 * rootMargin 更稳——章节很长时中间不会出现「无高亮」的空档。
 */
export function TocSidebar({ items }: { items: TocItem[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    if (items.length === 0) return;

    const els = items
      .map((it) => document.getElementById(it.id))
      .filter((el): el is HTMLElement => el !== null);
    if (els.length === 0) return;

    const threshold = 120; // 标题顶部越过视口顶部 120px 就算「当前节」

    const onScroll = () => {
      // 页面最顶端（还没任何标题越过阈值）时，默认高亮第一个标题，避免首屏无高亮
      let currentId: string | null = items[0]?.id ?? null;
      for (let i = 0; i < els.length; i++) {
        if (els[i].getBoundingClientRect().top <= threshold) {
          currentId = items[i].id;
        } else {
          break; // 标题按文档顺序排列，一旦在阈值下方，后面的都更靠下
        }
      }
      setActiveId(currentId);
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [items]);

  if (items.length === 0) return null;

  return (
    <aside className="docs-toc">
      <nav>
        <div className="docs-toc-title">本页目录</div>
        <ul className="toc-list">
          {items.map((item) => (
            <li key={item.id} className={`toc-item ${item.level === 3 ? 'sub' : ''}`}>
              <a
                href={`#${item.id}`}
                title={item.text}
                className={`toc-link ${activeId === item.id ? 'active' : ''}`}
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById(item.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }}
              >
                {item.text}
              </a>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
