'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { NavItem } from '@/lib/markdown';

export function NavSidebar({ items, currentPath }: { items: NavItem[]; currentPath: string }) {
  const pathname = usePathname();
  const active = currentPath || pathname;

  return (
    <aside className="docs-sidebar">
      <nav>
        <ul className="nav-list">
          {items.map((item) => (
            <li key={item.href} className="nav-item">
              <Link
                href={item.href}
                className={`nav-link ${active === item.href ? 'active' : ''}`}
              >
                {item.title}
              </Link>
              {item.children && item.children.length > 0 && (
                <ul className="nav-sub-list">
                  {item.children.map((child) => (
                    <li key={child.href}>
                      <Link
                        href={child.href}
                        className={`nav-link sub ${active === child.href ? 'active' : ''}`}
                      >
                        {child.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
