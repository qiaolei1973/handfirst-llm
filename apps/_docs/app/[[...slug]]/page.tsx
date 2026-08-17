import { getNavigation, readContentFile, renderMarkdown } from '@/lib/markdown';
import { ImageViewer } from '@handfirst/components';
import { notFound } from 'next/navigation';
import { NavSidebar } from './NavSidebar';
import { TocSidebar } from './TocSidebar';

interface PageParams {
  slug?: string[];
}

export default async function Page({ params }: { params: Promise<PageParams> }) {
  const { slug = [] } = await params;
  const { content, exists } = readContentFile(slug);

  if (!exists) notFound();

  const nav = getNavigation();
  const { html, toc } = renderMarkdown(content);

  return (
    <div className="docs-layout">
      <NavSidebar items={nav} currentPath={'/' + slug.join('/')} />
      <main className="docs-content">
        <ImageViewer html={html} />
      </main>
      <TocSidebar items={toc} />
    </div>
  );
}
