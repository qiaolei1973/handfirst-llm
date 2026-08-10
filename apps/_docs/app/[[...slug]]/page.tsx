import { getNavigation, readContentFile, renderMarkdown } from '@/lib/markdown';
import { notFound } from 'next/navigation';
import { NavSidebar } from './NavSidebar';
import { ImageViewer } from './ImageViewer';

interface PageParams {
  slug?: string[];
}

export default async function Page({ params }: { params: Promise<PageParams> }) {
  const { slug = [] } = await params;
  const { content, exists } = readContentFile(slug);

  if (!exists) notFound();

  const nav = getNavigation();
  const html = renderMarkdown(content);

  return (
    <div className="docs-layout">
      <NavSidebar items={nav} currentPath={'/' + slug.join('/')} />
      <main className="docs-content">
        <ImageViewer html={html} />
      </main>
    </div>
  );
}
