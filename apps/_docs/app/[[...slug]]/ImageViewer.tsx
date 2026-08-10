"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  html: string;
}

export function ImageViewer({ html }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomed, setZoomed] = useState<string | null>(null);

  // Event delegation — capture clicks on any <img> inside the article
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handler = (e: MouseEvent) => {
      const img = (e.target as HTMLElement).closest("img");
      if (img?.src) {
        setZoomed(img.src);
      }
    };

    el.addEventListener("click", handler);
    return () => el.removeEventListener("click", handler);
  }, []);

  const close = useCallback(() => setZoomed(null), []);

  // Close on Escape
  useEffect(() => {
    if (!zoomed) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [zoomed, close]);

  return (
    <>
      <article
        ref={containerRef}
        dangerouslySetInnerHTML={{ __html: html }}
      />

      {zoomed && (
        <div className="lightbox-overlay" onClick={close}>
          {/* Close button */}
          <button className="lightbox-close" onClick={close} aria-label="关闭">
            ✕
          </button>
          <img
            src={zoomed}
            alt=""
            className="lightbox-img"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
