"use client";

import "./ImageViewer.css";

import { useCallback, useEffect, useRef, useState } from "react";

export interface ImageViewerProps {
  html: string;
}

/**
 * Client component that renders markdown HTML and enables:
 * - Image lightbox (click to zoom, Escape to close)
 * - Image loading placeholders (skeleton pulse until loaded)
 */
export function ImageViewer({ html }: ImageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomed, setZoomed] = useState<string | null>(null);

  // ── Lightbox: event delegation on <img> click ──
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: MouseEvent) => {
      const img = (e.target as HTMLElement).closest("img");
      if (img?.src) setZoomed(img.src);
    };
    el.addEventListener("click", handler);
    return () => el.removeEventListener("click", handler);
  }, []);

  const close = useCallback(() => setZoomed(null), []);

  useEffect(() => {
    if (!zoomed) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [zoomed, close]);

  // ── Image loading placeholders ──
  // After HTML is injected, wrap each <img> in a container with a loading skeleton.
  // The placeholder auto-hides once the image loads or errors.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const imgs = el.querySelectorAll<HTMLImageElement>("img:not(.lightbox-img)");
    const wrappers: HTMLElement[] = [];

    for (const img of imgs) {
      // Skip if already wrapped
      if (img.parentElement?.classList.contains("img-loading-wrapper")) continue;

      const wrapper = document.createElement("div");
      wrapper.className = "img-loading-wrapper";

      const placeholder = document.createElement("div");
      placeholder.className = "img-loading-placeholder";
      placeholder.innerHTML =
        `<span class="img-loading-icon">&#8203;</span>` +
        `<span class="img-loading-text">loading...</span>`;

      // Preserve the img's natural placement
      img.style.opacity = "0";
      img.style.transition = "opacity 0.3s ease";

      img.parentElement!.insertBefore(wrapper, img);
      wrapper.appendChild(img);
      wrapper.appendChild(placeholder);
      wrappers.push(wrapper);

      const show = () => {
        img.style.opacity = "1";
        placeholder.remove();
      };

      if (img.complete) {
        show();
      } else {
        img.addEventListener("load", show, { once: true });
        img.addEventListener("error", show, { once: true });
      }
    }

    return () => {
      // Cleanup event listeners (handled by {once: true})
    };
  }, [html]);

  return (
    <>
      <article
        ref={containerRef}
        dangerouslySetInnerHTML={{ __html: html }}
      />

      {zoomed && (
        <div className="lightbox-overlay" onClick={close}>
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
