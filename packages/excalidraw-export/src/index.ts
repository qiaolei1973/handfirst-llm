/**
 * excalidraw-export — 将 .excalidraw 文件导出为 SVG。
 *
 * 纯 Node.js，零依赖。不依赖浏览器、Playwright 或任何 runtime 框架。
 *
 * ## 作为库使用
 *
 * ```ts
 * import { excalidrawToSvg } from "@handfirst/excalidraw-export";
 *
 * const svg = excalidrawToSvg({
 *   elements: [...],
 *   appState: { ... },
 * });
 * writeFileSync("output.svg", svg);
 * ```
 *
 * ## 作为 CLI 使用
 *
 * ```bash
 * npx tsx packages/excalidraw-export/src/cli.ts scene.excalidraw -o output.svg
 * ```
 */

// ── 类型 ──

export interface ExcalidrawElement {
  id: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  angle?: number;
  strokeColor?: string;
  backgroundColor?: string;
  fillStyle?: string;
  strokeWidth?: number;
  strokeStyle?: string;
  roughness?: number;
  opacity?: number;
  roundness?: { type: number } | null;
  text?: string;
  fontSize?: number;
  fontFamily?: number;
  textAlign?: string;
  verticalAlign?: string;
  points?: [number, number][];
  boundElements?: { id: string; type: string }[] | null;
}

export interface ExcalidrawScene {
  elements: ExcalidrawElement[];
  appState?: Record<string, unknown>;
  files?: Record<string, unknown>;
}

export interface ExportOptions {
  /** Render a white background behind the scene. Default: true. */
  background?: boolean;
}

// ── 核心导出函数 ── //

export function excalidrawToSvg(scene: ExcalidrawScene, opts: ExportOptions = {}): string {
  const elements = scene.elements || [];
  if (elements.length === 0) {
    throw new Error("No elements found in scene");
  }

  const bg = opts.background !== false;

  // Compute canvas bounds
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const el of elements) {
    if (el.type === "arrow" || el.type === "line") {
      for (const [px, py] of el.points || []) {
        minX = Math.min(minX, el.x + px);
        minY = Math.min(minY, el.y + py);
        maxX = Math.max(maxX, el.x + px);
        maxY = Math.max(maxY, el.y + py);
      }
    }
    minX = Math.min(minX, el.x);
    minY = Math.min(minY, el.y);
    maxX = Math.max(maxX, el.x + el.width);
    maxY = Math.max(maxY, el.y + el.height);
  }

  const pad = 20;
  const vbW = maxX - minX + pad * 2;
  const vbH = maxY - minY + pad * 2;

  // Hand-drawn (sketch) filter — zero-dep SVG turbulence
  const sketchFilter = `<filter id="sketch" x="-2%" y="-2%" width="104%" height="104%">`
    + `<feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="3" result="noise"/>`
    + `<feDisplacementMap in="SourceGraphic" in2="noise" scale="2" xChannelSelector="R" yChannelSelector="G"/>`
    + `</filter>\n`;

  // Pre-collect arrow marker defs
  let defs = sketchFilter;
  for (const el of elements) {
    if (el.type !== "arrow") continue;
    const pts = el.points || [];
    if (pts.length < 2) continue;
    const col = svgColor(el.strokeColor, "#64748b");
    const markerId = `mk-${el.id}`;
    // Apply sketch filter to arrowheads too
    defs += `<marker id="${markerId}" markerWidth="18" markerHeight="14" refX="14" refY="7"`
         + ` orient="auto" markerUnits="userSpaceOnUse">`
         + `<g filter="url(#sketch)">`
         + `<path d="M 1,2 L 14,7 L 1,12" fill="none" stroke="${col}"`
         + ` stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"`
         + ` opacity="${svgOpacity(el.opacity)}"/>`
         + `</g>`
         + `</marker>\n`;
  }

  // Build SVG
  let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  svg += `<svg xmlns="http://www.w3.org/2000/svg"`
       + ` viewBox="${minX - pad} ${minY - pad} ${vbW} ${vbH}"`
       + ` width="${vbW}" height="${vbH}">\n`;
  svg += `<defs>\n${defs}</defs>\n`;
  if (bg) {
    // Background stays clean (no sketch filter — it's the canvas, not the drawing)
    svg += `<rect x="${minX - pad}" y="${minY - pad}" width="${vbW}" height="${vbH}" fill="white"/>\n`;
  }
  // All drawn elements go through the sketch filter
  svg += `<g filter="url(#sketch)">\n`;

  for (const el of elements) {
    svg += renderElement(el);
  }

  svg += `</g>\n`;

  svg += `</svg>\n`;
  return svg;
}

// ── 内部渲染函数 ──

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function svgColor(c: string | undefined, fallback: string): string {
  if (!c || c === "transparent") return "none";
  if (c === "#1e1e1e") return fallback;
  return c;
}

function svgBg(c: string | undefined): string {
  if (!c || c === "transparent") return "none";
  return c;
}

function svgOpacity(o: number | undefined): number {
  return o !== undefined ? o / 100 : 1;
}

function svgDash(s: string | undefined): string {
  if (s === "dashed") return "6,4";
  if (s === "dotted") return "2,4";
  return "none";
}

function svgTextAnchor(a: string | undefined): string {
  if (a === "center") return "middle";
  if (a === "right") return "end";
  return "start";
}

function renderElement(el: ExcalidrawElement): string {
  switch (el.type) {
    case "rectangle": return renderRect(el);
    case "ellipse":   return renderEllipse(el);
    case "arrow":     return renderArrow(el);
    case "line":      return renderLine(el);
    case "text":      return renderText(el);
    default:          return `<!-- unknown type: ${el.type} -->\n`;
  }
}

function renderText(el: ExcalidrawElement): string {
  const fs = el.fontSize || 16;
  const ff = el.fontFamily === 2 ? "Helvetica, Arial, sans-serif"
           : el.fontFamily === 1 ? "Virgil, Segoe UI Emoji"
           : "Virgil, Segoe UI Emoji";
  const lines = (el.text || "").split("\n");
  const lineH = fs * 1.3;
  const totalH = lines.length * lineH;
  const sy = el.y + (el.verticalAlign === "middle" ? (el.height - totalH) / 2 + fs * 0.8 : fs);

  let s = "";
  for (let i = 0; i < lines.length; i++) {
    const anchorX = el.textAlign === "center" ? el.x + el.width / 2
                  : el.textAlign === "right"  ? el.x + el.width
                  : el.x;
    s += `<text x="${anchorX}" y="${sy + i * lineH}"`
       + ` font-family="${ff}" font-size="${fs}"`
       + ` fill="${svgColor(el.strokeColor, "#1e293b")}"`
       + ` text-anchor="${svgTextAnchor(el.textAlign)}"`
       + ` opacity="${svgOpacity(el.opacity)}">${esc(lines[i])}</text>\n`;
  }
  return s;
}

function renderRect(el: ExcalidrawElement): string {
  if (el.text && !el.backgroundColor) {
    return renderText(el);
  }

  const rx = el.roundness?.type === 3 ? 8
           : el.roundness?.type === 2 ? 6
           : el.roundness?.type === 1 ? 4 : 0;

  let s = `<rect x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}"`
        + ` rx="${rx}"`
        + ` fill="${svgBg(el.backgroundColor)}"`
        + ` stroke="${svgColor(el.strokeColor, "#1e293b")}"`
        + ` stroke-width="${el.strokeWidth || 1}"`
        + ` stroke-dasharray="${svgDash(el.strokeStyle)}"`
        + ` opacity="${svgOpacity(el.opacity)}"`;
  if (el.angle) s += ` transform="rotate(${el.angle}, ${el.x + el.width / 2}, ${el.y + el.height / 2})"`;
  s += "/>\n";

  if (el.text) {
    s += renderText({
      ...el,
      type: "text",
      x: el.x + 6,
      y: el.y,
      width: el.width - 12,
      height: el.height,
      strokeColor: el.strokeColor,
      fontSize: el.fontSize || 14,
      textAlign: el.textAlign || "center",
      verticalAlign: el.verticalAlign || "middle",
    });
  }
  return s;
}

function renderEllipse(el: ExcalidrawElement): string {
  const cx = el.x + el.width / 2;
  const cy = el.y + el.height / 2;
  const rx = el.width / 2;
  const ry = el.height / 2;
  let s = `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"`
        + ` fill="${svgBg(el.backgroundColor)}"`
        + ` stroke="${svgColor(el.strokeColor, "#1e293b")}"`
        + ` stroke-width="${el.strokeWidth || 1}"`
        + ` opacity="${svgOpacity(el.opacity)}"/>\n`;
  if (el.text) {
    s += renderText({
      ...el, type: "text",
      x: cx - rx * 0.7, y: cy - ry * 0.5,
      width: rx * 1.4, height: ry,
      strokeColor: el.strokeColor,
      fontSize: el.fontSize || 14,
      textAlign: "center", verticalAlign: "middle",
    });
  }
  return s;
}

function renderArrow(el: ExcalidrawElement): string {
  const pts = el.points || [];
  if (pts.length < 2) return "";

  const sw = el.strokeWidth || 2;
  const col = svgColor(el.strokeColor, "#64748b");

  const pathData = pts.map((p, i) =>
    `${i === 0 ? "M" : "L"} ${el.x + p[0]} ${el.y + p[1]}`
  ).join(" ");

  return `<path d="${pathData}" fill="none"`
       + ` stroke="${col}"`
       + ` stroke-width="${sw}"`
       + ` stroke-dasharray="${svgDash(el.strokeStyle)}"`
       + ` opacity="${svgOpacity(el.opacity)}"`
       + ` marker-end="url(#mk-${el.id})"/>\n`;
}

function renderLine(el: ExcalidrawElement): string {
  const pts = el.points || [];
  if (pts.length < 2) return "";

  const pathData = pts.map((p, i) =>
    `${i === 0 ? "M" : "L"} ${el.x + p[0]} ${el.y + p[1]}`
  ).join(" ");

  return `<path d="${pathData}" fill="none"`
       + ` stroke="${svgColor(el.strokeColor, "#64748b")}"`
       + ` stroke-width="${el.strokeWidth || 1}"`
       + ` stroke-dasharray="${svgDash(el.strokeStyle)}"`
       + ` opacity="${svgOpacity(el.opacity)}"/>\n`;
}
