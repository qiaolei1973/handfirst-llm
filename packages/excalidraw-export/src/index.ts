/**
 * excalirender — 将 .excalidraw 文件导出为 SVG。
 *
 * 依赖 roughjs（Generator API，不需要浏览器 DOM）。
 *
 * ```ts
 * import { excalidrawToSvg } from "excalirender";
 * const svg = excalidrawToSvg({ elements: [...], appState: {...} });
 * writeFileSync("output.svg", svg);
 * ```
 */

import { createRequire } from "module";
const { generator: newGenerator } = createRequire(import.meta.url)("roughjs");

// ── 类型 ──

export interface ExcalidrawElement {
  id: string; type: string;
  x: number; y: number; width: number; height: number;
  angle?: number; strokeColor?: string; backgroundColor?: string;
  fillStyle?: string; strokeWidth?: number; strokeStyle?: string;
  roughness?: number; opacity?: number;
  roundness?: { type: number } | null;
  text?: string; fontSize?: number; fontFamily?: number;
  textAlign?: string; verticalAlign?: string;
  points?: [number, number][];
  boundElements?: { id: string; type: string }[] | null;
}

export interface ExcalidrawScene {
  elements: ExcalidrawElement[];
  appState?: Record<string, unknown>;
  files?: Record<string, unknown>;
}

export interface ExportOptions {
  background?: boolean;   // default: true
  sketch?: boolean;       // default: true (roughjs hand-drawn style)
}

// ── 核心导出 ──

export function excalidrawToSvg(scene: ExcalidrawScene, opts: ExportOptions = {}): string {
  const elements = scene.elements || [];
  if (elements.length === 0) throw new Error("No elements found in scene");

  const bg = opts.background !== false;
  const sketch = opts.sketch !== false;

  // Canvas bounds
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const el of elements) {
    if (el.type === "arrow" || el.type === "line") {
      for (const [px, py] of el.points || []) {
        minX = Math.min(minX, el.x + px); maxX = Math.max(maxX, el.x + px);
        minY = Math.min(minY, el.y + py); maxY = Math.max(maxY, el.y + py);
      }
    }
    minX = Math.min(minX, el.x); maxX = Math.max(maxX, el.x + el.width);
    minY = Math.min(minY, el.y); maxY = Math.max(maxY, el.y + el.height);
  }

  const pad = 24;
  const vbW = maxX - minX + pad * 2;
  const vbH = maxY - minY + pad * 2;

  // Arrowhead markers
  let defs = "";
  for (const el of elements) {
    if (el.type !== "arrow") continue;
    const pts = el.points || [];
    if (pts.length < 2) continue;
    const col = svgColor(el.strokeColor, "#64748b");
    defs += `<marker id="mk-${el.id}" markerWidth="18" markerHeight="14" refX="14" refY="7"`
         + ` orient="auto" markerUnits="userSpaceOnUse">`
         + `<path d="M 1,2 L 14,7 L 1,12" fill="none" stroke="${col}"`
         + ` stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`
         + `</marker>\n`;
  }

  // Build SVG
  let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  svg += `<svg xmlns="http://www.w3.org/2000/svg"`
       + ` viewBox="${minX - pad} ${minY - pad} ${vbW} ${vbH}"`
       + ` width="${vbW}" height="${vbH}">\n`;
  // Very subtle text sketch filter — just enough to break mechanical perfection
  defs += `<filter id="text-sketch">`
       + `<feTurbulence type="fractalNoise" baseFrequency="0.06" numOctaves="2" seed="7" result="noise"/>`
       + `<feDisplacementMap in="SourceGraphic" in2="noise" scale="0.8" xChannelSelector="R" yChannelSelector="G"/>`
       + `</filter>\n`;

  if (defs) svg += `<defs>\n${defs}</defs>\n`;
  if (bg) {
    svg += `<rect x="${minX - pad}" y="${minY - pad}" width="${vbW}" height="${vbH}" fill="white"/>\n`;
  }

  for (const el of elements) {
    svg += render(el, sketch);
  }

  svg += `</svg>\n`;
  return svg;
}

// ── 每个元素渲染 ──

const NOS = "none";
const gen = newGenerator();

function render(el: ExcalidrawElement, sketch: boolean): string {
  switch (el.type) {
    case "rectangle": return renderRect(el, sketch);
    case "ellipse":   return renderEllipse(el, sketch);
    case "arrow":     return renderArrow(el, sketch);
    case "line":      return renderLine(el, sketch);
    case "text":      return renderText(el);
    default:          return `<!-- unknown: ${el.type} -->\n`;
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
  let s = `<g filter="url(#text-sketch)">\n`;
  for (let i = 0; i < lines.length; i++) {
    const ax = el.textAlign === "center" ? el.x + el.width / 2
             : el.textAlign === "right" ? el.x + el.width : el.x;
    s += `<text x="${ax.toFixed(1)}" y="${(sy + i * lineH).toFixed(1)}"`
       + ` font-family="${ff}" font-size="${fs}"`
       + ` fill="${svgColor(el.strokeColor, "#1e293b")}"`
       + ` text-anchor="${svgTA(el.textAlign)}" opacity="${svgOp(el.opacity)}">${esc(lines[i])}</text>\n`;
  }
  s += `</g>\n`;
  return s;
}

function renderRect(el: ExcalidrawElement, sketch: boolean): string {
  if (el.text && !el.backgroundColor) return renderText(el);

  const fill = svgBg(el.backgroundColor);
  const stroke = svgColor(el.strokeColor, "#1e293b");
  const sw = el.strokeWidth || 1;
  const op = svgOp(el.opacity);
  const rx = el.roundness?.type === 3 ? 8 : el.roundness?.type === 2 ? 6 : el.roundness?.type === 1 ? 4 : 0;

  if (!sketch) {
    return `<rect x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}" rx="${rx}"`
         + ` fill="${fill}" stroke="${stroke}" stroke-width="${sw}" opacity="${op}"/>\n`
         + (el.text ? innerText(el) : "");
  }

  // roughjs: generate hachure + multi-pass stroke paths
  const seed = hash32(el.id);
  const drawable = gen.rectangle(el.x, el.y, el.width, el.height, {
    seed, roughness: 1.2, bowing: 0.8,
    stroke, strokeWidth: sw,
    fill: fill !== NOS ? fill : undefined,
    fillStyle: "hachure",
    fillWeight: sw * 0.5,
    hachureAngle: 60,
    hachureGap: sw * 3,
  });

  return drawableToSvg(drawable, op) + (el.text ? innerText(el) : "");
}

function renderEllipse(el: ExcalidrawElement, sketch: boolean): string {
  const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
  const rx = el.width / 2, ry = el.height / 2;
  const fill = svgBg(el.backgroundColor);
  const stroke = svgColor(el.strokeColor, "#1e293b");
  const sw = el.strokeWidth || 1;
  const op = svgOp(el.opacity);

  if (!sketch) {
    return `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${fill}" stroke="${stroke}"`
         + ` stroke-width="${sw}" opacity="${op}"/>\n`
         + (el.text ? innerText(el) : "");
  }

  const seed = hash32(el.id);
  const drawable = gen.ellipse(cx, cy, rx * 2, ry * 2, {
    seed, roughness: 1.2, bowing: 0.8,
    stroke, strokeWidth: sw,
    fill: fill !== NOS ? fill : undefined,
    fillStyle: "hachure",
    fillWeight: sw * 0.5,
    hachureAngle: 60,
    hachureGap: sw * 3,
  });

  return drawableToSvg(drawable, op) + (el.text ? innerText(el) : "");
}

function renderArrow(el: ExcalidrawElement, sketch: boolean): string {
  const pts = el.points || [];
  if (pts.length < 2) return "";
  const sw = el.strokeWidth || 2;
  const col = svgColor(el.strokeColor, "#64748b");
  const op = svgOp(el.opacity);
  const absPts = pts.map(p => [el.x + p[0], el.y + p[1]] as [number, number]);

  let d: string;
  if (sketch) {
    const seed = hash32(el.id);
    const drawable = gen.linearPath(absPts, { seed, roughness: 1.2, bowing: 0.5, stroke: col, strokeWidth: sw });
    // Only the 'path' sets (not fill)
    d = drawable.sets.filter(s => s.type === "path").map(s => gen.opsToPath(s, 1)).join(" ");
  } else {
    d = absPts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  }

  return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${sw}"`
       + ` stroke-linecap="round" opacity="${op}"`
       + ` marker-end="url(#mk-${el.id})"/>\n`;
}

function renderLine(el: ExcalidrawElement, sketch: boolean): string {
  const pts = el.points || [];
  if (pts.length < 2) return "";
  const sw = el.strokeWidth || 1;
  const col = svgColor(el.strokeColor, "#64748b");
  const op = svgOp(el.opacity);
  const absPts = pts.map(p => [el.x + p[0], el.y + p[1]] as [number, number]);

  let d: string;
  if (sketch) {
    const seed = hash32(el.id);
    const drawable = gen.linearPath(absPts, { seed, roughness: 1.2, bowing: 0.5, stroke: col, strokeWidth: sw });
    d = drawable.sets.filter(s => s.type === "path").map(s => gen.opsToPath(s, 1)).join(" ");
  } else {
    d = absPts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  }

  return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${sw}"`
       + ` stroke-linecap="round" opacity="${op}"`
       + ` stroke-dasharray="${svgDash(el.strokeStyle)}"/>\n`;
}

// ── roughjs Drawable → SVG strings ──

function drawableToSvg(d: any, opacity: number): string {
  let s = "";
  const options = d.options || {};
  for (const set of d.sets) {
    const pathData = gen.opsToPath(set, 1);
    switch (set.type) {
      case "path":
        s += `<path d="${pathData}" fill="none" stroke="${options.stroke}"`
           + ` stroke-width="${options.strokeWidth}" stroke-linecap="round" stroke-linejoin="round"`
           + ` opacity="${opacity}"/>\n`;
        break;
      case "fillPath":
        s += `<path d="${pathData}" fill="${options.fill || NOS}" stroke="none" opacity="${opacity}"/>\n`;
        break;
      case "fillSketch":
        s += `<path d="${pathData}" fill="none" stroke="${options.fill || NOS}"`
           + ` stroke-width="${options.fillWeight || 1}" stroke-linecap="round"`
           + ` opacity="${opacity}"/>\n`;
        break;
    }
  }
  return s;
}

// ── 辅助 ──

function innerText(el: ExcalidrawElement): string {
  return renderText({ ...el, type: "text", x: el.x + 6, y: el.y,
    width: el.width - 12, height: el.height,
    strokeColor: el.strokeColor, fontSize: el.fontSize || 14,
    textAlign: el.textAlign || "center", verticalAlign: el.verticalAlign || "middle" });
}

function hash32(s: string): number {
  let h = 0x6d2b79f5;
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(h ^ s.charCodeAt(i), 0x5bd1e995);
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
  }
  return h ^ (h >>> 15);
}

function esc(s: string): string { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function svgColor(c: string | undefined, fb: string): string { return (!c || c === "transparent") ? NOS : c === "#1e1e1e" ? fb : c; }
function svgBg(c: string | undefined): string { return (!c || c === "transparent") ? NOS : c; }
function svgOp(o: number | undefined): number { return o !== undefined ? o / 100 : 1; }
function svgDash(s: string | undefined): string { return s === "dashed" ? "8,4" : s === "dotted" ? "2,4" : NOS; }
function svgTA(a: string | undefined): string { return a === "center" ? "middle" : a === "right" ? "end" : "start"; }
