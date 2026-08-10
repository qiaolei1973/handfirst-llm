/**
 * excalidraw-export — 将 .excalidraw 文件导出为 SVG。
 *
 * 纯 Node.js，零依赖。不依赖浏览器、Playwright 或任何 runtime 框架。
 *
 * ## 作为库使用
 *
 * ```ts
 * import { excalidrawToSvg } from "excalirender";
 *
 * const svg = excalidrawToSvg({ elements: [...] });
 * writeFileSync("output.svg", svg);
 * ```
 */

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
  background?: boolean;
  sketch?: boolean;  // default: true
}

// ── 确定性 PRNG（基于元素 id 的 hash） ── //

function hash32(s: string): number {
  let h = 0x6d2b79f5;
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(h ^ s.charCodeAt(i), 0x5bd1e995);
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995);
  }
  return h ^ (h >>> 15);
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── 手绘抖动核心 ── //

/** 把一组控制点变成带手绘抖动的 path d 字符串 */
function sketchPolyline(pts: [number, number][], closed: boolean, rng: () => number): string {
  if (pts.length < 2) return "";
  const segmentsPerEdge = 4;
  const jitterAmount = 2.5;  // max displacement in px

  const result: [number, number][] = [pts[0]];

  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[i + 1];
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) continue;
    // unit normal (perpendicular)
    const nx = -dy / len, ny = dx / len;

    for (let s = 1; s <= segmentsPerEdge; s++) {
      const t = s / (segmentsPerEdge + 1);
      const bx = x1 + dx * t;
      const by = y1 + dy * t;
      const j = (rng() - 0.5) * jitterAmount * 2;
      result.push([bx + nx * j, by + ny * j]);
    }
    result.push([x2, y2]);
  }

  const cmds = result.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0].toFixed(2)} ${p[1].toFixed(2)}`);
  if (closed && pts.length > 2) cmds.push("Z");
  return cmds.join(" ");
}

/** 把矩形画成带手绘抖动的闭合 path */
function sketchRect(x: number, y: number, w: number, h: number, rx: number, rng: () => number): string {
  const r = Math.min(rx, w / 2, h / 2);
  const corners: [number, number][] = [
    [x + r, y], [x + w - r, y],           // top edge
    [x + w, y + r], [x + w, y + h - r],   // right edge
    [x + w - r, y + h], [x + r, y + h],   // bottom edge
    [x, y + h - r], [x, y + r],           // left edge
  ];
  // add overshoot at corners — move each corner slightly past its ideal position
  const overshoot = 1.8;
  const jittered: [number, number][] = [];
  for (const [cx, cy] of corners) {
    // slight corner overshoot
    const ox = cx + (rng() - 0.5) * overshoot;
    const oy = cy + (rng() - 0.5) * overshoot;
    jittered.push([ox, oy]);
  }
  // But we should handle arcs at corners... simple approach: just use the jittered polyline
  // with intermediate subdivision
  return sketchPolyline(jittered, true, rng);
}

/** 椭圆抖动 */
function sketchEllipse(cx: number, cy: number, rx: number, ry: number, rng: () => number): string {
  const n = 24; // segments
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const jx = (rng() - 0.5) * 2;
    const jy = (rng() - 0.5) * 2;
    pts.push([cx + (rx + jx) * Math.cos(a), cy + (ry + jy) * Math.sin(a)]);
  }
  return sketchPolyline(pts, true, rng);
}

// ── 核心导出函数 ── //

export function excalidrawToSvg(scene: ExcalidrawScene, opts: ExportOptions = {}): string {
  const elements = scene.elements || [];
  if (elements.length === 0) throw new Error("No elements found in scene");

  const bg = opts.background !== false;
  const sketch = opts.sketch !== false;

  // Compute canvas bounds
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

  const pad = 24; // slightly more to accommodate wobble
  const vbW = maxX - minX + pad * 2;
  const vbH = maxY - minY + pad * 2;

  // Pre-collect arrowhead marker defs
  let defs = "";
  for (const el of elements) {
    if (el.type !== "arrow") continue;
    const pts = el.points || [];
    if (pts.length < 2) continue;
    const col = svgColor(el.strokeColor, "#64748b");
    const markerId = `mk-${el.id}`;
    const rng = mulberry32(hash32(`${el.id}-head`));
    const headPath = sketchPolyline([[1, 2], [14, 7], [1, 12]], false, rng);
    defs += `<marker id="${markerId}" markerWidth="18" markerHeight="14" refX="14" refY="7"`
         + ` orient="auto" markerUnits="userSpaceOnUse">`
         + `<path d="${headPath}" fill="none" stroke="${col}"`
         + ` stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"`
         + ` opacity="${svgOpacity(el.opacity)}"/>`
         + `</marker>\n`;
  }

  // Build SVG
  let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  svg += `<svg xmlns="http://www.w3.org/2000/svg"`
       + ` viewBox="${minX - pad} ${minY - pad} ${vbW} ${vbH}"`
       + ` width="${vbW}" height="${vbH}">\n`;
  svg += `<defs>\n${defs}</defs>\n`;
  if (bg) {
    svg += `<rect x="${minX - pad}" y="${minY - pad}" width="${vbW}" height="${vbH}" fill="white"/>\n`;
  }

  for (const el of elements) {
    svg += renderElement(el, sketch);
  }

  svg += `</svg>\n`;
  return svg;
}

// ── 按元素类型渲染 ── //

function renderElement(el: ExcalidrawElement, sketch: boolean): string {
  switch (el.type) {
    case "rectangle": return renderRect(el, sketch);
    case "ellipse":   return renderEllipse(el, sketch);
    case "arrow":     return renderArrow(el, sketch);
    case "line":      return renderLine(el, sketch);
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

function renderRect(el: ExcalidrawElement, sketch: boolean): string {
  if (el.text && !el.backgroundColor) return renderText(el);

  const rx = el.roundness?.type === 3 ? 8
           : el.roundness?.type === 2 ? 6
           : el.roundness?.type === 1 ? 4 : 0;
  const sw = el.strokeWidth || 1;
  const col = svgColor(el.strokeColor, "#1e293b");
  const fill = svgBg(el.backgroundColor);
  const op = svgOpacity(el.opacity);

  let s: string;
  if (sketch) {
    const rng = mulberry32(hash32(el.id));
    const d = sketchRect(el.x, el.y, el.width, el.height, rx, rng);
    s = `<path d="${d}" fill="${fill}" stroke="${col}" stroke-width="${sw}"`
      + ` stroke-linecap="round" stroke-linejoin="round" opacity="${op}"`
      + ` stroke-dasharray="${svgDash(el.strokeStyle)}"/>\n`;
  } else {
    s = `<rect x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}" rx="${rx}"`
      + ` fill="${fill}" stroke="${col}" stroke-width="${sw}"`
      + ` stroke-dasharray="${svgDash(el.strokeStyle)}" opacity="${op}"`;
    if (el.angle) s += ` transform="rotate(${el.angle}, ${el.x + el.width / 2}, ${el.y + el.height / 2})"`;
    s += "/>\n";
  }

  if (el.text) {
    s += renderText({ ...el, type: "text", x: el.x + 6, y: el.y,
      width: el.width - 12, height: el.height, strokeColor: el.strokeColor,
      fontSize: el.fontSize || 14, textAlign: el.textAlign || "center",
      verticalAlign: el.verticalAlign || "middle" });
  }
  return s;
}

function renderEllipse(el: ExcalidrawElement, sketch: boolean): string {
  const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
  const rx = el.width / 2, ry = el.height / 2;
  const sw = el.strokeWidth || 1;
  const fill = svgBg(el.backgroundColor);
  const col = svgColor(el.strokeColor, "#1e293b");
  const op = svgOpacity(el.opacity);

  let s: string;
  if (sketch) {
    const rng = mulberry32(hash32(el.id));
    const d = sketchEllipse(cx, cy, rx, ry, rng);
    s = `<path d="${d}" fill="${fill}" stroke="${col}" stroke-width="${sw}"`
      + ` stroke-linecap="round" stroke-linejoin="round" opacity="${op}"/>\n`;
  } else {
    s = `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"`
      + ` fill="${fill}" stroke="${col}" stroke-width="${sw}" opacity="${op}"/>\n`;
  }

  if (el.text) {
    s += renderText({ ...el, type: "text", x: cx - rx * 0.7, y: cy - ry * 0.5,
      width: rx * 1.4, height: ry, strokeColor: el.strokeColor,
      fontSize: el.fontSize || 14, textAlign: "center", verticalAlign: "middle" });
  }
  return s;
}

function renderArrow(el: ExcalidrawElement, sketch: boolean): string {
  const pts = el.points || [];
  if (pts.length < 2) return "";
  const sw = el.strokeWidth || 2;
  const col = svgColor(el.strokeColor, "#64748b");
  const absPts: [number, number][] = pts.map(p => [el.x + p[0], el.y + p[1]]);

  let d: string;
  if (sketch) {
    const rng = mulberry32(hash32(el.id));
    d = sketchPolyline(absPts, false, rng);
  } else {
    d = absPts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  }

  return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${sw}"`
       + ` stroke-linecap="round" stroke-linejoin="round" opacity="${svgOpacity(el.opacity)}"`
       + ` stroke-dasharray="${svgDash(el.strokeStyle)}"`
       + ` marker-end="url(#mk-${el.id})"/>\n`;
}

function renderLine(el: ExcalidrawElement, sketch: boolean): string {
  const pts = el.points || [];
  if (pts.length < 2) return "";
  const absPts: [number, number][] = pts.map(p => [el.x + p[0], el.y + p[1]]);

  let d: string;
  if (sketch) {
    const rng = mulberry32(hash32(el.id));
    d = sketchPolyline(absPts, false, rng);
  } else {
    d = absPts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  }

  return `<path d="${d}" fill="none" stroke="${svgColor(el.strokeColor, "#64748b")}"`
       + ` stroke-width="${el.strokeWidth || 1}" stroke-linecap="round" opacity="${svgOpacity(el.opacity)}"`
       + ` stroke-dasharray="${svgDash(el.strokeStyle)}"/>\n`;
}

// ── svg 辅助函数 ──

function esc(s: string): string { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function svgColor(c: string | undefined, fb: string): string { return (!c || c === "transparent") ? "none" : c === "#1e1e1e" ? fb : c; }
function svgBg(c: string | undefined): string { return (!c || c === "transparent") ? "none" : c; }
function svgOpacity(o: number | undefined): number { return o !== undefined ? o / 100 : 1; }
function svgDash(s: string | undefined): string { return s === "dashed" ? "6,4" : s === "dotted" ? "2,4" : "none"; }
function svgTextAnchor(a: string | undefined): string { return a === "center" ? "middle" : a === "right" ? "end" : "start"; }
