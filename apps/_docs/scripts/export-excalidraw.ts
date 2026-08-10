#!/usr/bin/env tsx
/**
 * 将 .excalidraw 文件导出为 SVG。
 * 纯 Node.js，不依赖浏览器、Playwright、或任何第三方包。
 *
 * Usage:
 *   npx tsx scripts/export-excalidraw.ts <scene.excalidraw> [-o output.svg]
 */

import { readFileSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";

// ── 参数 ──
const args = process.argv.slice(2);
if (args.length === 0 || args.includes("--help")) {
  console.log("Usage: npx tsx scripts/export-excalidraw.ts <scene.excalidraw> [-o output.svg]");
  process.exit(0);
}

const inputPath = resolve(args[0]);
let outputPath = "";
for (let i = 1; i < args.length; i++) {
  if (args[i] === "-o" && i + 1 < args.length) outputPath = resolve(args[++i]);
}
if (!outputPath) outputPath = inputPath.replace(/\.excalidraw$/, ".svg");

// ── 读取 scene ──
const scene = JSON.parse(readFileSync(inputPath, "utf-8"));

// ── 类型 ──
type Element = {
  id: string; type: string; x: number; y: number; width: number; height: number;
  angle?: number; strokeColor?: string; backgroundColor?: string;
  fillStyle?: string; strokeWidth?: number; strokeStyle?: string;
  roughness?: number; opacity?: number; roundness?: { type: number } | null;
  text?: string; fontSize?: number; fontFamily?: number;
  textAlign?: string; verticalAlign?: string;
  points?: [number, number][];
  boundElements?: { id: string; type: string }[] | null;
};

// ── SVG 构建 ──
function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function color(c: string | undefined, fallback: string): string {
  if (!c || c === "transparent") return "none";
  if (c === "#1e1e1e") return fallback;
  return c;
}

function bg(c: string | undefined): string {
  if (!c || c === "transparent") return "none";
  return c;
}

function opacity(o: number | undefined): number {
  return o !== undefined ? o / 100 : 1;
}

function dash(s: string | undefined): string {
  if (s === "dashed") return "6,4";
  if (s === "dotted") return "2,4";
  return "none";
}

function textAnchor(a: string | undefined): string {
  if (a === "center") return "middle";
  if (a === "right") return "end";
  return "start";
}

let defs = "";

function renderText(el: Element): string {
  const fs = el.fontSize || 16;
  const ff = el.fontFamily === 2 ? "Helvetica, Arial, sans-serif"
           : el.fontFamily === 1 ? "Virgil, Segoe UI Emoji"
           : "Virgil, Segoe UI Emoji";
  const lines = (el.text || "").split("\n");
  const lineH = fs * 1.3;
  const totalH = lines.length * lineH;
  // rough centering
  const sx = el.x;
  const sy = el.y + (el.verticalAlign === "middle" ? (el.height - totalH) / 2 + fs * 0.8 : fs);

  let svg = "";
  for (let i = 0; i < lines.length; i++) {
    const anchorX = el.textAlign === "center" ? el.x + el.width / 2
                  : el.textAlign === "right"  ? el.x + el.width
                  : sx;
    svg += `<text x="${anchorX}" y="${sy + i * lineH}"`
         +  ` font-family="${ff}" font-size="${fs}"`
         +  ` fill="${color(el.strokeColor, "#1e293b")}"`
         +  ` text-anchor="${textAnchor(el.textAlign)}"`
         +  ` opacity="${opacity(el.opacity)}">${esc(lines[i])}</text>\n`;
  }
  return svg;
}

function renderRect(el: Element): string {
  if (el.text && !el.backgroundColor) {
    return renderText(el);
  }

  const rx = el.roundness?.type === 3 ? 8
           : el.roundness?.type === 2 ? 6
           : el.roundness?.type === 1 ? 4 : 0;

  let s = `<rect x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}"`
        + ` rx="${rx}"`
        + ` fill="${bg(el.backgroundColor)}"`
        + ` stroke="${color(el.strokeColor, "#1e293b")}"`
        + ` stroke-width="${el.strokeWidth || 1}"`
        + ` stroke-dasharray="${dash(el.strokeStyle)}"`
        + ` opacity="${opacity(el.opacity)}"`;
  if (el.angle) s += ` transform="rotate(${el.angle}, ${el.x + el.width/2}, ${el.y + el.height/2})"`;
  s += "/>\n";

  if (el.text) {
    const textEl: Element = {
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
    };
    s += renderText(textEl);
  }
  return s;
}

function renderEllipse(el: Element): string {
  const cx = el.x + el.width / 2;
  const cy = el.y + el.height / 2;
  const rx = el.width / 2;
  const ry = el.height / 2;
  let s = `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"`
        + ` fill="${bg(el.backgroundColor)}"`
        + ` stroke="${color(el.strokeColor, "#1e293b")}"`
        + ` stroke-width="${el.strokeWidth || 1}"`
        + ` opacity="${opacity(el.opacity)}"/>\n`;
  if (el.text) {
    const textEl: Element = {
      ...el, type: "text",
      x: cx - rx * 0.7, y: cy - ry * 0.5,
      width: rx * 1.4, height: ry,
      strokeColor: el.strokeColor,
      fontSize: el.fontSize || 14,
      textAlign: "center", verticalAlign: "middle",
    };
    s += renderText(textEl);
  }
  return s;
}

function arrowId(el: Element): string {
  return `arw-${el.id}`;
}

function collectArrowDefs() {
  for (const el of elements) {
    if (el.type !== "arrow") continue;
    const points = el.points || [];
    if (points.length < 2) continue;
    const col = color(el.strokeColor, "#64748b");
    const markerId = `mk-${el.id}`;
    // Excalidraw-style open V arrowhead (not filled triangle)
    defs += `<marker id="${markerId}" markerWidth="18" markerHeight="14" refX="14" refY="7"`
         + ` orient="auto" markerUnits="userSpaceOnUse">`
         + `<path d="M 1,2 L 14,7 L 1,12" fill="none" stroke="${col}"`
         + ` stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"`
         + ` opacity="${opacity(el.opacity)}"/>`
         + `</marker>\n`;
  }
}

function renderArrow(el: Element): string {
  const points = el.points || [];
  if (points.length < 2) return "";

  const sw = el.strokeWidth || 2;
  const col = color(el.strokeColor, "#64748b");
  const markerId = `mk-${el.id}`;

  const pathData = points.map((p, i) =>
    `${i === 0 ? "M" : "L"} ${el.x + p[0]} ${el.y + p[1]}`
  ).join(" ");

  return `<path d="${pathData}" fill="none"`
       + ` stroke="${col}"`
       + ` stroke-width="${sw}"`
       + ` stroke-dasharray="${dash(el.strokeStyle)}"`
       + ` opacity="${opacity(el.opacity)}"`
       + ` marker-end="url(#${markerId})"/>\n`;
}

function renderLine(el: Element): string {
  const points = el.points || [];
  if (points.length < 2) return "";

  const pathData = points.map((p, i) =>
    `${i === 0 ? "M" : "L"} ${el.x + p[0]} ${el.y + p[1]}`
  ).join(" ");

  return `<path d="${pathData}" fill="none"`
       + ` stroke="${color(el.strokeColor, "#64748b")}"`
       + ` stroke-width="${el.strokeWidth || 1}"`
       + ` stroke-dasharray="${dash(el.strokeStyle)}"`
       + ` opacity="${opacity(el.opacity)}"/>\n`;
}

function renderElement(el: Element): string {
  switch (el.type) {
    case "rectangle": return renderRect(el);
    case "ellipse":   return renderEllipse(el);
    case "arrow":     return renderArrow(el);
    case "line":      return renderLine(el);
    case "text":      return renderText(el);
    default:          return `<!-- unknown type: ${el.type} -->\n`;
  }
}

// ── 计算画布 bounds ──
const elements: Element[] = scene.elements || [];
if (elements.length === 0) {
  console.error("No elements found in scene");
  process.exit(1);
}

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

// ── 预收集箭头 marker defs ──
collectArrowDefs();

// ── 组装 SVG ──
let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
svg += `<svg xmlns="http://www.w3.org/2000/svg"`
     + ` viewBox="${minX - pad} ${minY - pad} ${vbW} ${vbH}"`
     + ` width="${vbW}" height="${vbH}">\n`;
svg += `<defs>\n${defs}</defs>\n`;
svg += `<rect x="${minX - pad}" y="${minY - pad}" width="${vbW}" height="${vbH}" fill="white"/>\n`;

for (const el of elements) {
  svg += renderElement(el);
}

svg += `</svg>\n`;

writeFileSync(outputPath, svg, "utf-8");
console.log(`  ✓ ${basename(outputPath)}  (${(svg.length / 1024).toFixed(1)} KB)`);
