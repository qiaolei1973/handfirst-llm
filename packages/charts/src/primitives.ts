import type { Coord, Padding } from './types';
import { COLORS, STYLE } from './colors';

/**
 * Compute nice tick values for a data range.
 */
export function niceTicks(lo: number, hi: number, maxN: number): number[] {
  const raw = (hi - lo) / maxN;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  let step: number;
  if (norm <= 1) step = 1 * mag;
  else if (norm <= 2) step = 2 * mag;
  else if (norm <= 5) step = 5 * mag;
  else step = 10 * mag;
  const start = Math.floor(lo / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= hi + step * 0.5; v += step) {
    if (v >= lo - step * 0.1) ticks.push(v);
  }
  return ticks;
}

/**
 * Set canvas pixel dimensions accounting for devicePixelRatio.
 * Short-circuits if dimensions are unchanged (avoids unnecessary clears).
 * Returns logical CSS-pixel dimensions and the 2D context.
 */
export function resizeCanvas(canvas: HTMLCanvasElement): {
  w: number;
  h: number;
  ctx: CanvasRenderingContext2D;
} {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = r.width;
  const h = r.height;
  if (canvas.width === w * dpr && canvas.height === h * dpr) {
    const ctx = canvas.getContext('2d')!;
    return { w, h, ctx };
  }
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.getContext('2d')!.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx: canvas.getContext('2d')! };
}

/**
 * Draw grid lines for each tick value.
 */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  coord: Coord,
  xTicks: number[],
  yTicks: number[],
  pad: Padding,
  plotW: number,
  plotH: number,
): void {
  ctx.strokeStyle = COLORS.grayBg;
  ctx.lineWidth = 0.3;
  for (const tx of xTicks) {
    const gx = coord.toX(tx);
    ctx.beginPath();
    ctx.moveTo(gx, pad.t);
    ctx.lineTo(gx, pad.t + plotH);
    ctx.stroke();
  }
  for (const ty of yTicks) {
    const gy = coord.toY(ty);
    ctx.beginPath();
    ctx.moveTo(pad.l, gy);
    ctx.lineTo(pad.l + plotW, gy);
    ctx.stroke();
  }
}

/**
 * Draw four-sided box axes with tick labels and centered axis labels.
 */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  coord: Coord,
  xTicks: number[],
  yTicks: number[],
  xLabel: string,
  yLabel: string,
  pad: Padding,
  plotW: number,
  plotH: number,
  fontSize?: number,
): void {
  const fs = fontSize ?? STYLE.font.tick;
  const family = STYLE.font.family;

  ctx.lineWidth = STYLE.spine.lineWidth;
  ctx.lineJoin = 'round';

  // Four-sided box
  // bottom & left — normal weight
  ctx.strokeStyle = COLORS.grayLight;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH);  // bottom
  ctx.lineTo(pad.l + plotW, pad.t);           // right
  ctx.moveTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l, pad.t);                   // left
  ctx.lineTo(pad.l + plotW, pad.t);            // top
  ctx.stroke();

  // X-axis tick labels
  ctx.font = `${fs}px ${family}`;
  ctx.fillStyle = COLORS.gray;
  ctx.textAlign = 'center';
  for (const tx of xTicks) {
    const label = Number.isInteger(tx) ? String(tx) : tx.toFixed(1);
    ctx.fillText(label, coord.toX(tx), pad.t + plotH + 14);
  }
  ctx.font = `500 ${STYLE.font.axisLabel}px ${family}`;
  ctx.fillText(xLabel, pad.l + plotW / 2, pad.t + plotH + 32);

  // Y-axis tick labels (right-aligned)
  ctx.font = `${fs}px ${family}`;
  ctx.textAlign = 'right';
  for (const ty of yTicks) {
    const label = Number.isInteger(ty) ? String(ty) : ty.toFixed(1);
    ctx.fillText(label, pad.l - 6, coord.toY(ty) + 4);
  }

  // Y-axis label (rotated -90°)
  ctx.save();
  ctx.translate(pad.l - 42, pad.t + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.font = `500 ${STYLE.font.axisLabel}px ${family}`;
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

/**
 * Draw a legend box. Anchored at (x, y) — top-left corner.
 * Auto-computes box size from items.
 */
export function drawLegend(
  ctx: CanvasRenderingContext2D,
  items: { color: string; label: string }[],
  x: number,
  y: number,
): void {
  if (items.length === 0) return;

  const { padding, swatchSize, gap, fontSize } = STYLE.legend;
  const family = STYLE.font.family;
  ctx.font = `${fontSize}px ${family}`;

  // Measure
  let maxW = 0;
  for (const item of items) {
    const w = ctx.measureText(item.label).width;
    if (w > maxW) maxW = w;
  }
  const lineH = fontSize * 1.5;
  const boxW = padding * 2 + swatchSize + gap + maxW;
  const boxH = padding * 2 + items.length * lineH;

  // Background
  ctx.fillStyle = COLORS.tooltipBg;
  ctx.strokeStyle = COLORS.grayLight;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, boxW, boxH, 6);
  ctx.fill();
  ctx.stroke();

  // Items
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < items.length; i++) {
    const iy = y + padding + lineH * i + lineH / 2;
    const sx = x + padding;
    const sy = iy - swatchSize / 2;

    // Swatch
    ctx.fillStyle = items[i].color;
    ctx.beginPath();
    ctx.roundRect(sx, sy, swatchSize, swatchSize, 3);
    ctx.fill();

    // Label
    ctx.fillStyle = COLORS.tooltipText;
    ctx.fillText(items[i].label, sx + swatchSize + gap, iy);
  }
}
