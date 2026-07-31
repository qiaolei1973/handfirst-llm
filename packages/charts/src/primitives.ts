import type { Coord, Padding } from './types';

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
 * Draw grid lines for each tick value (light gray).
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
  ctx.strokeStyle = '#f1f5f9';
  ctx.lineWidth = 0.5;
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
 * Draw L-shaped axes with tick labels and centered axis labels.
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
  const fs = fontSize ?? 9;
  ctx.font = `${fs}px sans-serif`;
  ctx.fillStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#cbd5e1';

  // L-shaped axis lines
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + plotH);
  ctx.moveTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH);
  ctx.stroke();

  // X-axis tick labels
  ctx.textAlign = 'center';
  for (const tx of xTicks) {
    const label = Number.isInteger(tx) ? String(tx) : tx.toFixed(1);
    ctx.fillText(label, coord.toX(tx), pad.t + plotH + 14);
  }
  ctx.fillText(xLabel, pad.l + plotW / 2, pad.t + plotH + 32);

  // Y-axis tick labels (right-aligned)
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
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}
