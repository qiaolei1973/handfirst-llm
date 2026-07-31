import { resizeCanvas, niceTicks, drawGrid, drawAxes } from './primitives';
import type { Padding, Coord, Point, LineSeries } from './types';

export interface LineChartOpts {
  xLabel?: string;
  yLabel?: string;
  padding?: Partial<Padding>;
}

export interface LineChartHandle {
  setSeries(series: LineSeries[]): void;
  append(seriesIdx: number, pt: Point): void;
  clear(): void;
  draw(): void;
  onHoverX(xValue: number | null): void;
  hoverAtPixel(px: number): void;
}

export function createLineChart(
  canvas: HTMLCanvasElement,
  opts?: LineChartOpts,
): LineChartHandle {
  const pad: Padding = { t: 20, r: 20, b: 36, l: 52, ...opts?.padding };
  const xLabel = opts?.xLabel ?? 'x';
  const yLabel = opts?.yLabel ?? 'y';

  const series: LineSeries[] = [];
  let hoverX: number | null = null;
  let lastRange: {
    xLo: number;
    xHi: number;
    yLo: number;
    yHi: number;
    pad: Padding;
    plotW: number;
    plotH: number;
  } | null = null;

  // ---- crosshair drawing ----
  function drawCrosshair(
    ctx: CanvasRenderingContext2D,
    toX: (v: number) => number,
    toY: (v: number) => number,
    pw: number,
    ph: number,
  ): void {
    if (hoverX == null) return;

    const cx = toX(hoverX);

    // vertical dashed line
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = 'rgba(100,116,139,0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, pad.t);
    ctx.lineTo(cx, pad.t + ph);
    ctx.stroke();
    ctx.setLineDash([]);

    // dots on each series + build tooltip
    const tooltipLines: string[] = [`Epoch ${Math.round(hoverX)}`];
    for (const s of series) {
      if (s.points.length < 2) continue;
      // find nearest point
      let near: Point | null = null;
      let best = Infinity;
      for (const pt of s.points) {
        const d = Math.abs(pt.x - hoverX);
        if (d < best) {
          best = d;
          near = pt;
        }
      }
      if (!near) continue;
      const py = toY(near.y);
      ctx.beginPath();
      ctx.arc(cx, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      tooltipLines.push(`${s.label}: ${near.y.toFixed(4)}`);
    }

    // tooltip box
    const tfs = pw < 350 ? 9 : 10;
    ctx.font = `${tfs}px monospace`;
    const tw = Math.max(...tooltipLines.map((l) => ctx.measureText(l).width)) + 14;
    const lh = tfs * 1.45;
    const th = tooltipLines.length * lh + 12;
    let tx = cx + 12;
    let ty = pad.t + 6;
    if (tx + tw > pad.l + pw) tx = cx - tw - 12;
    if (ty + th > pad.t + ph) ty = pad.t + ph - th;

    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tx, ty, tw, th, 6);
    ctx.fill();
    ctx.stroke();

    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < tooltipLines.length; i++) {
      ctx.fillStyle = i === 0 ? '#64748b' : series[i - 1].color;
      ctx.fillText(tooltipLines[i], tx + 7, ty + 8 + i * lh);
    }
  }

  // ---- main draw ----
  function draw(): void {
    const { w, h, ctx } = resizeCanvas(canvas);
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    ctx.clearRect(0, 0, w, h);

    // data ranges
    let xLo = Infinity;
    let xHi = -Infinity;
    let yLo = Infinity;
    let yHi = -Infinity;
    for (const s of series) {
      for (const pt of s.points) {
        if (pt.x < xLo) xLo = pt.x;
        if (pt.x > xHi) xHi = pt.x;
        if (pt.y < yLo) yLo = pt.y;
        if (pt.y > yHi) yHi = pt.y;
      }
    }
    if (!isFinite(xLo)) {
      xLo = 0;
      xHi = 1;
      yLo = 0;
      yHi = 1;
    }
    if (xHi - xLo < 20) xHi = xLo + 20;
    const yPad = (yHi - yLo) * 0.15 || 0.5;
    yLo -= yPad;
    yHi += yPad;

    // store range for hover conversion
    lastRange = { xLo, xHi, yLo, yHi, pad, plotW, plotH };

    const toX = (v: number) => pad.l + ((v - xLo) / (xHi - xLo)) * plotW;
    const toY = (v: number) => pad.t + ((yHi - v) / (yHi - yLo)) * plotH;
    const coord: Coord = { toX, toY };

    const xTicks = niceTicks(xLo, xHi, 6);
    const yTicks = niceTicks(yLo, yHi, 5);

    drawGrid(ctx, coord, xTicks, yTicks, pad, plotW, plotH);

    // data lines
    for (const s of series) {
      if (s.points.length < 2) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      const pts = s.points;
      for (let i = 0; i < pts.length; i++) {
        const px = toX(pts[i].x);
        const py = toY(pts[i].y);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // crosshair
    if (hoverX != null && series.length > 0 && series[0].points.length > 0) {
      drawCrosshair(ctx, toX, toY, plotW, plotH);
    }

    drawAxes(ctx, coord, xTicks, yTicks, xLabel, yLabel, pad, plotW, plotH, plotW < 350 ? 8 : 9);
  }

  // ---- public API ----
  return {
    setSeries(s: LineSeries[]): void {
      series.length = 0;
      for (const item of s) series.push(item);
    },
    append(idx: number, pt: Point): void {
      while (series.length <= idx) {
        series.push({ label: '', color: '#94a3b8', points: [] });
      }
      series[idx].points.push(pt);
    },
    clear(): void {
      series.length = 0;
      hoverX = null;
      lastRange = null;
    },
    draw,
    onHoverX(xValue: number | null): void {
      hoverX = xValue;
      draw();
    },
    hoverAtPixel(px: number): void {
      if (!lastRange) return;
      const { xLo, xHi, pad: rPad, plotW } = lastRange;
      const raw = xLo + ((px - rPad.l) / plotW) * (xHi - xLo);
      const clamped = Math.round(raw);
      const xValue = Math.max(Math.ceil(xLo), Math.min(Math.floor(xHi), clamped));
      this.onHoverX(xValue);
    },
  };
}
