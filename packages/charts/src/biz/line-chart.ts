/**
 * Line chart — multi-series line plot with crosshair tooltip.
 * Built on Axes (coordinate management) + CanvasManager (DOM).
 */
import type { Padding, Point, LineSeries } from '../types';
import { COLORS } from '../colors';
import { createAxes } from '../axes';
import type { AxesHandle } from '../axes';
import { createCanvasManager } from '../canvas';
import type { CanvasManager } from '../canvas';

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
  destroy(): void;
}

const DEFAULT_PAD: Padding = { t: 20, r: 20, b: 36, l: 52 };

export function createLineChart(
  canvas: HTMLCanvasElement,
  opts?: LineChartOpts,
): LineChartHandle {
  const pad: Padding = { ...DEFAULT_PAD, ...opts?.padding };
  const xLabel = opts?.xLabel ?? 'x';
  const yLabel = opts?.yLabel ?? 'y';

  const cm: CanvasManager = createCanvasManager(canvas);
  const ax: AxesHandle = createAxes(cm.ctx, cm.w, cm.h, pad, { xLabel, yLabel });

  let seriesData: LineSeries[] = [];
  let hoverX: number | null = null;

  cm.onResize = () => {
    drawInternal();
  };

  function syncLayers(): void {
    ax.clearLayers();
    for (const s of seriesData) {
      if (s.points.length < 2) continue;
      ax.addLayer({
        type: 'line',
        xs: s.points.map((p) => p.x),
        ys: s.points.map((p) => p.y),
        color: s.color,
        width: 2,
      });
    }
    ax.setLegend(
      seriesData
        .filter((s) => s.points.length > 0 && s.label)
        .map((s) => ({ color: s.color, label: s.label })),
    );
  }

  // ---- crosshair ----
  function drawCrosshair(): void {
    if (hoverX == null) return;
    const range = ax.getRange();
    const ctx = cm.ctx;
    const plotW = range.plotW;
    const plotH = range.plotH;
    const cx = ax.toX(hoverX);

    // vertical dashed line
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = COLORS.crosshair;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, pad.t);
    ctx.lineTo(cx, pad.t + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // dots on each series + tooltip lines
    const tooltipLines: string[] = [`Epoch ${Math.round(hoverX)}`];
    for (const s of seriesData) {
      if (s.points.length < 2) continue;
      let near: Point | null = null;
      let best = Infinity;
      for (const pt of s.points) {
        const d = Math.abs(pt.x - hoverX);
        if (d < best) { best = d; near = pt; }
      }
      if (!near) continue;
      const py = ax.toY(near.y);
      ctx.beginPath();
      ctx.arc(cx, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.strokeStyle = COLORS.white;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      tooltipLines.push(`${s.label}: ${near.y.toFixed(4)}`);
    }

    // tooltip box
    const tfs = plotW < 350 ? 9 : 10;
    ctx.font = `${tfs}px monospace`;
    const tw = Math.max(...tooltipLines.map((l) => ctx.measureText(l).width)) + 14;
    const lh = tfs * 1.45;
    const th = tooltipLines.length * lh + 12;
    let tx = cx + 12;
    let ty = pad.t + 6;
    if (tx + tw > pad.l + plotW) tx = cx - tw - 12;
    if (ty + th > pad.t + plotH) ty = pad.t + plotH - th;

    ctx.fillStyle = COLORS.tooltipBg;
    ctx.strokeStyle = COLORS.grayLight;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tx, ty, tw, th, 6);
    ctx.fill();
    ctx.stroke();

    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < tooltipLines.length; i++) {
      ctx.fillStyle = i === 0 ? COLORS.tooltipText : seriesData[i - 1]?.color ?? COLORS.tooltipText;
      ctx.fillText(tooltipLines[i], tx + 7, ty + 8 + i * lh);
    }
  }

  function drawInternal(): void {
    syncLayers();
    ax.draw();
    drawCrosshair();
  }

  // ---- public API ----

  return {
    setSeries(series: LineSeries[]): void {
      seriesData.length = 0;
      for (const item of series) seriesData.push(item);
    },
    append(idx: number, pt: Point): void {
      while (seriesData.length <= idx) {
        seriesData.push({ label: '', color: COLORS.gray, points: [] });
      }
      seriesData[idx].points.push(pt);
    },
    clear(): void {
      seriesData.length = 0;
      hoverX = null;
      ax.clearLayers();
    },
    draw(): void {
      drawInternal();
    },
    onHoverX(xValue: number | null): void {
      hoverX = xValue;
      drawInternal();
    },
    hoverAtPixel(px: number): void {
      const range = ax.getRange();
      const raw = range.xMin + ((px - pad.l) / range.plotW) * (range.xMax - range.xMin);
      const clamped = Math.round(raw);
      hoverX = Math.max(Math.ceil(range.xMin), Math.min(Math.floor(range.xMax), clamped));
      drawInternal();
    },
    destroy(): void {
      cm.destroy();
    },
  };
}
