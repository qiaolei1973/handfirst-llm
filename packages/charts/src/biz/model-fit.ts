/**
 * Model fit chart — scatter points + true line + model line + error bars.
 * Built on Axes + CanvasManager.
 */
import type { Padding } from '../types';
import { COLORS } from '../colors';
import { createAxes } from '../axes';
import type { AxesHandle } from '../axes';
import { createCanvasManager } from '../canvas';
import type { CanvasManager } from '../canvas';

export interface ModelFitConfig {
  points: { x: number; y: number }[];
  trueFn: (x: number) => number;
  trueLabel: string;
  W: number;
  bias: number;
}

export interface ModelFitHandle {
  update(config: ModelFitConfig): void;
  draw(): void;
  destroy(): void;
}

const PAD: Padding = { t: 30, r: 30, b: 42, l: 58 };

export function createModelFit(canvas: HTMLCanvasElement): ModelFitHandle {
  const cm: CanvasManager = createCanvasManager(canvas);
  const ax: AxesHandle = createAxes(cm.ctx, cm.w, cm.h, PAD, { xLabel: 'x', yLabel: 'y' });

  let cfg: ModelFitConfig | null = null;

  cm.onResize = () => drawInternal();

  function syncLayers(): void {
    if (!cfg) return;
    const { points, trueFn, trueLabel, W, bias } = cfg;
    const N = points.length;
    const rawX = points.map((p) => p.x);
    const labels = points.map((p) => p.y);
    const xMin = 0;
    const xMax = Math.max(...rawX);

    // y range
    const yAll = rawX.flatMap((x, i) => [labels[i], x * W + bias, trueFn(x)]);
    let yMin = Math.min(...yAll) - 2;
    let yMax = Math.max(...yAll) + 2;
    if (yMax - yMin < 5) { const mid = (yMax + yMin) / 2; yMin = mid - 4; yMax = mid + 4; }
    ax.setRange({ xMin, xMax, yMin, yMax });

    ax.clearLayers();

    // True function line (green dashed)
    const trueXs = [xMin, xMax];
    const trueYs = [trueFn(xMin), trueFn(xMax)];
    ax.addLayer({ type: 'line', xs: trueXs, ys: trueYs, color: COLORS.trueFn, width: 2, dash: [8, 5] });

    // Model line (purple solid)
    ax.addLayer({
      type: 'line',
      xs: trueXs,
      ys: [W * xMin + bias, W * xMax + bias],
      color: COLORS.purple,
      width: 2.5,
    });

    // Error segments (from data point to prediction)
    for (let i = 0; i < N; i++) {
      const xi = rawX[i];
      ax.addLayer({
        type: 'segment',
        x1: xi, y1: labels[i],
        x2: xi, y2: xi * W + bias,
        color: COLORS.errorLine,
        dash: [3, 3],
      });
    }

    // Data points (gray)
    ax.addLayer({ type: 'scatter', points, color: COLORS.gray, size: 4 });

    // Legend
    ax.setLegend([
      { color: COLORS.green, label: trueLabel },
      { color: COLORS.purple, label: `模型  y=${W.toFixed(2)}x+${bias.toFixed(2)}` },
      { color: COLORS.red, label: '误差' },
    ]);
  }

  function drawInternal(): void {
    syncLayers();
    ax.draw();
  }

  return {
    update(c: ModelFitConfig): void { cfg = c; },
    draw(): void { drawInternal(); },
    destroy(): void { cm.destroy(); },
  };
}
