/**
 * Model curve chart — scatter points + true function curve + model prediction curve.
 *
 * Unlike model-fit.ts (which assumes y=Wx+b), this chart works with arbitrary
 * nonlinear prediction functions — ideal for neural network outputs.
 * Built on Axes + CanvasManager.
 */
import type { Padding } from '../types';
import { COLORS } from '../colors';
import { createAxes } from '../axes';
import type { AxesHandle } from '../axes';
import { createCanvasManager } from '../canvas';
import type { CanvasManager } from '../canvas';

export interface ModelCurveConfig {
  points: { x: number; y: number }[];
  trueFn: (x: number) => number;
  trueLabel: string;
  predict: (x: number) => number;
  modelLabel?: string;
}

export interface ModelCurveHandle {
  update(config: ModelCurveConfig): void;
  draw(): void;
  destroy(): void;
}

const PAD: Padding = { t: 30, r: 30, b: 42, l: 58 };

export function createModelCurve(
  canvas: HTMLCanvasElement,
): ModelCurveHandle {
  const cm: CanvasManager = createCanvasManager(canvas);
  const ax: AxesHandle = createAxes(cm.ctx, cm.w, cm.h, PAD, {
    xLabel: 'x',
    yLabel: 'y',
  });

  let cfg: ModelCurveConfig | null = null;

  cm.onResize = () => drawInternal();

  function syncLayers(): void {
    if (!cfg) return;
    const { points, trueFn, trueLabel, predict, modelLabel } = cfg;

    const N_SAMPLE = 200;
    const rawX = points.map((p) => p.x);
    const xMin = Math.min(...rawX);
    const xMax = Math.max(...rawX);
    const xStep = (xMax - xMin) / (N_SAMPLE - 1);

    // Sample x values for smooth curves
    const curveXs: number[] = [];
    for (let i = 0; i < N_SAMPLE; i++) curveXs.push(xMin + i * xStep);

    // y range from all data
    const yAll = rawX.flatMap((x, i) => [
      points[i].y,
      predict(x),
      trueFn(x),
    ]);
    let yMin = Math.min(...yAll) - 0.4;
    let yMax = Math.max(...yAll) + 0.4;
    if (yMax - yMin < 1) {
      const mid = (yMax + yMin) / 2;
      yMin = mid - 0.8;
      yMax = mid + 0.8;
    }
    ax.setRange({ xMin, xMax, yMin, yMax });

    ax.clearLayers();

    // True function curve (green dashed)
    ax.addLayer({
      type: 'line',
      xs: curveXs,
      ys: curveXs.map(trueFn),
      color: COLORS.trueFn,
      width: 2,
      dash: [8, 5],
    });

    // Model prediction curve (purple solid)
    ax.addLayer({
      type: 'line',
      xs: curveXs,
      ys: curveXs.map(predict),
      color: COLORS.purple,
      width: 2.5,
    });

    // Data points (gray scatter)
    ax.addLayer({
      type: 'scatter',
      points,
      color: COLORS.gray,
      size: 3.5,
    });

    // Legend
    ax.setLegend([
      { color: COLORS.green, label: trueLabel },
      {
        color: COLORS.purple,
        label: modelLabel ?? '模型预测',
      },
    ]);
  }

  function drawInternal(): void {
    syncLayers();
    ax.draw();
  }

  return {
    update(c: ModelCurveConfig): void {
      cfg = c;
    },
    draw(): void {
      drawInternal();
    },
    destroy(): void {
      cm.destroy();
    },
  };
}
