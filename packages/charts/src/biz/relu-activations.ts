/**
 * ReLU Activations chart — per-neuron output across the input range.
 *
 * Shows how each hidden neuron "folds" at its own threshold,
 * making the piecewise-linear decomposition visible.
 * Built on Axes + CanvasManager.
 */
import type { Padding } from '../types';
import { COLORS } from '../colors';
import { createAxes } from '../axes';
import type { AxesHandle } from '../axes';
import { createCanvasManager } from '../canvas';
import type { CanvasManager } from '../canvas';

export interface ReLUActivationsConfig {
  xRange: { min: number; max: number };
  neurons: {
    w: number;
    b: number;
    label: string;
    color: string;
  }[];
}

export interface ReLUActivationsHandle {
  update(config: ReLUActivationsConfig): void;
  draw(): void;
  destroy(): void;
}

const PAD: Padding = { t: 24, r: 30, b: 42, l: 58 };

// Color palette for neurons
const NEURON_COLORS = [
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#10b981', // emerald
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
];

export function createReLUActivations(
  canvas: HTMLCanvasElement,
): ReLUActivationsHandle {
  const cm: CanvasManager = createCanvasManager(canvas);
  const ax: AxesHandle = createAxes(cm.ctx, cm.w, cm.h, PAD, {
    xLabel: 'x',
    yLabel: 'ReLU 输出',
  });

  let cfg: ReLUActivationsConfig | null = null;

  cm.onResize = () => drawInternal();

  function relu(v: number): number {
    return v > 0 ? v : 0;
  }

  function syncLayers(): void {
    if (!cfg) return;
    const { xRange, neurons } = cfg;
    const N = 200;

    const xs: number[] = [];
    const step = (xRange.max - xRange.min) / (N - 1);
    for (let i = 0; i < N; i++) xs.push(xRange.min + i * step);

    // y range: ReLU output is always >= 0
    let yMax = 0;
    for (const n of neurons) {
      for (const x of xs) {
        const v = relu(n.w * x + n.b);
        if (v > yMax) yMax = v;
      }
    }
    if (yMax < 0.5) yMax = 1;

    ax.setRange({
      xMin: xRange.min,
      xMax: xRange.max,
      yMin: -0.05 * yMax, // tiny negative padding for visual breathing room
      yMax: yMax * 1.1,
    });

    ax.clearLayers();

    // y=0 reference line
    ax.addLayer({
      type: 'hline',
      y: 0,
      color: COLORS.grayLight,
      dash: [2, 3],
    });

    // One line per neuron
    for (let i = 0; i < neurons.length; i++) {
      const n = neurons[i];
      const ys = xs.map((x) => relu(n.w * x + n.b));
      ax.addLayer({
        type: 'line',
        xs,
        ys,
        color: n.color,
        width: 1.8,
      });
    }

    // Legend
    ax.setLegend(
      neurons.map((n) => ({ color: n.color, label: n.label })),
    );
  }

  function drawInternal(): void {
    syncLayers();
    ax.draw();
  }

  return {
    update(c: ReLUActivationsConfig): void {
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
