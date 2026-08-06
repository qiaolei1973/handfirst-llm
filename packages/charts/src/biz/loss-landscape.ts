/**
 * Loss landscape chart — parabola with gradient tangent, valley marker,
 * and optimization trajectory. Built on Axes + CanvasManager.
 */
import type { Padding, Point } from '../types';
import { COLORS, STYLE } from '../colors';
import { createAxes } from '../axes';
import type { AxesHandle } from '../axes';
import { createCanvasManager } from '../canvas';
import type { CanvasManager } from '../canvas';

export interface LossLandscapeOpts {
  xLabel?: string;
  yLabel?: string;
}

export interface LossLandscapeConfig {
  a: number;
  b: number;
  c: number;
  currentX: number;
  gradient?: number;
  trajectory?: Point[];
  valleyX?: number;
}

export interface LossLandscapeHandle {
  update(config: LossLandscapeConfig): void;
  draw(): void;
  clearTrajectory(): void;
  destroy(): void;
}

export function createLossLandscape(
  canvas: HTMLCanvasElement,
  opts?: LossLandscapeOpts,
): LossLandscapeHandle {
  const pad: Padding = { t: 20, r: 25, b: 36, l: 58 };
  const xLabel = opts?.xLabel ?? 'param';
  const yLabel = opts?.yLabel ?? 'Loss';

  const cm: CanvasManager = createCanvasManager(canvas);
  const ax: AxesHandle = createAxes(cm.ctx, cm.w, cm.h, pad, { xLabel, yLabel });

  let config: LossLandscapeConfig | null = null;
  let trajectory: Point[] = [];

  cm.onResize = () => drawInternal();

  function f(x: number): number {
    if (!config) return 0;
    return config.a * x * x + config.b * x + config.c;
  }

  function derivative(x: number): number {
    if (!config) return 0;
    return 2 * config.a * x + config.b;
  }

  function syncLayers(): void {
    if (!config) return;
    const { a, b, c, currentX, gradient, valleyX } = config;
    const grad = gradient ?? derivative(currentX);
    const valley = valleyX ?? -b / (2 * a);

    ax.clearLayers();

    // x range from data
    const allX = [currentX, valley, ...trajectory.map((t) => t.x)];
    let xLo = Math.min(...allX) - 1.5;
    let xHi = Math.max(...allX) + 1.5;
    if (xHi - xLo < 4) { const mid = (xHi + xLo) / 2; xLo = mid - 2; xHi = mid + 2; }

    // Sample parabola
    const N = 200;
    const curveXs: number[] = [];
    const curveYs: number[] = [];
    for (let i = 0; i <= N; i++) {
      const xv = xLo + (i / N) * (xHi - xLo);
      curveXs.push(xv);
      curveYs.push(f(xv));
    }
    ax.addLayer({ type: 'line', xs: curveXs, ys: curveYs, color: COLORS.grayLight, width: 1.5 });

    // Parabola range
    const allY = [...curveYs, f(valley)];
    for (const t of trajectory) allY.push(f(t.x));
    const yLo = Math.min(...allY) - (Math.max(...allY) - Math.min(...allY)) * 0.1;
    const yHi = Math.max(...allY) + (Math.max(...allY) - Math.min(...allY)) * 0.1;
    ax.setRange({ xMin: xLo, xMax: xHi, yMin: yLo, yMax: yHi });

    // Trajectory line
    if (trajectory.length > 1) {
      ax.addLayer({
        type: 'line',
        xs: trajectory.map((t) => t.x),
        ys: trajectory.map((t) => f(t.x)),
        color: COLORS.trajectory,
        width: 1,
        dash: [2, 4],
      });
      // Sparse dots
      const step = Math.max(1, Math.floor(trajectory.length / 40));
      const dots = trajectory.filter((_, i) => i % step === 0).map((t) => ({ x: t.x, y: f(t.x) }));
      ax.addLayer({ type: 'scatter', points: dots, color: COLORS.trajectoryDot, size: 1.5 });
    }

    // Current point (no layer — drawn manually below for ring+stroke effect)

    // Tangent line (2-point segment)
    const tExtent = (xHi - xLo) * 0.15;
    ax.addLayer({
      type: 'line',
      xs: [currentX - tExtent, currentX + tExtent],
      ys: [f(currentX) + grad * (-tExtent), f(currentX) + grad * tExtent],
      color: COLORS.red,
      width: 2,
      dash: [4, 3],
    });

    // Valley marker
    if (valley > xLo && valley < xHi) {
      ax.addLayer({ type: 'vline', x: valley, color: COLORS.valley, dash: [2, 4] });
      ax.addLayer({
        type: 'annotation',
        text: `谷底 ${xLabel}=${valley.toFixed(2)}`,
        x: valley,
        y: yLo,
        color: COLORS.valleyLabel,
        ox: 0,
        oy: -10,
      });
    }
  }

  function drawInternal(): void {
    syncLayers();
    ax.draw();

    if (!config) return;
    const ctx = cm.ctx;
    const currentX = config.currentX;
    const curY = f(currentX);
    const grad = config.gradient ?? derivative(currentX);
    const cx = ax.toX(currentX);
    const cy = ax.toY(curY);
    const fontSize = ax.getRange().plotW < 400 ? 8 : 9;

    // Current point ring (stick-out visual — red fill + white stroke)
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.red;
    ctx.fill();
    ctx.strokeStyle = COLORS.white;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Gradient annotation
    ctx.fillStyle = COLORS.red;
    ctx.font = `600 ${fontSize}px ${STYLE.font.family}`;
    ctx.textAlign = 'left';
    ctx.fillText(`grad=${grad.toFixed(2)}`, cx + 10, cy - 10);
    ctx.fillText(`${xLabel}=${currentX.toFixed(2)}`, cx + 10, cy + 24);
  }

  return {
    update(cfg: LossLandscapeConfig): void {
      config = cfg;
      if (cfg.trajectory) trajectory = cfg.trajectory;
    },
    draw(): void {
      drawInternal();
    },
    clearTrajectory(): void {
      trajectory = [];
    },
    destroy(): void {
      cm.destroy();
    },
  };
}
