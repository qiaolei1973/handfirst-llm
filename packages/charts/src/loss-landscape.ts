import { resizeCanvas, niceTicks, drawGrid, drawAxes } from './primitives';
import type { Padding, Coord, Point } from './types';

export interface LossLandscapeOpts {
  xLabel?: string;
  yLabel?: string;
}

export interface LossLandscapeConfig {
  /** parabola coefficients: y = a·x² + b·x + c */
  a: number;
  b: number;
  c: number;
  /** current parameter value (x position) */
  currentX: number;
  /** gradient value (auto-computed as 2a·x + b if omitted) */
  gradient?: number;
  /** optimization trajectory dots */
  trajectory?: Point[];
  /** valley bottom x (auto-computed as -b/(2a) if omitted) */
  valleyX?: number;
}

export interface LossLandscapeHandle {
  update(config: LossLandscapeConfig): void;
  draw(): void;
  clearTrajectory(): void;
}

export function createLossLandscape(
  canvas: HTMLCanvasElement,
  opts?: LossLandscapeOpts,
): LossLandscapeHandle {
  const pad: Padding = { t: 20, r: 25, b: 36, l: 58 };
  const xLabel = opts?.xLabel ?? 'param';
  const yLabel = opts?.yLabel ?? 'Loss';

  let config: LossLandscapeConfig | null = null;
  let trajectory: Point[] = [];

  // parabola evaluation
  function f(x: number): number {
    if (!config) return 0;
    return config.a * x * x + config.b * x + config.c;
  }

  function derivative(x: number): number {
    if (!config) return 0;
    return 2 * config.a * x + config.b;
  }

  function draw(): void {
    const { w, h, ctx } = resizeCanvas(canvas);
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    ctx.clearRect(0, 0, w, h);

    if (!config) {
      // no data yet — draw empty axes with default range
      const { toX, toY } = makeCoord(0, 1, 0, 1, plotW, plotH);
      drawGrid(ctx, { toX, toY }, [], [], pad, plotW, plotH);
      drawAxes(ctx, { toX, toY }, [], [], xLabel, yLabel, pad, plotW, plotH);
      return;
    }

    const { a, b, c, currentX, gradient } = config;

    // auto-compute
    const grad = gradient ?? derivative(currentX);
    const valley = config.valleyX ?? -b / (2 * a / (config.c ? 1 : 1));

    // ---------- data range ----------
    // use trajectory + currentX + valley to determine x range
    const allX = [currentX, valley];
    for (const t of trajectory) allX.push(t.x);
    let xLo = Math.min(...allX) - 1.5;
    let xHi = Math.max(...allX) + 1.5;
    if (xHi - xLo < 4) {
      const mid = (xHi + xLo) / 2;
      xLo = mid - 2;
      xHi = mid + 2;
    }

    // sample parabola for y range
    const N = 200;
    let yLo = Infinity;
    let yHi = -Infinity;
    const curve: Point[] = [];
    for (let i = 0; i <= N; i++) {
      const xv = xLo + (i / N) * (xHi - xLo);
      const yv = f(xv);
      curve.push({ x: xv, y: yv });
      if (yv < yLo) yLo = yv;
      if (yv > yHi) yHi = yv;
    }
    // also include trajectory y values
    for (const t of trajectory) {
      const yv = f(t.x);
      if (yv < yLo) yLo = yv;
      if (yv > yHi) yHi = yv;
    }
    // include current point
    const curY = f(currentX);
    if (curY < yLo) yLo = curY;
    if (curY > yHi) yHi = curY;

    yLo -= (yHi - yLo) * 0.1;
    yHi += (yHi - yLo) * 0.1;

    const { toX, toY } = makeCoord(xLo, xHi, yLo, yHi, plotW, plotH);

    const xTicks = niceTicks(xLo, xHi, 6);
    const yTicks = niceTicks(yLo, yHi, 5);
    const fontSize = plotW < 400 ? 8 : 9;

    drawGrid(ctx, { toX, toY }, xTicks, yTicks, pad, plotW, plotH);

    // ---- parabola curve ----
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i < curve.length; i++) {
      const px = toX(curve[i].x);
      const py = toY(curve[i].y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // ---- trajectory ----
    if (trajectory.length > 1) {
      ctx.strokeStyle = 'rgba(59,130,246,0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      for (let i = 0; i < trajectory.length; i++) {
        const px = toX(trajectory[i].x);
        const py = toY(f(trajectory[i].x));
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // trajectory dots (sparse)
      const step = Math.max(1, Math.floor(trajectory.length / 40));
      for (let i = 0; i < trajectory.length; i += step) {
        const px = toX(trajectory[i].x);
        const py = toY(f(trajectory[i].x));
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(59,130,246,0.35)';
        ctx.fill();
      }
    }

    // ---- current point ----
    const cx = toX(currentX);
    const cy = toY(curY);
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#ef4444'; // red
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // ---- tangent line ----
    const tExtent = (xHi - xLo) * 0.15;
    const t1 = currentX - tExtent;
    const t2 = currentX + tExtent;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(t1), toY(curY + grad * (t1 - currentX)));
    ctx.lineTo(toX(t2), toY(curY + grad * (t2 - currentX)));
    ctx.stroke();
    ctx.setLineDash([]);

    // ---- valley marker ----
    const vly = valley > xLo && valley < xHi;
    if (vly) {
      const vx = toX(valley);
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = 'rgba(34,197,94,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(vx, toY(config.valleyX ? f(valley) : yLo) + 10);
      ctx.lineTo(vx, pad.t + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      // label
      ctx.fillStyle = 'rgba(34,197,95,0.7)';
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textAlign = 'center';
      const bottomLabel = (config.valleyX ?? valley).toFixed(2);
      ctx.fillText(`谷底 ${xLabel}=${bottomLabel}`, vx, pad.t + plotH + 14);
    }

    // ---- annotation ----
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`grad=${grad.toFixed(2)}`, cx + 10, cy - 10);
    ctx.fillText(`${xLabel}=${currentX.toFixed(2)}`, cx + 10, cy + 24);

    // axes
    drawAxes(ctx, { toX, toY }, xTicks, yTicks, xLabel, yLabel, pad, plotW, plotH, fontSize);
  }

  function makeCoord(xLo: number, xHi: number, yLo: number, yHi: number, pw: number, ph: number): Coord {
    return {
      toX: (v: number) => pad.l + ((v - xLo) / (xHi - xLo)) * pw,
      toY: (v: number) => pad.t + ((yHi - v) / (yHi - yLo)) * ph,
    };
  }

  return {
    update(cfg: LossLandscapeConfig): void {
      config = cfg;
      if (cfg.trajectory) {
        trajectory = cfg.trajectory;
      }
    },
    draw,
    clearTrajectory(): void {
      trajectory = [];
    },
  };
}
