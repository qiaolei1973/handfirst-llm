import { resizeCanvas, niceTicks, drawGrid, drawAxes } from './primitives';
import type { Coord } from './types';
import { COLORS } from './colors';

// ---- 配置 ----

export interface ModelFitConfig {
  /** 数据点 (x, y) */
  points: { x: number; y: number }[];
  /** 真实函数 y = f(x)，同时用于标注 */
  trueFn: (x: number) => number;
  /** 真实函数标注文字 */
  trueLabel: string;
  /** 模型参数 W（斜率） */
  W: number;
  /** 模型参数 bias（截距） */
  bias: number;
}

export interface ModelFitHandle {
  update(config: ModelFitConfig): void;
  draw(): void;
}

// ---- 实现 ----

export function createModelFit(
  canvas: HTMLCanvasElement,
): ModelFitHandle {
  let cfg: ModelFitConfig | null = null;

  function draw(): void {
    if (!cfg) return;

    const { points, trueFn, trueLabel, W, bias } = cfg;
    const N = points.length;
    const rawX = points.map((p) => p.x);
    const labels = points.map((p) => p.y);
    const xMin = 0;
    const xMax = Math.max(...rawX);

    const { w, h, ctx } = resizeCanvas(canvas);
    const pad = { t: 30, r: 30, b: 42, l: 58 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // y 范围：数据点 + 模型线 + 真实线
    const yAll = rawX.flatMap((x, i) => [
      labels[i],
      x * W + bias,
      trueFn(x),
    ]);
    let yMin = Math.min(...yAll) - 2;
    let yMax = Math.max(...yAll) + 2;
    if (yMax - yMin < 5) {
      const mid = (yMax + yMin) / 2;
      yMin = mid - 4;
      yMax = mid + 4;
    }

    const toX = (v: number) => pad.l + ((v - xMin) / (xMax - xMin)) * pw;
    const toY = (v: number) => pad.t + ((yMax - v) / (yMax - yMin)) * ph;
    const coord: Coord = { toX, toY };

    ctx.clearRect(0, 0, w, h);

    const xTicks = niceTicks(xMin, xMax, 8);
    const yTicks = niceTicks(yMin, yMax, 8);
    drawGrid(ctx, coord, xTicks, yTicks, pad, pw, ph);
    drawAxes(ctx, coord, xTicks, yTicks, 'x', 'y', pad, pw, ph, pw < 500 ? 8 : 9);

    // 真实线（绿色虚线）
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = COLORS.trueFn;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(xMin), toY(trueFn(xMin)));
    ctx.lineTo(toX(xMax), toY(trueFn(xMax)));
    ctx.stroke();
    ctx.setLineDash([]);

    // 模型线（紫色）
    ctx.strokeStyle = COLORS.purple;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(toX(xMin), toY(W * xMin + bias));
    ctx.lineTo(toX(xMax), toY(W * xMax + bias));
    ctx.stroke();

    // 误差线 + 散点
    for (let i = 0; i < N; i++) {
      const xi = rawX[i];
      const yt = labels[i];
      const yp = xi * W + bias;

      // 误差线（红色虚线）
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = COLORS.errorLine;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toX(xi), toY(yt));
      ctx.lineTo(toX(xi), toY(yp));
      ctx.stroke();
      ctx.setLineDash([]);

      // 数据点（灰色）
      ctx.beginPath();
      ctx.arc(toX(xi), toY(yt), 4, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.gray;
      ctx.fill();
      ctx.strokeStyle = COLORS.white;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 模型预测点（紫色小点）
      ctx.beginPath();
      ctx.arc(toX(xi), toY(yp), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.purple;
      ctx.fill();
    }

    // 标注
    const annotX = rawX[Math.floor(N * 0.6)];
    ctx.fillStyle = COLORS.trueFnLabel;
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(trueLabel, toX(annotX) + 4, toY(trueFn(annotX)) - 5);
    ctx.fillStyle = COLORS.purple;
    ctx.fillText(
      `y=${W.toFixed(2)}x+${bias.toFixed(2)}`,
      toX(annotX + 1) + 4,
      toY(W * (annotX + 1) + bias) + 12,
    );
  }

  return {
    update(c: ModelFitConfig): void {
      cfg = c;
    },
    draw,
  };
}
