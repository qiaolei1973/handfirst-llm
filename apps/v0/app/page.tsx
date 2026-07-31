'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import { niceTicks, drawGrid, drawAxes, resizeCanvas } from '@handfirst/charts';
import type { Coord } from '@handfirst/charts';
import { useLineChart } from '@/hooks/useLineChart';
import { useLossLandscape } from '@/hooks/useLossLandscape';

// ===== 颜色 =====
const C = {
  blue: '#3b82f6',
  amber: '#f59e0b',
  red: '#ef4444',
  purple: '#6366f1',
  green: '#22c55e',
  gray: '#94a3b8',
};

// ===== 数据生成（页面加载时跑一次） =====
const TRUE_W = 2;
const TRUE_BIAS = 10;
const N = 12;
const rawX = Array.from({ length: N }, (_, i) => i * 1.2);
const labels = rawX.map(
  (x) => TRUE_W * x + TRUE_BIAS + (Math.random() - 0.5) * 6,
);
const xMean = rawX.reduce((a, b) => a + b, 0) / N;
const xStd = Math.sqrt(
  rawX.reduce((s, x) => s + (x - xMean) ** 2, 0) / N,
);
const xNorm = rawX.map((x) => (x - xMean) / xStd);
const xMin = 0;
const xMax = Math.max(...rawX);

// ===== 损失/梯度计算（纯函数，不依赖状态） =====
const nCount = rawX.length;

function lossCoeffW(biasRaw: number): { a: number; b: number; c: number } {
  const sx2 = rawX.reduce((s, x) => s + x * x, 0);
  const sxe = rawX.reduce(
    (s, x, i) => s + 2 * x * (biasRaw - labels[i]),
    0,
  );
  const se2 = rawX.reduce(
    (s, x, i) => s + (biasRaw - labels[i]) ** 2,
    0,
  );
  return { a: sx2 / nCount, b: sxe / nCount, c: se2 / nCount };
}

function lossCoeffB(Wraw: number): { a: number; b: number; c: number } {
  const sxe = rawX.reduce(
    (s, x, i) => s + 2 * (Wraw * x - labels[i]),
    0,
  );
  const se2 = rawX.reduce(
    (s, x, i) => s + (Wraw * x - labels[i]) ** 2,
    0,
  );
  return { a: 1, b: sxe / nCount, c: se2 / nCount };
}

function lossVal(W: number, bias: number): number {
  return (
    rawX.reduce(
      (s, x, i) => s + (x * W + bias - labels[i]) ** 2,
      0,
    ) / nCount
  );
}

function gradWRaw(W: number, bias: number): number {
  return (
    rawX.reduce(
      (s, x, i) => s + 2 * x * (x * W + bias - labels[i]),
      0,
    ) / nCount
  );
}

function gradBRaw(W: number, bias: number): number {
  return (
    rawX.reduce(
      (s, x, i) => s + 2 * (x * W + bias - labels[i]),
      0,
    ) / nCount
  );
}

// ===== 主图画布绘制（自定义，使用 primitives） =====
function drawMain(
  canvas: HTMLCanvasElement,
  Wraw: number,
  Braw: number,
): void {
  const { w, h, ctx } = resizeCanvas(canvas);
  const pad = { t: 30, r: 30, b: 42, l: 58 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;

  const yAll = rawX.flatMap((x, i) => [
    labels[i],
    x * Wraw + Braw,
    TRUE_W * x + TRUE_BIAS,
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

  // 网格
  const xTicks = niceTicks(xMin, xMax, 8);
  const yTicks = niceTicks(yMin, yMax, 8);
  drawGrid(ctx, coord, xTicks, yTicks, pad, pw, ph);

  // 坐标轴
  drawAxes(ctx, coord, xTicks, yTicks, 'x', 'y', pad, pw, ph, pw < 500 ? 8 : 9);

  // 真实线 (绿色虚线)
  ctx.setLineDash([8, 5]);
  ctx.strokeStyle = 'rgba(34,197,94,0.45)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(toX(xMin), toY(TRUE_W * xMin + TRUE_BIAS));
  ctx.lineTo(toX(xMax), toY(TRUE_W * xMax + TRUE_BIAS));
  ctx.stroke();
  ctx.setLineDash([]);

  // 模型线 (紫色)
  ctx.strokeStyle = C.purple;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(toX(xMin), toY(Wraw * xMin + Braw));
  ctx.lineTo(toX(xMax), toY(Wraw * xMax + Braw));
  ctx.stroke();

  // 误差线 + 散点
  for (let i = 0; i < N; i++) {
    const xi = rawX[i];
    const yt = labels[i];
    const yp = xi * Wraw + Braw;

    // 误差线
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(239,68,68,0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toX(xi), toY(yt));
    ctx.lineTo(toX(xi), toY(yp));
    ctx.stroke();
    ctx.setLineDash([]);

    // 真实数据点
    ctx.beginPath();
    ctx.arc(toX(xi), toY(yt), 4, 0, Math.PI * 2);
    ctx.fillStyle = C.gray;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 模型预测点
    ctx.beginPath();
    ctx.arc(toX(xi), toY(yp), 2.5, 0, Math.PI * 2);
    ctx.fillStyle = C.purple;
    ctx.fill();
  }

  // 标注
  ctx.fillStyle = 'rgba(34,197,95,0.75)';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(
    'y=2x+10',
    toX(7) + 4,
    toY(TRUE_W * 7 + TRUE_BIAS) - 5,
  );
  ctx.fillStyle = C.purple;
  ctx.fillText(
    `y=${Wraw.toFixed(2)}x+${Braw.toFixed(2)}`,
    toX(8) + 4,
    toY(8 * Wraw + Braw) + 12,
  );
}

// ===== 页面组件 =====
export default function SurgeryPage() {
  const MAX_EPOCH = 600;

  // ---- 训练状态 (useRef — 不触发重渲染) ----
  const stateRef = useRef({
    W_norm: xStd, // 初始 W=1 → W_norm = 1 * xStd = xStd
    bias_norm: xMean, // 初始 bias=0 → bias_norm = bias_raw - W_raw * xMean / xStd 的反推
    epoch: 0,
    trailW: [] as { x: number; W: number }[],
    trailB: [] as { x: number; B: number }[],
    lossHistory: [] as { epoch: number; loss: number }[],
    gradWHist: [] as { epoch: number; grad: number }[],
    gradBHist: [] as { epoch: number; grad: number }[],
    gradWLast: 0,
    gradBLast: 0,
  });

  function currentW() {
    return stateRef.current.W_norm / xStd;
  }
  function currentB() {
    return stateRef.current.bias_norm - (stateRef.current.W_norm * xMean) / xStd;
  }

  // ---- UI 状态 ----
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 状态栏 DOM refs
  const stEpRef = useRef<HTMLSpanElement>(null);
  const stLossRef = useRef<HTMLSpanElement>(null);
  const stWRef = useRef<HTMLSpanElement>(null);
  const stBRef = useRef<HTMLSpanElement>(null);
  const stGWRef = useRef<HTMLSpanElement>(null);
  const stGBRef = useRef<HTMLSpanElement>(null);

  // 公式显示 DOM refs
  const fnWRef = useRef<HTMLDivElement>(null);
  const fnBRef = useRef<HTMLDivElement>(null);

  // ---- chart hooks ----
  const lossW = useLossLandscape({ xLabel: 'W', yLabel: 'Loss' });
  const lossB = useLossLandscape({ xLabel: 'bias', yLabel: 'Loss' });
  const lossHist = useLineChart({ xLabel: 'Epoch', yLabel: 'MSE' });
  const paramHist = useLineChart({ xLabel: 'Epoch', yLabel: '值' });
  const gradHist = useLineChart({ xLabel: 'Epoch', yLabel: '|梯度|' });
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);

  // ---- 初始化 history 图表 series ----
  useEffect(() => {
    paramHist.setSeries([
      { label: 'W', color: C.blue, points: [] },
      { label: 'bias', color: C.amber, points: [] },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- 训练 ----
  const oneEpoch = useCallback(() => {
    const s = stateRef.current;
    if (s.epoch >= MAX_EPOCH) return;

    // 小批量随机采样
    const batch: { x: number; y: number }[] = [];
    for (let k = 0; k < 8; k++) {
      const idx = Math.floor(Math.random() * N);
      batch.push({ x: xNorm[idx], y: labels[idx] });
    }

    // 梯度计算（归一化空间）
    let gWn = 0;
    let gBn = 0;
    for (const { x, y } of batch) {
      const d = x * s.W_norm + s.bias_norm - y;
      gWn += 2 * d * x;
      gBn += 2 * d;
    }
    gWn /= 8;
    gBn /= 8;

    // 参数更新
    const lr = 0.01;
    s.W_norm -= lr * gWn;
    s.bias_norm -= lr * gBn;

    const Wraw = currentW();
    const Braw = currentB();

    s.gradWLast = gradWRaw(Wraw, Braw);
    s.gradBLast = gradBRaw(Wraw, Braw);

    const ep = s.epoch + 1;

    s.trailW.push({ x: ep, W: Wraw });
    s.trailB.push({ x: ep, B: Braw });
    s.lossHistory.push({ epoch: ep, loss: lossVal(Wraw, Braw) });
    s.gradWHist.push({ epoch: ep, grad: Math.abs(s.gradWLast) });
    s.gradBHist.push({ epoch: ep, grad: Math.abs(s.gradBLast) });

    s.epoch = ep;
  }, []);

  // ---- 更新所有可视化 ----
  const updateUI = useCallback(() => {
    const s = stateRef.current;
    const Wraw = currentW();
    const Braw = currentB();

    // 状态栏
    if (stEpRef.current) stEpRef.current.textContent = String(s.epoch);
    if (stLossRef.current)
      stLossRef.current.textContent = lossVal(Wraw, Braw).toFixed(4);
    if (stWRef.current) stWRef.current.textContent = Wraw.toFixed(3);
    if (stBRef.current) stBRef.current.textContent = Braw.toFixed(3);
    if (stGWRef.current)
      stGWRef.current.textContent = s.gradWLast.toFixed(3);
    if (stGBRef.current)
      stGBRef.current.textContent = s.gradBLast.toFixed(3);

    // 主图
    if (mainCanvasRef.current) {
      drawMain(mainCanvasRef.current, Wraw, Braw);
    }

    // L(W)
    const lwc = lossCoeffW(Braw);
    lossW.update({
      a: lwc.a,
      b: lwc.b,
      c: lwc.c,
      currentX: Wraw,
      gradient: s.gradWLast,
      trajectory: s.trailW.map((t) => ({ x: t.W, y: lossVal(t.W, Braw) })),
      valleyX: -lwc.b / (2 * lwc.a),
    });
    lossW.draw();

    // L(bias)
    const lbc = lossCoeffB(Wraw);
    lossB.update({
      a: lbc.a,
      b: lbc.b,
      c: lbc.c,
      currentX: Braw,
      gradient: s.gradBLast,
      trajectory: s.trailB.map((t) => ({ x: t.B, y: lossVal(Wraw, t.B) })),
      valleyX: -lbc.b / (2 * lbc.a),
    });
    lossB.draw();

    // 公式文字
    if (fnWRef.current) {
      const fa = lwc.a.toFixed(2);
      const fb =
        lwc.b >= 0 ? '+' + lwc.b.toFixed(2) : lwc.b.toFixed(2);
      const fc =
        lwc.c >= 0 ? '+' + lwc.c.toFixed(2) : lwc.c.toFixed(2);
      fnWRef.current.innerHTML = `<b>L(W)</b> = <span style="color:#3b82f6">${fa}</span>W² <span style="color:#3b82f6">${fb}</span>W <span style="color:#3b82f6">${fc}</span> &nbsp;|&nbsp; (bias 固定在 ${Braw.toFixed(2)})`;
    }
    if (fnBRef.current) {
      const fb =
        lbc.b >= 0 ? '+' + lbc.b.toFixed(2) : lbc.b.toFixed(2);
      const fc =
        lbc.c >= 0 ? '+' + lbc.c.toFixed(2) : lbc.c.toFixed(2);
      fnBRef.current.innerHTML = `<b>L(bias)</b> = bias² <span style="color:#f59e0b">${fb}</span>bias <span style="color:#f59e0b">${fc}</span> &nbsp;|&nbsp; (W 固定在 ${Wraw.toFixed(2)})`;
    }

    // history charts
    lossHist.draw();
    paramHist.draw();
    gradHist.draw();
  }, [lossW, lossB, lossHist, paramHist, gradHist]);

  const drawHistoryData = useCallback(() => {
    const s = stateRef.current;
    // Loss history
    lossHist.setSeries([
      { label: 'Loss', color: C.red, points: s.lossHistory.map((h) => ({ x: h.epoch, y: h.loss })) },
    ]);
    // Param history
    paramHist.setSeries([
      { label: 'W', color: C.blue, points: s.trailW.map((t) => ({ x: t.x, y: t.W })) },
      { label: 'bias', color: C.amber, points: s.trailB.map((t) => ({ x: t.x, y: t.B })) },
    ]);
    // Grad history
    gradHist.setSeries([
      { label: '|gradW|', color: C.blue, points: s.gradWHist.map((h) => ({ x: h.epoch, y: h.grad })) },
      { label: '|gradBias|', color: C.amber, points: s.gradBHist.map((h) => ({ x: h.epoch, y: h.grad })) },
    ]);
  }, [lossHist, paramHist, gradHist]);

  // ---- 控制 ----
  const tick = useCallback(() => {
    const spd = speed;
    for (let i = 0; i < spd; i++) {
      if (stateRef.current.epoch >= MAX_EPOCH) {
        stop();
        break;
      }
      oneEpoch();
    }
    drawHistoryData();
    updateUI();

    if (stateRef.current.epoch >= MAX_EPOCH) {
      stop();
    }
  }, [speed, oneEpoch, updateUI, drawHistoryData]);

  function stop() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setPlaying(false);
  }

  function togglePlay() {
    if (playing) {
      stop();
      return;
    }
    if (stateRef.current.epoch >= MAX_EPOCH) {
      resetAll();
    }
    setPlaying(true);
    timerRef.current = setInterval(tick, 100);
  }

  function stepOne() {
    if (playing) stop();
    if (stateRef.current.epoch >= MAX_EPOCH) return;
    oneEpoch();
    drawHistoryData();
    updateUI();
  }

  function resetAll() {
    const s = stateRef.current;
    stop();
    s.W_norm = xStd;
    s.bias_norm = xMean;
    s.epoch = 0;
    s.trailW = [];
    s.trailB = [];
    s.lossHistory = [];
    s.gradWHist = [];
    s.gradBHist = [];
    s.gradWLast = 0;
    s.gradBLast = 0;

    lossW.update({
      a: 1, b: 0, c: 0, currentX: 0,
    });
    lossW.draw();
    lossB.update({
      a: 1, b: 0, c: 0, currentX: 0,
    });
    lossB.draw();

    lossHist.clear();
    lossHist.draw();
    gradHist.clear();
    gradHist.draw();

    paramHist.setSeries([
      { label: 'W', color: C.blue, points: [] },
      { label: 'bias', color: C.amber, points: [] },
    ]);

    // 状态栏
    if (stEpRef.current) stEpRef.current.textContent = '0';
    if (stLossRef.current) stLossRef.current.textContent = '—';
    if (stWRef.current) stWRef.current.textContent = '1.00';
    if (stBRef.current) stBRef.current.textContent = '0.00';
    if (stGWRef.current) stGWRef.current.textContent = '—';
    if (stGBRef.current) stGBRef.current.textContent = '—';

    // 主图初始渲染
    if (mainCanvasRef.current) {
      drawMain(mainCanvasRef.current, currentW(), currentB());
    }
  }

  // ---- 初始渲染 ----
  useEffect(() => {
    drawHistoryData();
    updateUI();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- hover 事件 ----
  useEffect(() => {
    function attachHover(
      canvas: HTMLCanvasElement | null,
      chart: ReturnType<typeof useLineChart>,
    ) {
      if (!canvas) return () => {};
      const onMove = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        chart.hoverAtPixel(e.clientX - rect.left);
      };
      const onLeave = () => {
        chart.onHoverX(null);
      };
      canvas.addEventListener('mousemove', onMove);
      canvas.addEventListener('mouseleave', onLeave);
      return () => {
        canvas.removeEventListener('mousemove', onMove);
        canvas.removeEventListener('mouseleave', onLeave);
      };
    }

    const c1 = attachHover(lossHist.canvasRef.current, lossHist);
    const c2 = attachHover(paramHist.canvasRef.current, paramHist);
    const c3 = attachHover(gradHist.canvasRef.current, gradHist);
    return () => {
      c1();
      c2();
      c3();
    };
  }, [lossHist, paramHist, gradHist]);

  // ---- cleanup timer on unmount ----
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // ===== JSX =====
  return (
    <div className="container">
      {/* 顶栏：标题 + 控件 */}
      <div className="top-bar">
        <span className="title">🔬 梯度下降的几何含义</span>
        <span className="sub">数据空间的直线 + loss 曲面的梯度滑行</span>
        <span className="sep">|</span>
        <button
          className={`primary${playing ? ' running' : ''}`}
          id="btnPlay"
          onClick={togglePlay}
        >
          {playing ? '⏸ 暂停' : '▶ 开始训练'}
        </button>
        <button onClick={stepOne}>⏭ 单步</button>
        <button onClick={resetAll}>↺ 重置</button>
        <span className="sep">|</span>
        <span>
          速度:{' '}
          <input
            type="range"
            min="1"
            max="15"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          />
        </span>
        <span className="speed-lbl">{speed}x</span>
      </div>

      {/* 状态栏 */}
      <div className="status-bar">
        <div className="stat">
          <div className="lbl">Epoch</div>
          <div className="val ep">
            <span ref={stEpRef}>0</span>
          </div>
        </div>
        <div className="stat">
          <div className="lbl">Loss</div>
          <div className="val loss">
            <span ref={stLossRef}>—</span>
          </div>
        </div>
        <div className="stat">
          <div className="lbl">W</div>
          <div className="val w">
            <span ref={stWRef}>1.00</span>
          </div>
        </div>
        <div className="stat">
          <div className="lbl">Bias</div>
          <div className="val b">
            <span ref={stBRef}>0.00</span>
          </div>
        </div>
        <div className="stat">
          <div className="lbl">gradW</div>
          <div className="val w">
            <span ref={stGWRef}>—</span>
          </div>
        </div>
        <div className="stat">
          <div className="lbl">gradBias</div>
          <div className="val b">
            <span ref={stGBRef}>—</span>
          </div>
        </div>
      </div>

      {/* 主图：数据空间 */}
      <div className="main-card">
        <h2>
          📊 数据空间 &nbsp;|&nbsp; 🟢 真实函数 y=2x+10 &nbsp; 🟣 模型
          &nbsp; 🔴 误差
        </h2>
        <canvas ref={mainCanvasRef} />
      </div>

      {/* 第二行：L(W) + L(bias) */}
      <div className="bottom-row">
        <div className="btm-card">
          <h3>🎢 L(W) — 固定 bias，只看 W 变化时的 loss</h3>
          <div className="fn" ref={fnWRef} />
          <canvas ref={lossW.canvasRef} />
        </div>
        <div className="btm-card">
          <h3>🎢 L(bias) — 固定 W，只看 bias 变化时的 loss</h3>
          <div className="fn" ref={fnBRef} />
          <canvas ref={lossB.canvasRef} />
        </div>
      </div>

      {/* 第三行：epoch history */}
      <div className="bottom-row-3">
        <div className="btm-card">
          <h3>📉 Loss 曲线</h3>
          <canvas ref={lossHist.canvasRef} />
        </div>
        <div className="btm-card">
          <h3>📈 参数收敛</h3>
          <canvas ref={paramHist.canvasRef} />
        </div>
        <div className="btm-card">
          <h3>📊 梯度衰减</h3>
          <canvas ref={gradHist.canvasRef} />
        </div>
      </div>
    </div>
  );
}
