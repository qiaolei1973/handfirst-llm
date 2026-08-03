'use client';

import './surgery-dashboard.css';

import { useRef, useState, useEffect, useCallback } from 'react';
import { COLORS } from '@handfirst/charts';
import { useLineChart } from './useLineChart';
import { useLossLandscape } from './useLossLandscape';
import { useModelFit } from './useModelFit';

// ============================================================
//  类型
// ============================================================

export interface EpochEvent {
  W: number;
  bias: number;
  loss: number;
  gradW: number;
  gradB: number;
}

export interface TrainerHandle {
  params: { W: number; bias: number };
  readonly epoch: number;
  readonly isDone: boolean;
  readonly history: readonly EpochEvent[];
  step(): EpochEvent;
  reset(): void;
}

export interface SurgeryDashboardProps {
  /** 训练器（由 train.ts 提供） */
  trainer: TrainerHandle;
  /** 真实函数 y=f(x) */
  trueFn: (x: number) => number;
  /** 数据集 */
  dataset: { features: number[]; labels: number[] };
  /** MSE 损失 */
  mse: (
    W: number,
    bias: number,
    dataset: { features: number[]; labels: number[] },
  ) => number;
  /** L(W) 或 L(bias) 的抛物线系数 */
  lossCoeffs: (
    lossFn: 'MSE',
    param: 'W' | 'bias',
    fixedValue: number,
    dataset: { features: number[]; labels: number[] },
  ) => { a: number; b: number; c: number };
  /** 标题（可选） */
  title?: string;
  /** 副标题（可选） */
  subtitle?: string;
}

// ============================================================
//  组件
// ============================================================

export function SurgeryDashboard(props: SurgeryDashboardProps) {
  const {
    trainer,
    trueFn,
    dataset,
    mse,
    lossCoeffs,
    title = '🔬 梯度下降的几何含义',
    subtitle = '数据空间的直线 + loss 曲面的梯度滑行',
  } = props;

  // ---- lastEvent ref（唯一需要跨渲染保持的 UI 状态） ----
  const lastEventRef = useRef<EpochEvent | null>(null);

  // ---- UI 状态 ----
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stEpRef = useRef<HTMLSpanElement>(null);
  const stLossRef = useRef<HTMLSpanElement>(null);
  const stWRef = useRef<HTMLSpanElement>(null);
  const stBRef = useRef<HTMLSpanElement>(null);
  const stGWRef = useRef<HTMLSpanElement>(null);
  const stGBRef = useRef<HTMLSpanElement>(null);
  const fnWRef = useRef<HTMLDivElement>(null);
  const fnBRef = useRef<HTMLDivElement>(null);

  // ---- chart hooks ----
  const lossW = useLossLandscape({ xLabel: 'W', yLabel: 'Loss' });
  const lossB = useLossLandscape({ xLabel: 'bias', yLabel: 'Loss' });
  const lossHist = useLineChart({ xLabel: 'Epoch', yLabel: 'MSE' });
  const paramHist = useLineChart({ xLabel: 'Epoch', yLabel: '值' });
  const gradHist = useLineChart({ xLabel: 'Epoch', yLabel: '|梯度|' });
  const modelFit = useModelFit();

  useEffect(() => {
    paramHist.setSeries([
      { label: 'W', color: COLORS.blue, points: [] },
      { label: 'bias', color: COLORS.amber, points: [] },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- 从 trainer.history 重建 chart 数据 ----
  const syncHistoryCharts = useCallback(() => {
    const h = trainer.history;
    lossHist.setSeries([
      { label: 'Loss', color: COLORS.red,
        points: h.map((ev, i) => ({ x: i + 1, y: ev.loss })) },
    ]);
    paramHist.setSeries([
      { label: 'W', color: COLORS.blue,
        points: h.map((ev, i) => ({ x: i + 1, y: ev.W })) },
      { label: 'bias', color: COLORS.amber,
        points: h.map((ev, i) => ({ x: i + 1, y: ev.bias })) },
    ]);
    gradHist.setSeries([
      { label: '|gradW|', color: COLORS.blue,
        points: h.map((ev, i) => ({ x: i + 1, y: Math.abs(ev.gradW) })) },
      { label: '|gradBias|', color: COLORS.amber,
        points: h.map((ev, i) => ({ x: i + 1, y: Math.abs(ev.gradB) })) },
    ]);
  }, [trainer, lossHist, paramHist, gradHist]);

  // ---- 更新所有可视化 ----
  const updateUI = useCallback(() => {
    const ev = lastEventRef.current;
    if (!ev) return;
    const Wraw = ev.W, Braw = ev.bias;

    // 状态栏
    if (stEpRef.current) stEpRef.current.textContent = String(trainer.epoch);
    if (stLossRef.current) stLossRef.current.textContent = ev.loss.toFixed(4);
    if (stWRef.current) stWRef.current.textContent = Wraw.toFixed(3);
    if (stBRef.current) stBRef.current.textContent = Braw.toFixed(3);
    if (stGWRef.current) stGWRef.current.textContent = ev.gradW.toFixed(3);
    if (stGBRef.current) stGBRef.current.textContent = ev.gradB.toFixed(3);

    // 主图
    modelFit.update({
      points: dataset.features.map((x, i) => ({ x, y: dataset.labels[i] })),
      trueFn,
      trueLabel: 'y=2x+10',
      W: Wraw,
      bias: Braw,
    });
    modelFit.draw();

    // L(W) 抛物线
    const lwc = lossCoeffs('MSE', 'W', Braw, dataset);
    lossW.update({
      a: lwc.a, b: lwc.b, c: lwc.c,
      currentX: Wraw,
      gradient: ev.gradW,
      trajectory: trainer.history.map((e) => ({ x: e.W, y: mse(e.W, Braw, dataset) })),
      valleyX: -lwc.b / (2 * lwc.a),
    });
    lossW.draw();

    // L(bias) 抛物线
    const lbc = lossCoeffs('MSE', 'bias', Wraw, dataset);
    lossB.update({
      a: lbc.a, b: lbc.b, c: lbc.c,
      currentX: Braw,
      gradient: ev.gradB,
      trajectory: trainer.history.map((e) => ({ x: e.bias, y: mse(Wraw, e.bias, dataset) })),
      valleyX: -lbc.b / (2 * lbc.a),
    });
    lossB.draw();

    // 公式
    if (fnWRef.current) {
      const fb = lwc.b >= 0 ? '+' + lwc.b.toFixed(2) : lwc.b.toFixed(2);
      const fc = lwc.c >= 0 ? '+' + lwc.c.toFixed(2) : lwc.c.toFixed(2);
      fnWRef.current.innerHTML = `<b>L(W)</b> = <span style="color:${COLORS.blue}">${lwc.a.toFixed(2)}</span>W² <span style="color:${COLORS.blue}">${fb}</span>W <span style="color:${COLORS.blue}">${fc}</span> &nbsp;|&nbsp; (bias 固定在 ${Braw.toFixed(2)})`;
    }
    if (fnBRef.current) {
      const fb = lbc.b >= 0 ? '+' + lbc.b.toFixed(2) : lbc.b.toFixed(2);
      const fc = lbc.c >= 0 ? '+' + lbc.c.toFixed(2) : lbc.c.toFixed(2);
      fnBRef.current.innerHTML = `<b>L(bias)</b> = bias² <span style="color:${COLORS.amber}">${fb}</span>bias <span style="color:${COLORS.amber}">${fc}</span> &nbsp;|&nbsp; (W 固定在 ${Wraw.toFixed(2)})`;
    }

    lossHist.draw();
    paramHist.draw();
    gradHist.draw();
  }, [trainer, dataset, trueFn, mse, lossCoeffs, lossW, lossB, lossHist, paramHist, gradHist]);

  // ---- 控制 ----
  const step = useCallback(() => {
    if (trainer.isDone) return;
    const ev = trainer.step();
    lastEventRef.current = ev;
  }, [trainer]);

  const tick = useCallback(() => {
    for (let i = 0; i < speed; i++) {
      if (trainer.isDone) { stop(); break; }
      step();
    }
    syncHistoryCharts();
    updateUI();
    if (trainer.isDone) stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speed, step, syncHistoryCharts, updateUI]);

  function stop() {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setPlaying(false);
  }

  function togglePlay() {
    if (playing) { stop(); return; }
    if (trainer.isDone) resetAll();
    setPlaying(true);
    timerRef.current = setInterval(tick, 100);
  }

  function stepOne() {
    if (playing) stop();
    step();
    syncHistoryCharts();
    updateUI();
  }

  function resetAll() {
    stop();
    trainer.reset();
    lastEventRef.current = null;

    lossW.update({ a: 1, b: 0, c: 0, currentX: 0 });
    lossW.draw();
    lossB.update({ a: 1, b: 0, c: 0, currentX: 0 });
    lossB.draw();
    lossHist.clear(); lossHist.draw();
    gradHist.clear(); gradHist.draw();
    paramHist.setSeries([
      { label: 'W', color: COLORS.blue, points: [] },
      { label: 'bias', color: COLORS.amber, points: [] },
    ]);

    if (stEpRef.current) stEpRef.current.textContent = '0';
    if (stLossRef.current) stLossRef.current.textContent = '—';
    if (stWRef.current) stWRef.current.textContent = '1.00';
    if (stBRef.current) stBRef.current.textContent = '0.00';
    if (stGWRef.current) stGWRef.current.textContent = '—';
    if (stGBRef.current) stGBRef.current.textContent = '—';

    modelFit.update({
      points: dataset.features.map((x, i) => ({ x, y: dataset.labels[i] })),
      trueFn,
      trueLabel: 'y=2x+10',
      W: trainer.params.W,
      bias: trainer.params.bias,
    });
    modelFit.draw();
  }

  // ---- 初始渲染 ----
  useEffect(() => {
    lastEventRef.current = {
      W: trainer.params.W,
      bias: trainer.params.bias,
      loss: mse(trainer.params.W, trainer.params.bias, dataset),
      gradW: 0,
      gradB: 0,
    };
    updateUI();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  // ===== JSX =====
  return (
    <div className="container">
      <div className="top-bar">
        <span className="title">{title}</span>
        <span className="sub">{subtitle}</span>
        <span className="sep">|</span>
        <button className={`primary${playing ? ' running' : ''}`} onClick={togglePlay}>
          {playing ? '⏸ 暂停' : '▶ 开始训练'}
        </button>
        <button onClick={stepOne}>⏭ 单步</button>
        <button onClick={resetAll}>↺ 重置</button>
        <span className="sep">|</span>
        <span>
          速度:{' '}
          <input type="range" min="1" max="15" value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))} />
        </span>
        <span className="speed-lbl">{speed}x</span>
      </div>

      <div className="status-bar">
        <div className="stat"><div className="lbl">Epoch</div><div className="val ep"><span ref={stEpRef}>0</span></div></div>
        <div className="stat"><div className="lbl">Loss</div><div className="val loss"><span ref={stLossRef}>—</span></div></div>
        <div className="stat"><div className="lbl">W</div><div className="val w"><span ref={stWRef}>1.00</span></div></div>
        <div className="stat"><div className="lbl">Bias</div><div className="val b"><span ref={stBRef}>0.00</span></div></div>
        <div className="stat"><div className="lbl">gradW</div><div className="val w"><span ref={stGWRef}>—</span></div></div>
        <div className="stat"><div className="lbl">gradBias</div><div className="val b"><span ref={stGBRef}>—</span></div></div>
      </div>

      <div className="main-card">
        <h2>📊 数据空间 &nbsp;|&nbsp; 🟢 真实函数 y=2x+10 &nbsp; 🟣 模型 &nbsp; 🔴 误差</h2>
        <canvas ref={modelFit.canvasRef} />
      </div>

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

      <div className="bottom-row-3">
        <div className="btm-card"><h3>📉 Loss 曲线</h3><canvas ref={lossHist.canvasRef} /></div>
        <div className="btm-card"><h3>📈 参数收敛</h3><canvas ref={paramHist.canvasRef} /></div>
        <div className="btm-card"><h3>📊 梯度衰减</h3><canvas ref={gradHist.canvasRef} /></div>
      </div>
    </div>
  );
}
