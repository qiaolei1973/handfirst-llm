'use client';

import './surgery-dashboard.css';

import { useRef, useState, useEffect, useCallback } from 'react';
import { COLORS } from '@handfirst/charts';
import { mse, lossCoeffs } from '@handfirst/utils';
import type { WsTrainer } from '@handfirst/utils';
import { useLineChart } from './useLineChart';
import { useLossLandscape } from './useLossLandscape';
import { useModelFit } from './useModelFit';

// ---- v0 专用事件形状 ----
type V0Params = { W: number; bias: number };
type V0EpochData = { params: V0Params; grads: V0Params; loss: number; epoch: number };

// ============================================================
//  Props
// ============================================================

export interface SurgeryDashboardProps {
  trainer: WsTrainer;
  trueFn: (x: number) => number;
  title?: string;
}

// ============================================================
//  组件
// ============================================================

export function SurgeryDashboard(props: SurgeryDashboardProps) {
  const {
    trainer,
    trueFn,
    title = '🔬 梯度下降的几何含义',
  } = props;

  // ---- UI 状态 ----
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [speed, setSpeed] = useState(4);

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
    const h = trainer.history as V0EpochData[];
    lossHist.setSeries([
      { label: 'Loss', color: COLORS.red,
        points: h.map((_, i) => ({ x: i + 1, y: h[i].loss })) },
    ]);
    paramHist.setSeries([
      { label: 'W', color: COLORS.blue,
        points: h.map((_, i) => ({ x: i + 1, y: h[i].params.W })) },
      { label: 'bias', color: COLORS.amber,
        points: h.map((_, i) => ({ x: i + 1, y: h[i].params.bias })) },
    ]);
    gradHist.setSeries([
      { label: '|gradW|', color: COLORS.blue,
        points: h.map((_, i) => ({ x: i + 1, y: Math.abs(h[i].grads.W) })) },
      { label: '|gradBias|', color: COLORS.amber,
        points: h.map((_, i) => ({ x: i + 1, y: Math.abs(h[i].grads.bias) })) },
    ]);
  }, [trainer, lossHist, paramHist, gradHist]);

  // ---- 渲染所有可视化 ----
  const render = useCallback((W: number, B: number, gradW: number, gradB: number, loss: number | null) => {
    const ds = trainer.dataset;
    if (!ds) return;
    const dataset = { features: ds.features, labels: ds.labels };
    const h = trainer.history as V0EpochData[];

    // 状态栏
    if (stEpRef.current) stEpRef.current.textContent = String(trainer.history.length);
    if (stLossRef.current) stLossRef.current.textContent = loss !== null ? loss.toFixed(4) : '—';
    if (stWRef.current) stWRef.current.textContent = W.toFixed(3);
    if (stBRef.current) stBRef.current.textContent = B.toFixed(3);
    if (stGWRef.current) stGWRef.current.textContent = loss !== null ? gradW.toFixed(3) : '—';
    if (stGBRef.current) stGBRef.current.textContent = loss !== null ? gradB.toFixed(3) : '—';

    // 主图
    modelFit.update({
      points: dataset.features.map((x, i) => ({ x, y: dataset.labels[i] })),
      trueFn,
      trueLabel: ds.trueFnLabel,
      W,
      bias: B,
    });
    modelFit.draw();

    // L(W) 抛物线
    const lwc = lossCoeffs('MSE', 'W', B, dataset);
    lossW.update({
      a: lwc.a, b: lwc.b, c: lwc.c,
      currentX: W,
      gradient: gradW,
      trajectory: h.map((e) => ({ x: e.params.W, y: mse(e.params.W, B, dataset) })),
      valleyX: -lwc.b / (2 * lwc.a),
    });
    lossW.draw();

    // L(bias) 抛物线
    const lbc = lossCoeffs('MSE', 'bias', W, dataset);
    lossB.update({
      a: lbc.a, b: lbc.b, c: lbc.c,
      currentX: B,
      gradient: gradB,
      trajectory: h.map((e) => ({ x: e.params.bias, y: mse(W, e.params.bias, dataset) })),
      valleyX: -lbc.b / (2 * lbc.a),
    });
    lossB.draw();

    // 公式
    if (fnWRef.current) {
      const fb = lwc.b >= 0 ? '+' + lwc.b.toFixed(2) : lwc.b.toFixed(2);
      const fc = lwc.c >= 0 ? '+' + lwc.c.toFixed(2) : lwc.c.toFixed(2);
      fnWRef.current.innerHTML = `<b>L(W)</b> = <span style="color:${COLORS.blue}">${lwc.a.toFixed(2)}</span>W² <span style="color:${COLORS.blue}">${fb}</span>W <span style="color:${COLORS.blue}">${fc}</span> &nbsp;|&nbsp; (bias 固定在 ${B.toFixed(2)})`;
    }
    if (fnBRef.current) {
      const fb = lbc.b >= 0 ? '+' + lbc.b.toFixed(2) : lbc.b.toFixed(2);
      const fc = lbc.c >= 0 ? '+' + lbc.c.toFixed(2) : lbc.c.toFixed(2);
      fnBRef.current.innerHTML = `<b>L(bias)</b> = bias² <span style="color:${COLORS.amber}">${fb}</span>bias <span style="color:${COLORS.amber}">${fc}</span> &nbsp;|&nbsp; (W 固定在 ${W.toFixed(2)})`;
    }

    lossHist.draw();
    paramHist.draw();
    gradHist.draw();
  }, [trainer, trueFn, lossW, lossB, lossHist, paramHist, gradHist, modelFit]);

  // ---- WS 事件回调 ----
  useEffect(() => {
    const u1 = trainer.onInit((initData) => {
      setReady(true);
      syncHistoryCharts();
      const p = initData.params as V0Params;
      render(p.W, p.bias, 0, 0, null);
    });
    const u2 = trainer.onEpoch((ev) => {
      const e = ev as V0EpochData;
      syncHistoryCharts();
      render(e.params.W, e.params.bias, e.grads.W, e.grads.bias, e.loss);
    });
    const u3 = trainer.onDone(() => {
      setPlaying(false);
    });
    const u4 = trainer.onReset(() => {
      const ds = trainer.dataset;
      if (!ds) return;
      const p = ds.params as V0Params;
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
      render(p.W, p.bias, 0, 0, null);
      setPlaying(false);
    });
    return () => { u1(); u2(); u3(); u4(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainer]);

  // ---- 控制 ----
  function togglePlay() {
    if (playing) {
      trainer.pause();
      setPlaying(false);
    } else {
      trainer.play(speed);
      setPlaying(true);
    }
  }

  function stepOne() {
    if (playing) { trainer.pause(); setPlaying(false); }
    trainer.step();
  }

  function resetAll() {
    if (playing) setPlaying(false);
    trainer.reset();
  }

  // ===== JSX =====
  return (
    <div className="container">
      {!ready && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(248,250,252,0.85)',
        }}>
          <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>🔌 等待训练服务连接...</span>
        </div>
      )}
      <div className="top-bar">
        <span className="title">{title}</span>
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
