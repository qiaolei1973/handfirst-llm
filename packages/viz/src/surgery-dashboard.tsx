'use client';

import './surgery-dashboard.css';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { COLORS } from '@handfirst/charts';
import { mse, lossCoeffs } from '@handfirst/utils';
import type { WsTrainer } from '@handfirst/utils';
import { LineChart } from './LineChart';
import { LossLandscape } from './LossLandscape';
import { ModelFit } from './ModelFit';

// ---- Data types ----
type V0Params = { W: number; bias: number };
type V0EpochData = { params: V0Params; grads: V0Params; loss: number; epoch: number };

// ---- Props ----
export interface SurgeryDashboardProps {
  trainer: WsTrainer;
  trueFn: (x: number) => number;
  title?: string;
}

// ============================================================
//  Component
// ============================================================

export function SurgeryDashboard(props: SurgeryDashboardProps) {
  const {
    trainer,
    trueFn,
    title = '🔬 梯度下降的几何含义',
  } = props;

  // ---- State ----
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [epoch, setEpoch] = useState(0);
  const [params, setParams] = useState<V0Params>({ W: 1, bias: 0 });
  const [grads, setGrads] = useState<V0Params>({ W: 0, bias: 0 });
  const [loss, setLoss] = useState<number | null>(null);
  const [history, setHistory] = useState<V0EpochData[]>([]);
  const [dataset, setDataset] = useState<{
    features: number[]; labels: number[]; trueFnLabel: string;
  } | null>(null);

  // ---- Derived chart data ----
  const modelFitData = useMemo(() => {
    if (!dataset) return null;
    return {
      points: dataset.features.map((x, i) => ({ x, y: dataset.labels[i] })),
      trueFn,
      trueLabel: dataset.trueFnLabel,
      W: params.W,
      bias: params.bias,
    };
  }, [dataset, trueFn, params]);

  const lossWData = useMemo(() => {
    if (!dataset) return null;
    const lwc = lossCoeffs('MSE', 'W', params.bias, { features: dataset.features, labels: dataset.labels });
    return {
      xLabel: 'W',
      yLabel: 'Loss',
      a: lwc.a, b: lwc.b, c: lwc.c,
      currentX: params.W,
      gradient: grads.W,
      trajectory: history.map((e) => ({ x: e.params.W, y: mse(e.params.W, params.bias, { features: dataset.features, labels: dataset.labels }) })),
      valleyX: -lwc.b / (2 * lwc.a),
    };
  }, [dataset, params.W, params.bias, grads.W, history]);

  const lossBData = useMemo(() => {
    if (!dataset) return null;
    const lbc = lossCoeffs('MSE', 'bias', params.W, { features: dataset.features, labels: dataset.labels });
    return {
      xLabel: 'bias',
      yLabel: 'Loss',
      a: lbc.a, b: lbc.b, c: lbc.c,
      currentX: params.bias,
      gradient: grads.bias,
      trajectory: history.map((e) => ({ x: e.params.bias, y: mse(params.W, e.params.bias, { features: dataset.features, labels: dataset.labels }) })),
      valleyX: -lbc.b / (2 * lbc.a),
    };
  }, [dataset, params.W, params.bias, grads.bias, history]);

  const lossHistSeries = useMemo(() => [{
    label: 'Loss',
    color: COLORS.red,
    points: history.map((_, i) => ({ x: i + 1, y: history[i].loss })),
  }], [history]);

  const paramHistSeries = useMemo(() => [
    { label: 'W', color: COLORS.blue, points: history.map((_, i) => ({ x: i + 1, y: history[i].params.W })) },
    { label: 'bias', color: COLORS.amber, points: history.map((_, i) => ({ x: i + 1, y: history[i].params.bias })) },
  ], [history]);

  const gradHistSeries = useMemo(() => [
    { label: '|gradW|', color: COLORS.blue, points: history.map((_, i) => ({ x: i + 1, y: Math.abs(history[i].grads.W) })) },
    { label: '|gradBias|', color: COLORS.amber, points: history.map((_, i) => ({ x: i + 1, y: Math.abs(history[i].grads.bias) })) },
  ], [history]);

  const lwFormula = useMemo(() => {
    if (!dataset) return '';
    const lwc = lossCoeffs('MSE', 'W', params.bias, { features: dataset.features, labels: dataset.labels });
    const fb = lwc.b >= 0 ? '+' + lwc.b.toFixed(2) : lwc.b.toFixed(2);
    const fc = lwc.c >= 0 ? '+' + lwc.c.toFixed(2) : lwc.c.toFixed(2);
    return `L(W) = ${lwc.a.toFixed(2)}W² ${fb}W ${fc}  |  (bias 固定在 ${params.bias.toFixed(2)})`;
  }, [dataset, params.bias]);

  const lbFormula = useMemo(() => {
    if (!dataset) return '';
    const lbc = lossCoeffs('MSE', 'bias', params.W, { features: dataset.features, labels: dataset.labels });
    const fb = lbc.b >= 0 ? '+' + lbc.b.toFixed(2) : lbc.b.toFixed(2);
    const fc = lbc.c >= 0 ? '+' + lbc.c.toFixed(2) : lbc.c.toFixed(2);
    return `L(bias) = bias² ${fb}bias ${fc}  |  (W 固定在 ${params.W.toFixed(2)})`;
  }, [dataset, params.W]);

  // ---- WS event subscriptions ----
  useEffect(() => {
    const u1 = trainer.onInit((initData) => {
      setReady(true);
      setDataset({
        features: initData.features as number[],
        labels: initData.labels as number[],
        trueFnLabel: initData.trueFnLabel as string,
      });
      const p = initData.params as V0Params;
      setParams(p);
      setGrads({ W: 0, bias: 0 });
      setLoss(null);
      setHistory([]);
      setEpoch(0);
    });

    const u2 = trainer.onEpoch((ev) => {
      const e = ev as V0EpochData;
      setEpoch(e.epoch);
      setParams(e.params);
      setGrads(e.grads);
      setLoss(e.loss);
      setHistory((prev) => [...prev, e]);
    });

    const u3 = trainer.onDone(() => {
      setPlaying(false);
    });

    const u4 = trainer.onReset(() => {
      setHistory([]);
      setEpoch(0);
      setLoss(null);
      setGrads({ W: 0, bias: 0 });
      if (trainer.dataset) {
        const p = trainer.dataset.params as V0Params;
        setParams(p);
      }
      setPlaying(false);
    });

    return () => { u1(); u2(); u3(); u4(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainer]);

  // ---- Controls ----
  const togglePlay = useCallback(() => {
    if (playing) {
      trainer.pause();
      setPlaying(false);
    } else {
      trainer.play(speed);
      setPlaying(true);
    }
  }, [playing, speed, trainer]);

  const stepOne = useCallback(() => {
    if (playing) { trainer.pause(); setPlaying(false); }
    trainer.step();
  }, [playing, trainer]);

  const resetAll = useCallback(() => {
    if (playing) setPlaying(false);
    trainer.reset();
  }, [playing, trainer]);

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
        <div className="stat"><div className="lbl">Epoch</div><div className="val ep">{epoch}</div></div>
        <div className="stat"><div className="lbl">Loss</div><div className="val loss">{loss !== null ? loss.toFixed(4) : '—'}</div></div>
        <div className="stat"><div className="lbl">W</div><div className="val w">{params.W.toFixed(3)}</div></div>
        <div className="stat"><div className="lbl">Bias</div><div className="val b">{params.bias.toFixed(3)}</div></div>
        <div className="stat"><div className="lbl">gradW</div><div className="val w">{loss !== null ? grads.W.toFixed(3) : '—'}</div></div>
        <div className="stat"><div className="lbl">gradBias</div><div className="val b">{loss !== null ? grads.bias.toFixed(3) : '—'}</div></div>
      </div>

      <div className="main-card">
        <h2>📊 数据空间</h2>
        {modelFitData && <ModelFit {...modelFitData} />}
        {!modelFitData && <canvas />}
      </div>

      <div className="bottom-row">
        <div className="btm-card">
          <h3>🎢 L(W) — 固定 bias，只看 W 变化时的 loss</h3>
          {lwFormula && <div className="fn" dangerouslySetInnerHTML={{ __html: `<b>${lwFormula}</b>` }} />}
          {lossWData && <LossLandscape {...lossWData} />}
          {!lossWData && <canvas />}
        </div>
        <div className="btm-card">
          <h3>🎢 L(bias) — 固定 W，只看 bias 变化时的 loss</h3>
          {lbFormula && <div className="fn" dangerouslySetInnerHTML={{ __html: `<b>${lbFormula}</b>` }} />}
          {lossBData && <LossLandscape {...lossBData} />}
          {!lossBData && <canvas />}
        </div>
      </div>

      <div className="bottom-row-3">
        <div className="btm-card"><h3>📉 Loss 曲线</h3>
          <LineChart series={lossHistSeries} xLabel="Epoch" yLabel="MSE" />
        </div>
        <div className="btm-card"><h3>📈 参数收敛</h3>
          <LineChart series={paramHistSeries} xLabel="Epoch" yLabel="值" />
        </div>
        <div className="btm-card"><h3>📊 梯度衰减</h3>
          <LineChart series={gradHistSeries} xLabel="Epoch" yLabel="|梯度|" />
        </div>
      </div>
    </div>
  );
}
