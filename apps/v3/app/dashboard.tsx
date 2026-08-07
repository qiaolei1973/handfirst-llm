'use client';

import './dashboard.css';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  createModelCurve,
  createReLUActivations,
  createLineChart,
  COLORS,
} from '@handfirst/charts';
import type {
  ModelCurveHandle,
  ReLUActivationsHandle,
  LineChartHandle,
} from '@handfirst/charts';
import type { WsTrainer, InitData } from '@handfirst/utils';

// ---- Types ----

interface V3Params {
  numNeurons: number;
  hiddenW: number[];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

interface V3EpochData {
  params: V3Params;
  loss: number;
  epoch: number;
}

// 归一化因子：模型在 x̂ ∈ [0,1] 上训练，图表还原到原始 x ∈ [0,2π]。
// 这只是为了让 x 轴刻度可读的演示转换，实际训练中不需要。
const X_SCALE = 2 * Math.PI;

// Neuron color palette
const NEURON_COLORS = [
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#10b981', // emerald
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#f43f5e', // rose
  '#14b8a6', // teal
  '#eab308', // yellow
  '#a855f7', // purple
  '#22d3ee', // sky
  '#fb923c', // orange-light
  '#a3e635', // green-light
];

// ---- Formula display helper ----

function formatFormula(params: V3Params): string {
  const { hiddenW, hiddenB, outputW, outputB } = params;
  const terms: string[] = [];

  for (let i = 0; i < hiddenW.length; i++) {
    const wo = outputW[i];
    if (Math.abs(wo) < 0.001) continue; // skip near-zero contributions

    const sign = wo >= 0 ? '+' : '−';
    const absWo = Math.abs(wo).toFixed(3);
    const wi = hiddenW[i].toFixed(2);
    const bi = hiddenB[i].toFixed(2);
    const biSign = hiddenB[i] >= 0 ? '+' : '';

    terms.push(`${sign} ${absWo}·ReLU(${wi}x̂ ${biSign}${bi})`);
  }

  const bOutStr =
    outputB >= 0
      ? `+ ${outputB.toFixed(3)}`
      : `− ${Math.abs(outputB).toFixed(3)}`;

  return `f(x̂) = ${terms.join(' ')}  ${bOutStr}`;
}

// ---- Model prediction helper (JavaScript, no Layer import needed) ----

function makePredict(params: V3Params): (x: number) => number {
  const { hiddenW, hiddenB, outputW, outputB } = params;
  const N = hiddenW.length;
  return (x: number): number => {
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const a = Math.max(0, x * hiddenW[i] + hiddenB[i]); // ReLU
      sum += outputW[i] * a;
    }
    return sum + outputB;
  };
}

// ---- Props ----

interface DashboardProps {
  trainer: WsTrainer;
  trueFn: (x: number) => number;
  title?: string;
}

// ============================================================
//  Component
// ============================================================

export function Dashboard({
  trainer,
  trueFn,
  title = '画曲线',
}: DashboardProps) {
  // ---- State ----
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [epoch, setEpoch] = useState(0);
  const [params, setParams] = useState<V3Params | null>(null);
  const [loss, setLoss] = useState<number | null>(null);
  const [history, setHistory] = useState<V3EpochData[]>([]);
  const [dataset, setDataset] = useState<{
    features: number[];
    labels: number[];
    trueFnLabel: string;
  } | null>(null);

  // ---- Canvas refs ----
  const mcCanvasRef = useRef<HTMLCanvasElement>(null);
  const raCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const mcRef = useRef<ModelCurveHandle | null>(null);
  const raRef = useRef<ReLUActivationsHandle | null>(null);
  const lossRef = useRef<LineChartHandle | null>(null);

  // ---- Mount charts ----
  useEffect(() => {
    if (mcCanvasRef.current) {
      mcRef.current = createModelCurve(mcCanvasRef.current);
    }
    if (raCanvasRef.current) {
      raRef.current = createReLUActivations(raCanvasRef.current);
    }
    if (lossCanvasRef.current) {
      lossRef.current = createLineChart(lossCanvasRef.current, {
        xLabel: 'Epoch',
        yLabel: 'MSE',
      });
    }
    return () => {
      mcRef.current?.destroy();
      raRef.current?.destroy();
      lossRef.current?.destroy();
    };
  }, []);

  // ---- Derived chart data ----

  const modelCurveData = useMemo(() => {
    if (!dataset || !params) return null;
    // 模型在归一化 x̂ ∈ [0,1] 上训练，图表显示原始 x ∈ [0,2π]。
    // 这只是视觉效果——把 x 轴放大 2π 倍、预测函数做反向缩放。
    const predictNorm = makePredict(params);
    const displayPredict = (xOrig: number) => predictNorm(xOrig / X_SCALE);
    const displayTrueFn = (xOrig: number) => trueFn(xOrig);
    return {
      points: dataset.features.map((xNorm, i) => ({
        x: xNorm * X_SCALE,
        y: dataset.labels[i],
      })),
      trueFn: displayTrueFn,
      trueLabel: 'y = sin(x)',
      predict: displayPredict,
      modelLabel: `模型  (${params.numNeurons} 个 ReLU 神经元)`,
    };
  }, [dataset, params, trueFn]);

  const reluData = useMemo(() => {
    if (!dataset || !params) return null;
    // ReLU 图表也显示原始 x ∈ [0,2π]：将 w 除以 2π，
    // 使 ReLU(w_disp * x_orig + b) = ReLU(w * x̂ + b)
    return {
      xRange: { min: 0, max: X_SCALE },
      neurons: params.hiddenW.map((w, i) => ({
        w: w / X_SCALE,
        b: params.hiddenB[i],
        label: `Neuron ${i + 1}`,
        color: NEURON_COLORS[i % NEURON_COLORS.length],
      })),
    };
  }, [dataset, params]);

  const lossSeries = useMemo(
    () => [
      {
        label: 'Loss',
        color: COLORS.red,
        points: history.map((_, i) => ({ x: i + 1, y: history[i].loss })),
      },
    ],
    [history],
  );

  // ---- Sync charts with data ----
  useEffect(() => {
    if (!mcRef.current || !modelCurveData) return;
    mcRef.current.update(modelCurveData);
    mcRef.current.draw();
  }, [modelCurveData]);

  useEffect(() => {
    if (!raRef.current || !reluData) return;
    raRef.current.update(reluData);
    raRef.current.draw();
  }, [reluData]);

  useEffect(() => {
    if (!lossRef.current) return;
    lossRef.current.setSeries(lossSeries);
    lossRef.current.draw();
  }, [lossSeries]);

  // ---- WS event subscriptions ----
  useEffect(() => {
    const u1 = trainer.onInit((initData: InitData) => {
      setReady(true);
      setDataset({
        features: initData.features as number[],
        labels: initData.labels as number[],
        trueFnLabel: initData.trueFnLabel as string,
      });
      const p = initData.params as V3Params;
      setParams(p);
      setLoss(null);
      setHistory([]);
      setEpoch(0);
    });

    const u2 = trainer.onEpoch((ev) => {
      const e = ev as unknown as V3EpochData;
      setEpoch(e.epoch);
      setParams(e.params);
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
      if (trainer.dataset) {
        const p = trainer.dataset.params as V3Params;
        setParams(p);
      }
      setPlaying(false);
    });

    return () => {
      u1();
      u2();
      u3();
      u4();
    };
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
    if (playing) {
      trainer.pause();
      setPlaying(false);
    }
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
        <div
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(248,250,252,0.85)',
          }}
        >
          <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
            🔌 等待训练服务连接...
          </span>
        </div>
      )}

      {/* ---- Controls ---- */}
      <div className="top-bar">
        <span className="title">{title}</span>
        <span className="sep">|</span>
        <button
          className={`primary${playing ? ' running' : ''}`}
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

      {/* ---- Status Bar ---- */}
      <div className="status-bar">
        <div className="stat">
          <div className="lbl">Epoch</div>
          <div className="val ep">{epoch}</div>
        </div>
        <div className="stat">
          <div className="lbl">Loss</div>
          <div className="val loss">
            {loss !== null ? loss.toFixed(6) : '—'}
          </div>
        </div>
        <div className="stat">
          <div className="lbl">隐藏层</div>
          <div className="val neurons">
            {params ? `${params.numNeurons} ReLU` : '—'}
          </div>
        </div>
      </div>

      {/* ---- Formula ---- */}
      {params && (
        <div className="formula-bar">
          <code>{formatFormula(params)}</code>
          <span className="note">（x̂ = x/2π，图表轴已还原为原始 x）</span>
        </div>
      )}

      {/* ---- Row 1: Model Curve ---- */}
      <div className="main-card">
        <h2>📊 数据空间 — 折线逼近 sin(x)</h2>
        <canvas ref={mcCanvasRef} />
      </div>

      {/* ---- Row 2: ReLU Activations ---- */}
      <div className="bottom-row">
        <div className="btm-card">
          <h3>🧩 ReLU 神经元的输出</h3>
          <canvas ref={raCanvasRef} />
        </div>

        {/* ---- Row 2 right: Loss Curve ---- */}
        <div className="btm-card">
          <h3>📉 Loss 曲线</h3>
          <canvas ref={lossCanvasRef} />
        </div>
      </div>
    </div>
  );
}
