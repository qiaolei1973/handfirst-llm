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

interface V4Params {
  numNeurons: number;
  hiddenW: number[];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

interface V4EpochData {
  params: V4Params;
  trainLoss: number;
  valLoss: number;
  stopped?: boolean;
  epoch: number;
}

// 归一化因子：模型在 x̂ ∈ [0,1] 上训练，图表还原到原始 x ∈ [0,2π]
const X_SCALE = 2 * Math.PI;

const NEURON_COLORS = [
  '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6',
  '#10b981', '#ec4899', '#06b6d4', '#f97316',
  '#84cc16', '#f43f5e', '#14b8a6', '#eab308',
  '#a855f7', '#22d3ee', '#fb923c', '#a3e635',
];

// ---- Predict helper ----

function makePredict(params: V4Params): (x: number) => number {
  const { hiddenW, hiddenB, outputW, outputB } = params;
  const N = hiddenW.length;
  return (x: number): number => {
    let sum = 0;
    for (let i = 0; i < N; i++) {
      sum += outputW[i] * Math.max(0, x * hiddenW[i] + hiddenB[i]);
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
  title = '优化曲线',
}: DashboardProps) {
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [epoch, setEpoch] = useState(0);
  const [params, setParams] = useState<V4Params | null>(null);
  const [trainLoss, setTrainLoss] = useState<number | null>(null);
  const [valLoss, setValLoss] = useState<number | null>(null);
  const [bestValLoss, setBestValLoss] = useState(Infinity);
  const isBest = valLoss !== null && valLoss < bestValLoss;
  const [stopped, setStopped] = useState(false);
  const [history, setHistory] = useState<V4EpochData[]>([]);
  const [dataset, setDataset] = useState<{
    features: number[];
    labels: number[];
    trueFnLabel: string;
  } | null>(null);

  // Canvas refs
  const mcCanvasRef = useRef<HTMLCanvasElement>(null);
  const raCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const mcRef = useRef<ModelCurveHandle | null>(null);
  const raRef = useRef<ReLUActivationsHandle | null>(null);
  const lossRef = useRef<LineChartHandle | null>(null);

  // Mount charts
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

  // v4: 双 loss 曲线 — 训练 loss + 验证 loss
  const lossSeries = useMemo(
    () => [
      {
        label: '训练 Loss',
        color: COLORS.blue,
        points: history.map((_, i) => ({ x: i + 1, y: history[i].trainLoss })),
      },
      {
        label: '验证 Loss',
        color: COLORS.red,
        points: history.map((_, i) => ({ x: i + 1, y: history[i].valLoss })),
      },
    ],
    [history],
  );

  // ---- Sync charts ----
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
      const p = initData.params as V4Params;
      setParams(p);
      setTrainLoss(null);
      setValLoss(null);
      setBestValLoss(Infinity);
      setStopped(false);
      setHistory([]);
      setEpoch(0);
    });

    const u2 = trainer.onEpoch((ev) => {
      const e = ev as unknown as V4EpochData;
      setEpoch(e.epoch);
      setParams(e.params);
      setTrainLoss(e.trainLoss);
      setValLoss(e.valLoss);
      if (e.valLoss < bestValLoss) setBestValLoss(e.valLoss);
      setHistory((prev) => [...prev, e]);

      if (e.stopped) {
        setStopped(true);
        setPlaying(false);
      }
    });

    const u3 = trainer.onDone(() => {
      setPlaying(false);
    });

    const u4 = trainer.onReset(() => {
      setHistory([]);
      setEpoch(0);
      setTrainLoss(null);
      setValLoss(null);
      setBestValLoss(Infinity);
      setStopped(false);
      if (trainer.dataset) {
        const p = trainer.dataset.params as V4Params;
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
        <div className="overlay">
          <span>🔌 等待训练服务连接...</span>
        </div>
      )}

      {/* ---- Controls ---- */}
      <div className="top-bar">
        <span className="title">{title}</span>
        <span className="sep">|</span>
        <button
          className={`primary${playing ? ' running' : ''}`}
          onClick={togglePlay}
          disabled={stopped}
        >
          {stopped ? '✅ 训练完成' : playing ? '⏸ 暂停' : '▶ 开始训练'}
        </button>
        <button onClick={stepOne} disabled={stopped}>⏭ 单步</button>
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
        {stopped && <span className="stopped-badge">⏹ Early Stop</span>}
      </div>

      {/* ---- Status Bar ---- */}
      <div className="status-bar">
        <div className="stat">
          <div className="lbl">Epoch</div>
          <div className="val ep">{epoch}</div>
        </div>
        <div className="stat">
          <div className="lbl">训练 Loss</div>
          <div className="val train-loss">
            {trainLoss !== null ? trainLoss.toFixed(6) : '—'}
          </div>
        </div>
        <div className="stat">
          <div className="lbl">
            验证 Loss{isBest ? ' ★' : ''}
          </div>
          <div className="val val-loss">
            {valLoss !== null ? valLoss.toFixed(6) : '—'}
          </div>
        </div>
        <div className="stat">
          <div className="lbl">隐藏层</div>
          <div className="val neurons">
            {params ? `${params.numNeurons} ReLU` : '—'}
          </div>
        </div>
      </div>

      {/* ---- Row 1: Model Curve ---- */}
      <div className="main-card">
        <h2>📊 数据空间 — 折线逼近 sin(x)</h2>
        <canvas ref={mcCanvasRef} />
      </div>

      {/* ---- Row 2: ReLU + Loss ---- */}
      <div className="bottom-row">
        <div className="btm-card">
          <h3>🧩 ReLU 神经元的输出</h3>
          <canvas ref={raCanvasRef} />
        </div>

        <div className="btm-card">
          <h3>📉 Loss 曲线 — 训练 vs 验证</h3>
          <canvas ref={lossCanvasRef} />
        </div>
      </div>
    </div>
  );
}
