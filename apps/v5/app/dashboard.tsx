"use client";

import "./dashboard.css";
import { useState, useEffect, useRef } from "react";
import { createLineChart, COLORS } from "@handfirst/charts";
import type { LineChartHandle } from "@handfirst/charts";
import type { WsTrainer } from "@handfirst/utils";

// ---- Types ----

interface V5Params {
  inputDim: number;
  numNeurons: number;
  hiddenW: number[][];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

interface V5EpochData {
  params: V5Params;
  trainLoss: number;
  valLoss: number;
  isBest: boolean;
  stopped?: boolean;
  epoch: number;
}

// ---- Predict helper ----

function predict(params: V5Params, x: number[]): number {
  const { hiddenW, hiddenB, outputW, outputB } = params;
  const N = hiddenW.length;
  const d = params.inputDim;
  const h = new Float64Array(N);
  for (let j = 0; j < N; j++) {
    let sum = hiddenB[j];
    for (let i = 0; i < d; i++) sum += hiddenW[j][i] * x[i];
    h[j] = sum > 0 ? sum : 0;
  }
  let y = outputB;
  for (let j = 0; j < N; j++) y += outputW[j] * h[j];
  return y;
}

// ---- Heatmap engine ----
// Renders a scalar function f(x1,x2) over [-1,1]x[-1,1] as a coolwarm heatmap.

const HM = 80; // heatmap pixel size (per subplot)

function renderScalarField(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  fn: (x1: number, x2: number) => number,
  title: string,
) {
  ctx.clearRect(0, 0, w, h);
  const imageData = ctx.createImageData(w, h);
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const x1 = (px / w) * 2 - 1;
      const x2 = -(py / h) * 2 + 1;
      const z = fn(x1, x2);
      const t = Math.max(0, Math.min(1, (z + 1) / 2));
      const r = Math.round(t * 220 + 35);
      const g = Math.round(80 + (1 - Math.abs(t - 0.5) * 2) * 100);
      const b = Math.round((1 - t) * 220 + 35);
      const idx = (py * w + px) * 4;
      imageData.data[idx] = r;
      imageData.data[idx + 1] = Math.round(t * 50 + 100);
      imageData.data[idx + 2] = b;
      imageData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  // Title
  ctx.fillStyle = "#475569";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(title, w / 2, h - 4);
  ctx.textAlign = "start";
}

function neuronFn(params: V5Params, j: number): (x1: number, x2: number) => number {
  const d = params.inputDim;
  return (x1, x2) => {
    let z = params.hiddenB[j];
    for (let i = 0; i < d; i++) {
      z += params.hiddenW[j][i] * (i === 0 ? x1 : x2);
    }
    return z > 0 ? z : 0;
  };
}

// ---- Component ----

interface DashboardProps {
  trainer: WsTrainer;
  title?: string;
}

export function Dashboard({ trainer, title = "v5" }: DashboardProps) {
  const [playing, setPlaying] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [trainLoss, setTrainLoss] = useState(0);
  const [valLoss, setValLoss] = useState(0);
  const [params, setParams] = useState<V5Params | null>(null);
  const [history, setHistory] = useState<{ epoch: number; trainLoss: number; valLoss: number }[]>([]);

  const lossChartRef = useRef<HTMLCanvasElement>(null);
  const surfRef = useRef<HTMLCanvasElement>(null);
  const neuronRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const lossChartHandle = useRef<LineChartHandle | null>(null);
  const MAX_SHOWN = 16;

  // Epoch handler
  useEffect(() => {
    return trainer.onEpoch((ev) => {
      const data = ev as unknown as V5EpochData;
      setEpoch(data.epoch);
      setTrainLoss(data.trainLoss);
      setValLoss(data.valLoss);
      setParams(data.params);
      setHistory((prev) => [
        ...prev.slice(-200),
        { epoch: data.epoch, trainLoss: data.trainLoss, valLoss: data.valLoss },
      ]);
    });
  }, [trainer]);

  // Prediction surface heatmap
  useEffect(() => {
    const canvas = surfRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!params) {
      ctx.fillStyle = "#94a3b8"; ctx.font = "14px sans-serif";
      ctx.fillText("训练中…", w/2-30, h/2);
      return;
    }
    renderScalarField(ctx, w, h, (x1, x2) => predict(params, [x1, x2]), "x1 → (蓝=-1, 红=+1) ← x2");
  }, [params]);

  // Neuron activation heatmaps
  useEffect(() => {
    if (!params) return;
    const N = Math.min(params.numNeurons, MAX_SHOWN);
    for (let j = 0; j < N; j++) {
      const canvas = neuronRefs.current[j];
      if (!canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      renderScalarField(ctx, canvas.width, canvas.height, neuronFn(params, j), `神经元 ${j+1}`);
    }
  }, [params]);

  // Loss chart render
  useEffect(() => {
    const canvas = lossChartRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);

    if (!lossChartHandle.current) {
      lossChartHandle.current = createLineChart(ctx, canvas.clientWidth, canvas.clientHeight, {
        xLabel: "Epoch",
        yLabel: "MSE Loss",
        darkMode: false,
      });
    }

    const chart = lossChartHandle.current;
    chart.clear();
    chart.resize(canvas.clientWidth, canvas.clientHeight);

    if (history.length > 1) {
      chart.addSeries(
        history.map((h) => ({ x: h.epoch, y: h.trainLoss })),
        { stroke: COLORS.blue, lineWidth: 1.5, label: "Train Loss" },
      );
      chart.addSeries(
        history.map((h) => ({ x: h.epoch, y: h.valLoss })),
        { stroke: COLORS.red, lineWidth: 1.5, label: "Val Loss" },
      );
    }

    chart.render();
  }, [history]);

  return (
    <div className="dashboard-v5">
      <header className="dash-header">
        <h1>{title}</h1>
        <div className="dash-controls">
          <button onClick={() => { setPlaying(!playing); playing ? trainer.pause() : trainer.play(20); }}>
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => { trainer.step(); }}>▶ 单步</button>
          <button onClick={() => { trainer.reset(); setHistory([]); setEpoch(0); }}>↺ 重置</button>
        </div>
        <div className="dash-stats">
          <span>Epoch: {epoch}</span>
          <span style={{ color: COLORS.blue }}>Train Loss: {trainLoss.toFixed(5)}</span>
          <span style={{ color: COLORS.red }}>Val Loss: {valLoss.toFixed(5)}</span>
        </div>
      </header>

      <div className="dash-grid">
        <div className="dash-panel">
          <canvas ref={surfRef} width={320} height={320} style={{ width: 320, height: 320 }} />
          <div className="panel-label">预测曲面</div>
        </div>
        <div className="dash-panel">
          <canvas ref={lossChartRef} className="loss-canvas" />
          <div className="panel-label">Loss 曲线</div>
        </div>
      </div>
      <div className="neuron-grid">
        {Array.from({ length: MAX_SHOWN }, (_, j) => (
          <div key={j} className="neuron-cell">
            <canvas
              ref={(el) => { neuronRefs.current[j] = el; }}
              width={HM} height={HM}
              style={{ width: HM, height: HM }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
