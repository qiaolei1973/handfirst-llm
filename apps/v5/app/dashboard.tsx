"use client";

import "./dashboard.css";
import { useState, useEffect, useRef } from "react";
import { createLineChart, COLORS, createMatmulAnim } from "@handfirst/charts";
import type { LineChartHandle } from "@handfirst/charts";
import type { WsTrainer } from "@handfirst/utils";

// ---- Types ----

interface V5EpochData {
  trainLoss: number;
  valLoss: number;
  stopped?: boolean;
  epoch: number;
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
  const [history, setHistory] = useState<{ epoch: number; trainLoss: number; valLoss: number }[]>([]);

  const lossChartRef = useRef<HTMLCanvasElement>(null);
  const matmulRef = useRef<HTMLCanvasElement>(null);
  const lossChartHandle = useRef<LineChartHandle | null>(null);

  // Epoch handler
  useEffect(() => {
    return trainer.onEpoch((ev) => {
      const data = ev as unknown as V5EpochData;
      setEpoch(data.epoch);
      setTrainLoss(data.trainLoss);
      setValLoss(data.valLoss);
      setHistory((prev) => [
        ...prev.slice(-200),
        { epoch: data.epoch, trainLoss: data.trainLoss, valLoss: data.valLoss },
      ]);
    });
  }, [trainer]);

  // 矩阵乘法动画：自洽循环，挂载时启动
  useEffect(() => {
    const canvas = matmulRef.current;
    if (!canvas) return;
    const handle = createMatmulAnim(canvas);
    return () => handle.destroy();
  }, []);

  // Loss chart render
  useEffect(() => {
    const canvas = lossChartRef.current;
    if (!canvas) return;

    if (!lossChartHandle.current) {
      lossChartHandle.current = createLineChart(canvas, {
        xLabel: "Epoch",
        yLabel: "MSE Loss",
      });
    }

    const chart = lossChartHandle.current;
    if (history.length > 1) {
      chart.setSeries([
        { label: "Train Loss", color: COLORS.blue, points: history.map((h) => ({ x: h.epoch, y: h.trainLoss })) },
        { label: "Val Loss", color: COLORS.red, points: history.map((h) => ({ x: h.epoch, y: h.valLoss })) },
      ]);
    } else {
      chart.clear();
    }
    chart.draw();
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

      <div className="dash-stack">
        <div className="dash-panel">
          <canvas ref={lossChartRef} className="loss-canvas" />
          <div className="panel-label">Loss 曲线</div>
        </div>
        <div className="dash-panel">
          <canvas ref={matmulRef} className="matmul-canvas" />
          <div className="panel-label">矩阵乘法：C[i][j] = Σ A[i][k]·B[k][j]（行 × 列，逐格相乘求和）</div>
        </div>
      </div>
    </div>
  );
}
