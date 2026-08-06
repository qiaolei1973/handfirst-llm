'use client';

import { useRef, useEffect } from 'react';
import { createModelFit } from '@handfirst/charts';
import type { ModelFitConfig, ModelFitHandle } from '@handfirst/charts';

export interface ModelFitProps {
  points: { x: number; y: number }[];
  trueFn: (x: number) => number;
  trueLabel: string;
  W: number;
  bias: number;
}

export function ModelFit({ points, trueFn, trueLabel, W, bias }: ModelFitProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<ModelFitHandle | null>(null);

  // Mount / re-create when trueFn changes (reference equality)
  useEffect(() => {
    if (!canvasRef.current) return;
    chartRef.current = createModelFit(canvasRef.current);
    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, []);

  // Sync data (exclude trueFn from deps — it's a new closure each render)
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.update({ points, trueFn, trueLabel, W, bias } as ModelFitConfig);
    chartRef.current.draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, trueLabel, W, bias]);

  return <canvas ref={canvasRef} />;
}
