'use client';

import { useRef, useEffect, useCallback } from 'react';
import {
  createModelFit,
  resizeCanvas,
  type ModelFitHandle,
  type ModelFitConfig,
} from '@handfirst/charts';

export function useModelFit() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<ModelFitHandle | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!chartRef.current) {
      chartRef.current = createModelFit(canvas);
    }

    const observer = new ResizeObserver(() => {
      if (!canvasRef.current || !chartRef.current) return;
      resizeCanvas(canvasRef.current);
      chartRef.current.draw();
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const update = useCallback((cfg: ModelFitConfig) => {
    chartRef.current?.update(cfg);
  }, []);

  const draw = useCallback(() => {
    chartRef.current?.draw();
  }, []);

  return { canvasRef, update, draw };
}
