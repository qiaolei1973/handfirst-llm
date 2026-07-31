'use client';

import { useRef, useEffect, useCallback } from 'react';
import {
  createLossLandscape,
  resizeCanvas,
  type LossLandscapeHandle,
  type LossLandscapeOpts,
  type LossLandscapeConfig,
} from '@handfirst/charts';

export function useLossLandscape(opts?: LossLandscapeOpts) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<LossLandscapeHandle | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!chartRef.current) {
      chartRef.current = createLossLandscape(canvas, opts);
    }

    const observer = new ResizeObserver(() => {
      if (!canvasRef.current || !chartRef.current) return;
      resizeCanvas(canvasRef.current);
      chartRef.current.draw();
    });
    observer.observe(canvas);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const update = useCallback((cfg: LossLandscapeConfig) => {
    chartRef.current?.update(cfg);
  }, []);

  const draw = useCallback(() => {
    chartRef.current?.draw();
  }, []);

  return { canvasRef, update, draw };
}
