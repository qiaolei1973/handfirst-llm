'use client';

import { useRef, useEffect } from 'react';
import { createLossLandscape } from '@handfirst/charts';
import type { Point, LossLandscapeConfig, LossLandscapeHandle } from '@handfirst/charts';

export interface LossLandscapeProps extends LossLandscapeConfig {
  xLabel?: string;
  yLabel?: string;
}

export function LossLandscape({ xLabel = 'param', yLabel = 'Loss', a, b, c, currentX, gradient, trajectory, valleyX }: LossLandscapeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<LossLandscapeHandle | null>(null);

  // Mount
  useEffect(() => {
    if (!canvasRef.current) return;
    chartRef.current = createLossLandscape(canvasRef.current, { xLabel, yLabel });
    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xLabel, yLabel]);

  // Sync data — useLayoutEffect to batch with canvas repaint
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.update({ a, b, c, currentX, gradient, trajectory, valleyX });
    chartRef.current.draw();
  }, [a, b, c, currentX, gradient, trajectory, valleyX]);

  return <canvas ref={canvasRef} />;
}
