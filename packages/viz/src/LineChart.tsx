'use client';

import { useRef, useEffect } from 'react';
import { createLineChart } from '@handfirst/charts';
import type { LineSeries, LineChartHandle } from '@handfirst/charts';

export interface LineChartProps {
  series: LineSeries[];
  xLabel?: string;
  yLabel?: string;
}

export function LineChart({ series, xLabel = 'x', yLabel = 'y' }: LineChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<LineChartHandle | null>(null);

  // Mount / re-create when labels change
  useEffect(() => {
    if (!canvasRef.current) return;
    chartRef.current = createLineChart(canvasRef.current, { xLabel, yLabel });

    // Initial draw with current data
    chartRef.current.setSeries(series);
    chartRef.current.draw();

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xLabel, yLabel]);

  // Sync series data
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.setSeries(series);
    chartRef.current.draw();
  }, [series]);

  return <canvas ref={canvasRef} />;
}
