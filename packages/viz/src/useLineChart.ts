'use client';

import { useRef, useEffect, useCallback } from 'react';
import {
  createLineChart,
  resizeCanvas,
  type LineChartHandle,
  type LineChartOpts,
  type LineSeries,
  type Point,
} from '@handfirst/charts';

export function useLineChart(opts?: LineChartOpts) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<LineChartHandle | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!chartRef.current) {
      chartRef.current = createLineChart(canvas, opts);
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

  const setSeries = useCallback((series: LineSeries[]) => {
    chartRef.current?.setSeries(series);
  }, []);

  const append = useCallback((idx: number, pt: Point) => {
    chartRef.current?.append(idx, pt);
  }, []);

  const clear = useCallback(() => {
    chartRef.current?.clear();
  }, []);

  const draw = useCallback(() => {
    chartRef.current?.draw();
  }, []);

  // ---- hover（chart 自己管理鼠标事件） ----
  const onHoverX = useCallback((xValue: number | null) => {
    chartRef.current?.onHoverX(xValue);
  }, []);

  const hoverAtPixel = useCallback((px: number) => {
    chartRef.current?.hoverAtPixel(px);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      chartRef.current?.hoverAtPixel(e.clientX - rect.left);
    };
    const onLeave = () => {
      chartRef.current?.onHoverX(null);
    };

    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseleave', onLeave);
    return () => {
      canvas.removeEventListener('mousemove', onMove);
      canvas.removeEventListener('mouseleave', onLeave);
    };
  }, []);

  return { canvasRef, setSeries, append, clear, draw };
}
