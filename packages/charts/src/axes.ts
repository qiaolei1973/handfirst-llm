/**
 * Axes — coordinate management + layer list + drawing orchestration.
 *
 * Manages the data → pixel coordinate mapping, accumulates draw layers,
 * and orchestrates the full draw pipeline: grid → layers → axes → legend.
 */
import { drawGrid, drawAxes, drawLegend } from './primitives';
import type { Padding } from './types';
import { COLORS, STYLE } from './colors';

// ---- Layer types ----

export type Layer =
  | { type: 'line'; xs: number[]; ys: number[]; color: string; width?: number; dash?: number[] }
  | { type: 'scatter'; points: { x: number; y: number }[]; color: string; size?: number }
  | { type: 'vline'; x: number; color: string; dash?: number[] }
  | { type: 'hline'; y: number; color: string; dash?: number[] }
  | { type: 'segment'; x1: number; y1: number; x2: number; y2: number; color: string; dash?: number[] }
  | { type: 'annotation'; text: string; x: number; y: number; color: string; ox?: number; oy?: number };

// ---- Config ----

export interface AxesConfig {
  xLabel: string;
  yLabel: string;
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
}

// ---- Handle ----

export interface AxesHandle {
  addLayer(layer: Layer): void;
  clearLayers(): void;
  setRange(r: { xMin: number; xMax: number; yMin: number; yMax: number }): void;
  autoRange(): void;
  setLegend(items: { color: string; label: string }[]): void;
  getRange(): { xMin: number; xMax: number; yMin: number; yMax: number; pad: Padding; plotW: number; plotH: number };
  getLayers(): Layer[];
  draw(): void;
  toX(v: number): number;
  toY(v: number): number;
}

// ---- Internal state ----

interface AxesState {
  layers: Layer[];
  legendItems: { color: string; label: string }[];
  range: { xMin: number; xMax: number; yMin: number; yMax: number } | null;
  autoRangeEnabled: boolean;
}

// ---- Implementation ----

export function createAxes(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  pad: Padding,
  config: AxesConfig,
): AxesHandle {
  const state: AxesState = {
    layers: [],
    legendItems: [],
    range: null,
    autoRangeEnabled: true,
  };

  // Apply initial explicit range from config
  if (config.xMin != null && config.xMax != null && config.yMin != null && config.yMax != null) {
    state.range = { xMin: config.xMin, xMax: config.xMax, yMin: config.yMin, yMax: config.yMax };
    state.autoRangeEnabled = false;
  }

  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  // ---- internal helpers ----

  function niceTicks(lo: number, hi: number, maxN: number): number[] {
    const raw = (hi - lo) / maxN;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    let step: number;
    if (norm <= 1) step = 1 * mag;
    else if (norm <= 2) step = 2 * mag;
    else if (norm <= 5) step = 5 * mag;
    else step = 10 * mag;
    const start = Math.floor(lo / step) * step;
    const ticks: number[] = [];
    for (let v = start; v <= hi + step * 0.5; v += step) {
      if (v >= lo - step * 0.1) ticks.push(v);
    }
    return ticks;
  }

  function computeRange(): { xMin: number; xMax: number; yMin: number; yMax: number } {
    if (!state.autoRangeEnabled && state.range) return state.range;

    let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity;

    for (const layer of state.layers) {
      switch (layer.type) {
        case 'line': {
          for (const x of layer.xs) { if (x < xLo) xLo = x; if (x > xHi) xHi = x; }
          for (const y of layer.ys) { if (y < yLo) yLo = y; if (y > yHi) yHi = y; }
          break;
        }
        case 'scatter': {
          for (const pt of layer.points) { if (pt.x < xLo) xLo = pt.x; if (pt.x > xHi) xHi = pt.x; if (pt.y < yLo) yLo = pt.y; if (pt.y > yHi) yHi = pt.y; }
          break;
        }
        case 'vline': { if (layer.x < xLo) xLo = layer.x; if (layer.x > xHi) xHi = layer.x; break; }
        case 'hline': { if (layer.y < yLo) yLo = layer.y; if (layer.y > yHi) yHi = layer.y; break; }
        case 'segment': {
          if (layer.x1 < xLo) xLo = layer.x1; if (layer.x1 > xHi) xHi = layer.x1;
          if (layer.x2 < xLo) xLo = layer.x2; if (layer.x2 > xHi) xHi = layer.x2;
          if (layer.y1 < yLo) yLo = layer.y1; if (layer.y1 > yHi) yHi = layer.y1;
          if (layer.y2 < yLo) yLo = layer.y2; if (layer.y2 > yHi) yHi = layer.y2;
          break;
        }
        case 'annotation': { if (layer.x < xLo) xLo = layer.x; if (layer.x > xHi) xHi = layer.x; if (layer.y < yLo) yLo = layer.y; if (layer.y > yHi) yHi = layer.y; break; }
      }
    }

    // Fallback defaults
    if (!isFinite(xLo)) { xLo = 0; xHi = 1; yLo = 0; yHi = 1; }

    // Expand x only slightly
    const xPad = (xHi - xLo) * 0.05 || 1;
    xLo -= xPad; xHi += xPad;

    // Expand y with more room (15%)
    const yPad = (yHi - yLo) * 0.15 || 0.5;
    yLo -= yPad; yHi += yPad;

    return { xMin: xLo, xMax: xHi, yMin: yLo, yMax: yHi };
  }

  function buildCoord(range: { xMin: number; xMax: number; yMin: number; yMax: number }) {
    return {
      toX: (v: number) => pad.l + ((v - range.xMin) / (range.xMax - range.xMin)) * plotW,
      toY: (v: number) => pad.t + ((range.yMax - v) / (range.yMax - range.yMin)) * plotH,
    };
  }

  // ---- public API ----

  const api: AxesHandle = {
    addLayer(layer: Layer): void {
      state.layers.push(layer);
    },

    clearLayers(): void {
      state.layers.length = 0;
      state.legendItems.length = 0;
    },

    setRange(r: { xMin: number; xMax: number; yMin: number; yMax: number }): void {
      state.range = r;
      state.autoRangeEnabled = false;
    },

    autoRange(): void {
      state.autoRangeEnabled = true;
    },

    setLegend(items: { color: string; label: string }[]): void {
      state.legendItems = items;
    },

    toX(v: number): number {
      const range = computeRange();
      const coord = buildCoord(range);
      return coord.toX(v);
    },

    toY(v: number): number {
      const range = computeRange();
      const coord = buildCoord(range);
      return coord.toY(v);
    },

    getRange() {
      const range = computeRange();
      return { ...range, pad, plotW, plotH };
    },

    getLayers(): Layer[] {
      return state.layers;
    },

    draw(): void {
      ctx.clearRect(0, 0, w, h);

      const range = computeRange();
      const coord = buildCoord(range);
      const xTicks = niceTicks(range.xMin, range.xMax, 6);
      const yTicks = niceTicks(range.yMin, range.yMax, 5);
      const fontSize = plotW < 400 ? 8 : STYLE.font.tick;

      // 1. Grid
      drawGrid(ctx, coord, xTicks, yTicks, pad, plotW, plotH);

      // 2. Data layers
      for (const layer of state.layers) {
        ctx.setLineDash([]);
        switch (layer.type) {
          case 'line': {
            if (layer.xs.length < 2) continue;
            ctx.strokeStyle = layer.color;
            ctx.lineWidth = layer.width ?? 2;
            if (layer.dash) ctx.setLineDash(layer.dash);
            ctx.beginPath();
            ctx.moveTo(coord.toX(layer.xs[0]), coord.toY(layer.ys[0]));
            for (let i = 1; i < layer.xs.length; i++) {
              ctx.lineTo(coord.toX(layer.xs[i]), coord.toY(layer.ys[i]));
            }
            ctx.stroke();
            break;
          }
          case 'scatter': {
            const size = layer.size ?? 2.5;
            ctx.fillStyle = layer.color;
            for (const pt of layer.points) {
              ctx.beginPath();
              ctx.arc(coord.toX(pt.x), coord.toY(pt.y), size, 0, Math.PI * 2);
              ctx.fill();
            }
            break;
          }
          case 'vline': {
            ctx.strokeStyle = layer.color;
            ctx.lineWidth = 1;
            if (layer.dash) ctx.setLineDash(layer.dash);
            ctx.beginPath();
            const vx = coord.toX(layer.x);
            ctx.moveTo(vx, pad.t);
            ctx.lineTo(vx, pad.t + plotH);
            ctx.stroke();
            break;
          }
          case 'hline': {
            ctx.strokeStyle = layer.color;
            ctx.lineWidth = 1;
            if (layer.dash) ctx.setLineDash(layer.dash);
            ctx.beginPath();
            const vy = coord.toY(layer.y);
            ctx.moveTo(pad.l, vy);
            ctx.lineTo(pad.l + plotW, vy);
            ctx.stroke();
            break;
          }
          case 'segment': {
            ctx.strokeStyle = layer.color;
            ctx.lineWidth = 1.5;
            if (layer.dash) ctx.setLineDash(layer.dash);
            ctx.beginPath();
            const sx = coord.toX(layer.x1);
            const sy = coord.toY(layer.y1);
            const ex = coord.toX(layer.x2);
            const ey = coord.toY(layer.y2);
            ctx.moveTo(sx, sy);
            ctx.lineTo(ex, ey);
            ctx.stroke();
            break;
          }
          case 'annotation': {
            ctx.fillStyle = layer.color;
            ctx.font = `${fontSize}px ${STYLE.font.family}`;
            ctx.textAlign = 'left';
            const ax = coord.toX(layer.x) + (layer.ox ?? 4);
            const ay = coord.toY(layer.y) + (layer.oy ?? -5);
            ctx.fillText(layer.text, ax, ay);
            break;
          }
        }
      }
      ctx.setLineDash([]);

      // 3. Axes (spines + tick labels)
      drawAxes(ctx, coord, xTicks, yTicks, config.xLabel, config.yLabel, pad, plotW, plotH, fontSize);

      // 4. Legend
      if (state.legendItems.length > 0) {
        ctx.font = `${STYLE.legend.fontSize}px ${STYLE.font.family}`;
        const maxW = Math.max(...state.legendItems.map((it) => ctx.measureText(it.label).width));
        const lg = STYLE.legend;
        const lx = pad.l + plotW - lg.padding * 2 - lg.swatchSize * 2 - lg.gap - maxW;
        drawLegend(ctx, state.legendItems, lx, pad.t + 4);
      }
    },
  };

  return api;
}
