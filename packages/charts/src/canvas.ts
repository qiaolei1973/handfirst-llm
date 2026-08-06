/**
 * Canvas DOM management layer.
 * Handles ResizeObserver, DPR, and resize-driven redraw callbacks.
 */

export interface CanvasManager {
  readonly ctx: CanvasRenderingContext2D;
  readonly w: number;
  readonly h: number;
  onResize: (() => void) | null;
  destroy(): void;
}

/**
 * Set canvas pixel dimensions accounting for devicePixelRatio.
 * Returns CSS-pixel dimensions and 2D context.
 * Short-circuits if dimensions are unchanged.
 */
function resizeCanvas(canvas: HTMLCanvasElement): {
  w: number;
  h: number;
  ctx: CanvasRenderingContext2D;
} {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = r.width;
  const h = r.height;
  if (canvas.width === w * dpr && canvas.height === h * dpr) {
    return { w, h, ctx: canvas.getContext('2d')! };
  }
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.getContext('2d')!.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx: canvas.getContext('2d')! };
}

export function createCanvasManager(canvas: HTMLCanvasElement): CanvasManager {
  // Resolve initial dimensions (canvas may not be in DOM yet, so use 0 fallback)
  let state = { w: 1, h: 1, ctx: canvas.getContext('2d')! };
  try {
    state = resizeCanvas(canvas);
  } catch { /* canvas not yet in DOM */ }

  const mgr: CanvasManager = {
    get ctx() { return state.ctx; },
    get w() { return state.w; },
    get h() { return state.h; },
    onResize: null,

    destroy() {
      observer.disconnect();
    },
  };

  const observer = new ResizeObserver(() => {
    if (!canvas.isConnected) return;
    state = resizeCanvas(canvas);
    mgr.onResize?.();
  });

  observer.observe(canvas.parentElement ?? canvas);
  return mgr;
}
