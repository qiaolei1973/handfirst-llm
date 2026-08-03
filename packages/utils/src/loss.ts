/**
 * Loss function utilities.
 *
 * These are pure math — they don't depend on training state or charts.
 * train.ts has its own inline MSE for learning; this module is what the UI
 * calls to draw the L(W) / L(bias) parabola charts.
 */

export type LossFn = 'MSE';

// ---- MSE ------------------------------------------------------------------

/**
 * Compute full-dataset MSE: L = (1/n) Σ (x·W + bias - y)²
 */
export function mse(
  W: number,
  bias: number,
  dataset: { features: number[]; labels: number[] },
): number {
  const { features: X, labels: Y } = dataset;
  const n = X.length;
  return X.reduce((s, x, i) => {
    const d = x * W + bias - Y[i];
    return s + d * d;
  }, 0) / n;
}

// ---- Parabola coefficients -------------------------------------------------
//
// L(W) or L(bias) is always a parabola for MSE + linear model.
//
//   L = (1/n) Σ (x·W + bias - y)²
//
// Fix one variable → the loss as a function of the other is quadratic:
//   L(W)    = a·W² + b·W + c       (bias fixed)
//   L(bias) = bias² + b·bias + c   (W fixed, a = 1)
//
// The chart draws these parabola curves + the current point's tangent.

export function lossCoeffs(
  lossFn: LossFn,
  param: 'W' | 'bias',
  fixedValue: number, // the OTHER param's current value
  dataset: { features: number[]; labels: number[] },
): { a: number; b: number; c: number } {
  const { features: X, labels: Y } = dataset;
  const n = X.length;

  if (param === 'W') {
    // L(W) = aW² + bW + c   with bias = fixedValue
    const bias = fixedValue;
    const sx2 = X.reduce((s, x) => s + x * x, 0);
    const sxe = X.reduce((s, x, i) => s + 2 * x * (bias - Y[i]), 0);
    const se2 = X.reduce((s, _, i) => s + (bias - Y[i]) ** 2, 0);
    return { a: sx2 / n, b: sxe / n, c: se2 / n };
  } else {
    // L(bias) = bias² + b·bias + c   with W = fixedValue
    const W = fixedValue;
    const sxe = X.reduce((s, x, i) => s + 2 * (W * x - Y[i]), 0);
    const se2 = X.reduce((s, x, i) => s + (W * x - Y[i]) ** 2, 0);
    return { a: 1, b: sxe / n, c: se2 / n };
  }
}
