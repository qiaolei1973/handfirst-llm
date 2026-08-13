/**
 * Adam 优化器 — Momentum + RMSProp + 偏差修正。
 *
 *   m = β₁·m + (1-β₁)·g          (一阶矩 — 惯性)
 *   v = β₂·v + (1-β₂)·g²         (二阶矩 — 自适应步长)
 *   m̂ = m / (1-β₁ᵗ)             (偏差修正)
 *   v̂ = v / (1-β₂ᵗ)
 *   参数 -= lr · m̂ / √(v̂ + ε)
 */

import { Parameter } from "./parameter";

export class Adam {
  private _ms: Float64Array[] = [];
  private _vs: Float64Array[] = [];
  private _t = 0;

  constructor(
    private _params: Parameter[],
    private _lr = 0.001,
    private _beta1 = 0.9,
    private _beta2 = 0.999,
    private _eps = 1e-8,
  ) {
    for (const p of this._params) {
      this._ms.push(new Float64Array(p.data.length));
      this._vs.push(new Float64Array(p.data.length));
    }
  }

  reset(): void {
    for (const m of this._ms) m.fill(0);
    for (const v of this._vs) v.fill(0);
    this._t = 0;
  }

  step(): void {
    this._t++;
    for (let k = 0; k < this._params.length; k++) {
      const p = this._params[k];
      const m = this._ms[k];
      const v = this._vs[k];
      for (let i = 0; i < p.data.length; i++) {
        m[i] = this._beta1 * m[i] + (1 - this._beta1) * p.grad[i];
        v[i] = this._beta2 * v[i] + (1 - this._beta2) * p.grad[i] * p.grad[i];
        const mHat = m[i] / (1 - Math.pow(this._beta1, this._t));
        const vHat = v[i] / (1 - Math.pow(this._beta2, this._t));
        p.data[i] -= this._lr * mHat / (Math.sqrt(vHat) + this._eps);
      }
    }
  }

  get t() { return this._t; }
}
