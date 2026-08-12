/**
 * Linear 层 — y = W·x + b，矩阵实现。
 */

import { Mat } from "@handfirst/utils";

export class Linear {
  readonly w: Float64Array;
  readonly b: Float64Array;
  readonly gradW: Float64Array;
  readonly gradB: Float64Array;
  readonly inDim: number;
  readonly outDim: number;

  private _initW: Float64Array | null = null;
  private _initB: Float64Array | null = null;
  private _x: Float64Array | null = null;

  constructor(inDim: number, outDim: number) {
    this.inDim = inDim;
    this.outDim = outDim;
    const n = outDim * inDim;

    this.w = new Float64Array(n);
    this.b = new Float64Array(outDim);
    this.gradW = new Float64Array(n);
    this.gradB = new Float64Array(outDim);

    for (let i = 0; i < n; i++) this.w[i] = Math.random() * 1.2 - 0.6;
    for (let j = 0; j < outDim; j++) this.b[j] = -this.w[j * inDim] * Math.random();

    this._initW = new Float64Array(this.w);
    this._initB = new Float64Array(this.b);
  }

  resetParameters(): void {
    if (this._initW) this.w.set(this._initW);
    if (this._initB) this.b.set(this._initB);
  }

  // ---- 前向：y = W @ x + b ----

  forward(x: Float64Array | number[]): Float64Array {
    this._x = x instanceof Float64Array ? x : new Float64Array(x);

    const W = new Mat(this.outDim, this.inDim, this.w);
    const X = new Mat(this.inDim, 1, this._x);
    const B = new Mat(this.outDim, 1, this.b);

    const out = W.matmul(X).add(B);
    return out.data;
  }

  // ---- 反向 ----

  backward(gradOut: Float64Array): Float64Array {
    if (!this._x) throw new Error("必须先调用 forward()");

    const GO = new Mat(this.outDim, 1, gradOut);
    const X  = new Mat(this.inDim, 1, this._x);

    // ∂L/∂W = GO @ X^T（外积），∂L/∂b = GO
    const dW = GO.matmul(X.transpose());
    for (let i = 0; i < dW.data.length; i++) this.gradW[i] += dW.data[i];
    for (let j = 0; j < this.outDim; j++) this.gradB[j] += GO.data[j];

    // ∂L/∂x = W^T @ gradOut
    const WT = new Mat(this.outDim, this.inDim, this.w);
    const gradIn = WT.transpose().matmul(GO);
    return gradIn.data;
  }

  // ---- 工具 ----

  zeroGrad(): void { this.gradW.fill(0); this.gradB.fill(0); }

  parameters(): Array<{ data: Float64Array; grad: Float64Array }> {
    return [
      { data: this.w, grad: this.gradW },
      { data: this.b, grad: this.gradB },
    ];
  }
}
