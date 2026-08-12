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

  constructor(
    inDim: number,
    outDim: number,
    readonly name = "",
  ) {
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

    const weightMat = new Mat(this.outDim, this.inDim, this.w);
    const inputMat = new Mat(this.inDim, 1, this._x);
    const biasMat  = new Mat(this.outDim, 1, this.b);

    return weightMat.matmul(inputMat).add(biasMat).data;
  }

  // ---- 反向：链式法则 ----
  //
  // 已知上游梯度 ∂L/∂y（shape [outDim × 1]），需要计算：
  //   ∂L/∂W = ∂L/∂y ⊗ x      （外积，shape [outDim × inDim]）
  //   ∂L/∂b = ∂L/∂y           （shape [outDim × 1]）
  //   ∂L/∂x = W^T · ∂L/∂y     （传给前一层，shape [inDim × 1]）

  backward(gradOut: Float64Array): Float64Array {
    if (!this._x) throw new Error("必须先调用 forward()");

    const gradOutMat = new Mat(this.outDim, 1, gradOut);
    const inputMat   = new Mat(this.inDim, 1, this._x);

    // ∂L/∂W = ∂L/∂y ⊗ x，∂L/∂b = ∂L/∂y
    const gradWMat = gradOutMat.matmul(inputMat.transpose());
    for (let i = 0; i < gradWMat.data.length; i++) this.gradW[i] += gradWMat.data[i];
    for (let j = 0; j < this.outDim; j++) this.gradB[j] += gradOutMat.data[j];

    // ∂L/∂x = W^T · ∂L/∂y
    const weightMat = new Mat(this.outDim, this.inDim, this.w);
    return weightMat.transpose().matmul(gradOutMat).data;
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
