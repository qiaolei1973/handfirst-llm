/**
 * Linear 层 — Y = W·X + b，矩阵实现。
 *
 * 输入 X 是 [inDim × B] 矩阵（每列一个样本），一次 forward 处理 B 个样本。
 * 这就是"批处理"——矩阵乘法天然同时算 B 列。
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
  private _x: Mat | null = null;

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

  // ---- 前向：Y = W @ X + b ----
  // W: [outDim × inDim]，X: [inDim × B]，b: 广播到每列

  forward(x: Mat): Mat {
    this._x = x;
    const weight = new Mat(this.outDim, this.inDim, this.w);
    const bias = new Mat(this.outDim, 1, this.b);
    return weight.matmul(x).add(bias);
  }

  // ---- 反向：链式法则（batch 维自然累加） ----
  //
  // 已知上游梯度 ∂L/∂Y（[outDim × B]）：
  //   ∂L/∂W = ∂L/∂Y @ X^T     [outDim×B] @ [B×inDim] = [outDim×inDim]
  //   ∂L/∂b = ∂L/∂Y 沿 batch 求和  （行求和）
  //   ∂L/∂X = W^T @ ∂L/∂Y     [inDim×outDim] @ [outDim×B] = [inDim×B]

  backward(gradOut: Mat): Mat {
    if (!this._x) throw new Error("必须先调用 forward()");
    const x = this._x;

    // ∂L/∂W = ∂L/∂Y @ X^T（B 个样本的梯度自动累加进矩阵乘法）
    const gradWMat = gradOut.matmul(x.transpose());
    for (let i = 0; i < gradWMat.data.length; i++) this.gradW[i] += gradWMat.data[i];

    // ∂L/∂b = ∂L/∂Y 沿 batch 维求和
    const gradBMat = gradOut.sum(1);
    for (let j = 0; j < this.outDim; j++) this.gradB[j] += gradBMat.data[j];

    // ∂L/∂X = W^T @ ∂L/∂Y
    const weight = new Mat(this.outDim, this.inDim, this.w);
    return weight.transpose().matmul(gradOut);
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
