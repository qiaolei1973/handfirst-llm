/**
 * Linear 层 — Y = W·X + b，矩阵实现。
 *
 * 输入 X 是 [inDim × B] 矩阵（每列一个样本），一次 forward 处理 B 个样本。
 * 两个可学习参数：weight（[outDim × inDim]）和 bias（[outDim]）。
 */

import { Mat } from "@handfirst/utils";
import { Parameter } from "./parameter";

export class Linear {
  readonly weight: Parameter;
  readonly bias: Parameter;
  readonly inDim: number;
  readonly outDim: number;

  private _x: Mat | null = null;

  constructor(inDim: number, outDim: number) {
    this.inDim = inDim;
    this.outDim = outDim;

    this.weight = new Parameter(outDim * inDim);
    this.bias = new Parameter(outDim);

    for (let i = 0; i < this.weight.data.length; i++) {
      this.weight.data[i] = Math.random() * 1.2 - 0.6;
    }
    for (let j = 0; j < outDim; j++) {
      this.bias.data[j] = Math.random() * 1.2 - 0.6;
    }
  }

  // ---- 前向：Y = W @ X + b ----
  // W: [outDim × inDim]，X: [inDim × B]，b: 广播到每列

  forward(x: Mat): Mat {
    this._x = x;
    const weight = new Mat(this.outDim, this.inDim, this.weight.data);
    const bias = new Mat(this.outDim, 1, this.bias.data);
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
    for (let i = 0; i < gradWMat.data.length; i++) this.weight.grad[i] += gradWMat.data[i];

    // ∂L/∂b = ∂L/∂Y 沿 batch 维求和
    const gradBMat = gradOut.sum(1);
    for (let j = 0; j < this.outDim; j++) this.bias.grad[j] += gradBMat.data[j];

    // ∂L/∂X = W^T @ ∂L/∂Y
    const weight = new Mat(this.outDim, this.inDim, this.weight.data);
    return weight.transpose().matmul(gradOut);
  }

  // ---- 工具 ----

  zeroGrad(): void {
    this.weight.grad.fill(0);
    this.bias.grad.fill(0);
  }

  parameters(): Parameter[] {
    return [this.weight, this.bias];
  }
}
