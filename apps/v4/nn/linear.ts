/**
 * Linear 层 — 全连接（Wx + b），无激活。
 *
 *   y = W·x + b
 *
 * 两个可学习参数：weight（权重矩阵）和 bias（偏置向量）。
 * forward() 计算输出，backward() 用链式法则计算梯度。
 */

import { Parameter } from "./parameter";

export class Linear {
  readonly weight: Parameter;
  readonly bias: Parameter;
  readonly inDim: number;
  readonly outDim: number;

  private _x: Float64Array | null = null;

  constructor(inDim: number, outDim: number) {
    this.inDim = inDim;
    this.outDim = outDim;

    this.weight = new Parameter(outDim * inDim);
    this.bias = new Parameter(outDim);

    // 随机初始化，打破对称性
    for (let i = 0; i < this.weight.data.length; i++) {
      this.weight.data[i] = Math.random() * 1.2 - 0.6;
    }
    for (let j = 0; j < outDim; j++) {
      this.bias.data[j] = Math.random() * 1.2 - 0.6;
    }

    this.weight.saveInit();
    this.bias.saveInit();
  }

  resetParameters(): void {
    this.weight.reset();
    this.bias.reset();
  }

  // ---- 前向：y = W·x + b（每个输出 = 权重 × 对应输入 + 偏置） ----

  forward(x: Float64Array | number[]): Float64Array {
    const xn = x instanceof Float64Array ? x : new Float64Array(x);
    this._x = xn;

    const out = new Float64Array(this.outDim);
    for (let j = 0; j < this.outDim; j++) {
      let s = this.bias.data[j];           // y_j = b_j + Σ_i w_{j,i} · x_i
      const off = j * this.inDim;
      for (let i = 0; i < this.inDim; i++) {
        s += this.weight.data[off + i] * xn[i];
      }
      out[j] = s;
    }
    return out;
  }

  // ---- 反向：链式法则 ----
  //
  // 已知上游梯度 ∂L/∂y，需要计算：
  //   ∂L/∂W：对每个 w_{j,i}，累加 gradOut_j · x_i
  //   ∂L/∂b：对每个 b_j，累加 gradOut_j
  //   ∂L/∂x：传给前一层 = Σ_j w_{j,i} · gradOut_j

  backward(gradOut: Float64Array): Float64Array {
    const x = this._x;
    if (!x) throw new Error("必须先调用 forward()");

    // ∂L/∂W 和 ∂L/∂b
    for (let j = 0; j < this.outDim; j++) {
      const gradY = gradOut[j];
      if (gradY === 0) continue;
      const off = j * this.inDim;
      for (let i = 0; i < this.inDim; i++) {
        this.weight.grad[off + i] += gradY * x[i];
      }
      this.bias.grad[j] += gradY;
    }

    // ∂L/∂x = W^T · ∂L/∂y（传给前一层）
    const gradIn = new Float64Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) {
      let s = 0;
      for (let j = 0; j < this.outDim; j++) {
        s += this.weight.data[j * this.inDim + i] * gradOut[j];
      }
      gradIn[i] = s;
    }
    return gradIn;
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
