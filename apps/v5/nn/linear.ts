/**
 * Linear 层 — 全连接（Wx + b），无激活。
 *
 * 这个类是神经网络里最基本的运算单元：
 *   y = W·x + b
 *
 * forward() 计算输出，同时缓存输入，供 backward() 使用。
 * backward() 接收上游梯度 ∂L/∂y，计算 ∂L/∂W、∂L/∂b 并传给下游。
 */

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

    // 随机初始化，存的初始值供 resetParameters 恢复
    for (let i = 0; i < n; i++) {
      this.w[i] = Math.random() * 1.2 - 0.6;
    }
    for (let j = 0; j < outDim; j++) {
      this.b[j] = -this.w[j * inDim] * Math.random();
    }
    this._initW = new Float64Array(this.w);
    this._initB = new Float64Array(this.b);
  }

  resetParameters(): void {
    if (this._initW) this.w.set(this._initW);
    if (this._initB) this.b.set(this._initB);
  }

  // ---- 前向 ----

  forward(x: Float64Array | number[]): Float64Array {
    const xn = x instanceof Float64Array ? x : new Float64Array(x);
    this._x = xn;

    const out = new Float64Array(this.outDim);
    for (let j = 0; j < this.outDim; j++) {
      let s = this.b[j];
      const off = j * this.inDim;
      for (let i = 0; i < this.inDim; i++) {
        s += this.w[off + i] * xn[i];
      }
      out[j] = s;
    }
    return out;
  }

  // ---- 反向 ----

  backward(gradOut: Float64Array): Float64Array {
    const x = this._x;
    if (!x) throw new Error("必须先调用 forward()");

    // 累加 ∂L/∂W 和 ∂L/∂b
    for (let j = 0; j < this.outDim; j++) {
      const go = gradOut[j];
      if (go === 0) continue;
      const off = j * this.inDim;
      for (let i = 0; i < this.inDim; i++) {
        this.gradW[off + i] += go * x[i];
      }
      this.gradB[j] += go;
    }

    // ∂L/∂x = Wᵀ · ∂L/∂y
    const gradIn = new Float64Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) {
      let s = 0;
      for (let j = 0; j < this.outDim; j++) {
        s += this.w[j * this.inDim + i] * gradOut[j];
      }
      gradIn[i] = s;
    }
    return gradIn;
  }

  // ---- 工具 ----

  zeroGrad(): void {
    this.gradW.fill(0);
    this.gradB.fill(0);
  }

  parameters(): Array<{ data: Float64Array; grad: Float64Array }> {
    return [
      { data: this.w, grad: this.gradW },
      { data: this.b, grad: this.gradB },
    ];
  }
}
