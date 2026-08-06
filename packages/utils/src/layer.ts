/**
 * Layer — 神经网络层，封装 Linear + Activation。
 *
 * 每个 Layer 接收一个样本（number[]），输出一个向量（number[]）。
 * forward() 做前向传播，backward() 累加梯度到 this.gradW / this.gradB。
 *
 * v3 只用 forward() + 更新参数（黑盒），backward() 的数学推导留给 v4。
 */

export type Activation = 'relu' | 'linear';

export class Layer {
  readonly w: Float64Array;
  readonly b: Float64Array;
  readonly gradW: Float64Array;
  readonly gradB: Float64Array;
  readonly inputDim: number;
  readonly outputDim: number;
  readonly activation: Activation;

  // 反向传播缓存（v3: 黑盒；v4: 展开讲）
  private _x: number[] | null = null;
  private _z: Float64Array | null = null;

  constructor(
    inputDim: number,
    outputDim: number,
    activation: Activation = 'linear',
  ) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.activation = activation;

    // 小随机值初始化权重，打破对称性
    this.w = new Float64Array(outputDim * inputDim);
    for (let i = 0; i < this.w.length; i++) {
      this.w[i] = Math.random() * 1.2 - 0.6;
    }

    // 偏置：ReLU 让每个神经元的"折点"随机分布在输入范围内，linear 用 0
    // 折点 x_kink = -b/w：w>0 时 ReLU 在 x>x_kink 激活；w<0 时在 x<x_kink 激活
    this.b = new Float64Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      if (activation === 'relu') {
        const r = Math.random();          // 折点位置 ∈ [0, 1]
        this.b[i] = -this.w[i] * r;       // b = -w·r → w·r + b = 0
      } else {
        this.b[i] = 0;
      }
    }

    // 梯度累加器（每次 backward 会累加，调用方负责清零）
    this.gradW = new Float64Array(this.w.length);
    this.gradB = new Float64Array(outputDim);
  }

  // ---- 前向传播 ----

  /**
   * 前向传播: x → W·x + b → activation
   *
   * @param x 输入向量 (inputDim 个元素)
   * @returns 输出向量 (outputDim 个元素)
   */
  forward(x: number[]): Float64Array {
    this._x = x;
    const z = new Float64Array(this.outputDim);

    // 线性变换: z_j = b_j + Σ_i w_{j,i} * x_i
    for (let j = 0; j < this.outputDim; j++) {
      let sum = this.b[j];
      for (let i = 0; i < this.inputDim; i++) {
        sum += this.w[j * this.inputDim + i] * x[i];
      }
      z[j] = sum;
    }
    this._z = z;

    // 激活函数
    if (this.activation === 'relu') {
      const out = new Float64Array(this.outputDim);
      for (let j = 0; j < this.outputDim; j++) {
        out[j] = z[j] > 0 ? z[j] : 0;
      }
      return out;
    }
    return z; // linear: 直通
  }

  // ---- 反向传播（黑盒 — v4 展开讲数学） ----

  /**
   * 反向传播：计算梯度并**累加**到 this.gradW / this.gradB。
   *
   * @param gradOutput 上游传来的梯度 (outputDim 个元素)
   * @returns 传递给前一层的梯度 (inputDim 个元素)
   */
  backward(gradOutput: Float64Array): Float64Array {
    if (!this._x || !this._z) {
      throw new Error('必须先调用 forward() 再调用 backward()');
    }

    // 激活函数的梯度: gradPreAct = gradOutput ⊙ activation'(z)
    let gradPreAct: Float64Array;
    if (this.activation === 'relu') {
      gradPreAct = new Float64Array(gradOutput.length);
      for (let j = 0; j < gradOutput.length; j++) {
        gradPreAct[j] = this._z[j] > 0 ? gradOutput[j] : 0;
      }
    } else {
      gradPreAct = gradOutput; // linear: 直通
    }

    // 累加 gradW: gradW_{j,i} += gradPreAct_j * x_i
    for (let j = 0; j < this.outputDim; j++) {
      for (let i = 0; i < this.inputDim; i++) {
        this.gradW[j * this.inputDim + i] += gradPreAct[j] * this._x[i];
      }
      // 累加 gradB
      this.gradB[j] += gradPreAct[j];
    }

    // 计算传给前一层的梯度: gradInput_i = Σ_j W_{j,i} * gradPreAct_j
    const gradInput = new Float64Array(this.inputDim);
    for (let i = 0; i < this.inputDim; i++) {
      let sum = 0;
      for (let j = 0; j < this.outputDim; j++) {
        sum += this.w[j * this.inputDim + i] * gradPreAct[j];
      }
      gradInput[i] = sum;
    }

    return gradInput;
  }

  // ---- 梯度清零 ----

  zeroGrad(): void {
    this.gradW.fill(0);
    this.gradB.fill(0);
  }
}
