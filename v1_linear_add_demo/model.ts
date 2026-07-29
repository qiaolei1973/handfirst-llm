import { Mat } from '../utils/mat';

export abstract class Module {
  private _params: { data: Mat, grad: Mat }[] = [];
  private _moduls: Module[] = [];

  abstract forward(X: Mat): Mat;
  abstract backward(dY: Mat): Mat;

  protected registerParam(data: Mat, grad: Mat) {
    this._params.push({ data, grad });
  }

  protected registerModule(module: Module) {
    this._moduls.push(module);
  }

  getAllParams(): { data: Mat, grad: Mat }[] {
    let params = [...this._params];
    for (const m of this._moduls) {
      params = params.concat(m.getAllParams());
    }
    return params;
  }
}

export class LinearLayer extends Module {
  /** 权重矩阵 */
  W: Mat;
  /** 偏置向量 */
  bias: Mat;
  /** 权重梯度 */
  gradW: Mat;
  /** 偏置梯度 */
  gradBias: Mat;

  private _input: Mat | null = null;

  constructor(inDim: number, outDim: number) {
    super();
    this.W = Mat.from(Array.from({ length: inDim * outDim }, () => Math.random() * 0.01)).reshape(outDim, inDim);
    this.bias = Mat.zeros(outDim, 1);
    this.gradW = Mat.zeros(outDim, inDim);
    this.gradBias = Mat.zeros(outDim, 1);
  }

  /**
   * 前向传播
   * Y = X @ W^T + bias
   * @param X - 输入矩阵，形状 (batch, inDim)
   *            每一行是一个样本，每列是一个特征
   * @returns 输出矩阵，形状 (batch, outDim)
   *          每一行是对应样本经过线性变换的结果
   */
  forward(X: Mat): Mat {
    this._input = X;
    const Y = X.matmul(this.W.transpose()).add(this.bias);
    return Y;
  }

  /**
   * 反向传播: 计算 dW、db、dX
   *
   *   dW = dY^T @ X    → (outDim, batch) @ (batch, inDim) = (outDim, inDim)
   *   db = sum(dY)     → 按 batch 维求和，得到 (outDim,)
   *   dX = dY @ W      → (batch, outDim) @ (outDim, inDim) = (batch, inDim)
   *
   * @param dY - 上游梯度，形状 (batch, outDim)
   *               loss 对输出 Z + b 的偏导数
   * @returns dX - 下游梯度，形状 (batch, inDim)
   *               loss 对输入 X 的偏导数，传给前一层继续反向
   */
  backward(dY: Mat): Mat {
    const X = this._input!;
    this.gradW = dY.transpose().matmul(X);
    this.gradBias = dY.reshape(dY.rows, 1).sum(0);
    const dX = dY.matmul(this.W);
    return dX;
  }
}