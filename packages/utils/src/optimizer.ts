import { Mat } from "./mat";

export class SGDOptimizer {
  private params: { data: Mat, grad: Mat }[];
  // learning rate（学习率）
  private lr: number;

  constructor(params: { data: Mat, grad: Mat }[], lr: number = 0.01) {
    this.params = params;
    this.lr = lr;
  }

  zeroGrad() {
    for (const p of this.params) {
      p.grad.data.fill(0);
    }
  }

  step() {
    for (const p of this.params) {
      p.data.add_(p.grad.scale(-this.lr)); // 更新参数
    }
  }
}