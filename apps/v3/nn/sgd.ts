/**
 * SGD 优化器 — 朴素梯度下降。
 *
 *   参数 = 参数 - lr × 梯度
 */

import { Parameter } from "./parameter";

export class SGD {
  constructor(
    private _params: Parameter[],
    private _lr: number = 0.02,
  ) {}

  reset(): void { /* SGD 无内部状态，无需重置 */ }

  step(): void {
    for (const p of this._params) {
      for (let i = 0; i < p.data.length; i++) {
        p.data[i] -= this._lr * p.grad[i];
      }
    }
  }
}
