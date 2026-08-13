/**
 * ReLU 激活函数 — max(0, z)，逐元素操作。
 *
 * 没有可学习参数。矩阵进来矩阵出去，对每个元素独立计算。
 * z > 0 → z，导数 = 1；z ≤ 0 → 0，导数 = 0。
 */

import { Mat } from "@handfirst/utils";
import type { Parameter } from "./parameter";

export class ReLU {
  private _z: Mat | null = null;

  forward(z: Mat): Mat {
    this._z = z;
    const out = new Mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) {
      out.data[i] = z.data[i] > 0 ? z.data[i] : 0;
    }
    return out;
  }

  backward(gradOut: Mat): Mat {
    if (!this._z) throw new Error("必须先调用 forward()");
    const z = this._z;
    const out = new Mat(gradOut.rows, gradOut.cols);
    for (let i = 0; i < z.data.length; i++) {
      out.data[i] = z.data[i] > 0 ? gradOut.data[i] : 0;
    }
    return out;
  }

  parameters(): Parameter[] {
    return [];
  }
}
