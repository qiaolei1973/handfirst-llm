/**
 * ReLU 激活函数 — max(0, z)。
 *
 * 没有可学习参数，只做非线性变换。
 * z > 0 → 输出 z 本身，导数 = 1
 * z ≤ 0 → 输出 0，导数 = 0
 */

import type { Parameter } from "./parameter";

export class ReLU {
  private _z: Float64Array | null = null;

  forward(z: Float64Array): Float64Array {
    this._z = z;
    const out = new Float64Array(z.length);
    for (let i = 0; i < z.length; i++) {
      out[i] = z[i] > 0 ? z[i] : 0;
    }
    return out;
  }

  backward(gradOut: Float64Array): Float64Array {
    const z = this._z;
    if (!z) throw new Error("必须先调用 forward()");
    const out = new Float64Array(z.length);
    for (let i = 0; i < z.length; i++) {
      out[i] = z[i] > 0 ? gradOut[i] : 0;
    }
    return out;
  }

  parameters(): Parameter[] {
    return [];
  }
}
