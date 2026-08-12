/**
 * Sequential — 把多个层串在一起，逐层调用 forward / backward。
 *
 * forward: 输入 → layer[0] → layer[1] → ... → 输出
 * backward: 梯度从输出端反向往回传
 */

import type { Linear } from "./linear";
import type { ReLU } from "./relu";

export type Module = Linear | ReLU;

export class Sequential {
  readonly layers: Module[];

  constructor(layers: Module[]) {
    this.layers = layers;
  }

  forward(x: Float64Array | number[]): Float64Array {
    let out: Float64Array = x instanceof Float64Array
      ? x
      : new Float64Array(x);
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }

  backward(gradOut: Float64Array): void {
    let g: Float64Array = gradOut;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      g = this.layers[i].backward(g);
    }
  }

  zeroGrad(): void {
    for (const layer of this.layers) {
      if ("zeroGrad" in layer) layer.zeroGrad();
    }
  }

  parameters(): Array<{ data: Float64Array; grad: Float64Array }> {
    const out: Array<{ data: Float64Array; grad: Float64Array }> = [];
    for (const layer of this.layers) {
      out.push(...layer.parameters());
    }
    return out;
  }

  /** 序列化可学习参数（给仪表盘用） */
  paramsJSON() {
    const hw = this.layers[0] as Linear, ow = this.layers[2] as Linear;
    return {
      hiddenW: Array.from(hw.w), hiddenB: Array.from(hw.b),
      outputW: Array.from(ow.w), outputB: ow.b[0],
      numNeurons: hw.outDim, inputDim: hw.inDim,
    };
  }

  /** 重置所有层参数到初始随机值 */
  resetParameters(): void {
    for (const layer of this.layers) {
      if ("resetParameters" in layer) layer.resetParameters();
    }
  }
}
