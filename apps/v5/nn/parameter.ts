/**
 * Parameter — 一个可学习参数：数据 + 梯度，配对在一起。
 *
 * 优化器面对的就是一组 Parameter，不需要知道它是 weight 还是 bias。
 */

export class Parameter {
  readonly data: Float64Array;
  readonly grad: Float64Array;

  constructor(size: number) {
    this.data = new Float64Array(size);
    this.grad = new Float64Array(size);
  }
}
