/**
 * Parameter — 一个可学习参数：数据 + 梯度，配对在一起。
 *
 * 优化器面对的就是一组 Parameter，不需要知道它是 weight 还是 bias。
 * saveInit() 在初始化后调用一次，reset() 恢复到初始值。
 */

export class Parameter {
  readonly data: Float64Array;
  readonly grad: Float64Array;
  private _init: Float64Array | null = null;

  constructor(size: number) {
    this.data = new Float64Array(size);
    this.grad = new Float64Array(size);
  }

  /** 把当前 data 存为初始值（初始化完权重后调用一次） */
  saveInit(): void {
    this._init = new Float64Array(this.data);
  }

  /** 恢复到初始值 */
  reset(): void {
    if (this._init) this.data.set(this._init);
  }
}
