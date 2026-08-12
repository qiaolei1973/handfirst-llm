import { WSServer } from './ws-server';

export abstract class Trainer<P = unknown, E extends object = object> {
  abstract params: P;
  readonly history: E[] = [];

  abstract step(): E;

  private _resettableModel: { resetParameters(): void } | null = null;
  private _resettableOpt: { reset(): void } | null = null;

  /** 子类 constructor 中调用一次，将 model/opt 注册到 reset() */
  protected _setupReset(
    model: { resetParameters(): void },
    opt: { reset(): void },
  ): void {
    this._resettableModel = model;
    this._resettableOpt = opt;
  }

  reset(): void {
    this.history.length = 0;
    this._resettableModel?.resetParameters();
    this._resettableOpt?.reset();
  }

  static server(opts: {
    port?: number;
    maxEpochs: number;
    features: number[] | number[][];
    labels: number[];
    trueFnLabel: string;
    factory: () => Trainer;
  }): void {
    new WSServer(opts.port ?? 3101, opts.maxEpochs, opts.factory, {
      features: opts.features,
      labels: opts.labels,
      trueFnLabel: opts.trueFnLabel,
    });
  }
}
