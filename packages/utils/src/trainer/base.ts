import { WSServer } from './ws-server';
import type { ITrainer } from './types';

export abstract class Trainer<TEvent = object> {
  private _epoch = 0;
  get epoch() { return this._epoch; }

  // 框架职责：驱动一次训练 + 记录步数。
  // 子类只实现 _step()（纯训练），不感知 epoch / 记录。
  step(): TEvent {
    const ev = this._step();
    this._epoch++;
    return ev;
  }

  protected abstract _step(): TEvent;

  static server(opts: {
    port?: number;
    maxEpochs: number;
    features: number[] | number[][];
    labels: number[];
    trueFnLabel: string;
    factory: () => ITrainer;
  }): void {
    new WSServer(opts.port ?? 3101, opts.maxEpochs, opts.factory, {
      features: opts.features,
      labels: opts.labels,
      trueFnLabel: opts.trueFnLabel,
    });
  }
}
