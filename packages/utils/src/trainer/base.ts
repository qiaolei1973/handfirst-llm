import { WSServer } from './ws-server';

export abstract class Trainer<P = unknown, E extends object = object> {
  abstract params: P;
  readonly history: E[] = [];

  abstract step(): E;

  reset(): void { this.history.length = 0; }

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
