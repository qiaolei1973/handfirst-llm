import { WSServer } from './ws-server';

export abstract class Trainer {
  readonly history: object[] = [];

  abstract step(): object;

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
