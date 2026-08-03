import { WSServer } from './ws-server';

export abstract class Trainer<P = unknown, E extends object = object> {
  abstract params: P;
  readonly history: E[] = [];

  /** 一步 SGD，子类实现 */
  abstract step(): E;

  /** 重置到初始状态 */
  abstract reset(): void;

  /**
   * 启动 WebSocket 训练服务。
   * 每个客户端连接时调用 factory 创建新的 Trainer 实例。
   */
  static server(opts: {
    port: number;
    maxEpochs: number;
    features: number[];
    labels: number[];
    trueFnLabel: string;
    factory: () => Trainer;
  }): void {
    new WSServer(opts.port, opts.maxEpochs, opts.factory, {
      features: opts.features,
      labels: opts.labels,
      trueFnLabel: opts.trueFnLabel,
    });
  }
}
