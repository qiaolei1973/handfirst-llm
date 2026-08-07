// ---- 训练事件 ----
// 基类不假设 params/grads 形状，由各 version 的 Trainer 自行定义

export interface EpochEvent {
  params: unknown;
  grads: unknown;
  loss: number;
}

// ---- Trainer 接口（WS 层不关心 params 形状） ----

export interface ITrainer {
  params: unknown;
  readonly history: readonly unknown[];
  step(): object;
  reset(): void;
  /** 可选：训练是否已完成（例如 early stopping 触发）。返回 true 时 WS 服务会发送 done 并停止。 */
  isDone?(): boolean;
}

// ---- 连接初始化（服务端第一次推送） ----

export interface InitData {
  features: number[];
  labels: number[];
  trueFnLabel: string;
  params: unknown;
}

// ---- 消息协议 ----

export type ClientMsg =
  | { type: 'step' }
  | { type: 'play'; speed: number }
  | { type: 'pause' }
  | { type: 'reset' };

export type ServerMsg =
  | { type: 'init'; data: InitData }
  | { type: 'epoch'; data: Record<string, unknown> & { epoch: number } }
  | { type: 'done' }
  | { type: 'reset'; params: unknown };

// ---- 客户端 interface（useWsTrainer 返回值） ----

export type WsTrainer = {
  readonly params: unknown;
  readonly history: readonly (EpochEvent & { epoch: number })[];
  readonly isDone: boolean;
  readonly dataset: InitData | null;
  readonly lastEvent: (EpochEvent & { epoch: number }) | null;

  step: () => void;
  play: (speed: number) => void;
  pause: () => void;
  reset: () => void;

  onInit: (fn: (data: InitData) => void) => () => void;
  onEpoch: (fn: (ev: EpochEvent & { epoch: number }) => void) => () => void;
  onDone: (fn: () => void) => () => void;
  onReset: (fn: () => void) => () => void;
};
