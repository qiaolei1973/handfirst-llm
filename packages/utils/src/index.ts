export { Mat } from './mat';
export { arr, setArr, mat, setMat } from './serial';
export { mse, lossCoeffs } from './loss';
export type { LossFn } from './loss';

export { Trainer } from './trainer/base';
export { WSServer } from './trainer/ws-server';
export { useWsTrainer } from './trainer/useWsTrainer';
export type {
  EpochEvent,
  ITrainer,
  InitData,
  ClientMsg,
  ServerMsg,
  WsTrainer,
} from './trainer/types';
