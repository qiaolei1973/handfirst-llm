// ============================================================
// 多维输入 + 批处理 — d 个特征、B 个样本一次算完
//
// 模型: H = ReLU(W_h @ X + b_h)     W_h: [N × d], X: [d × B]
//        Ŷ = W_o @ H + b_o          W_o: [1 × N], Ŷ: [1 × B]
// ============================================================

import { Trainer as BaseTrainer } from "@handfirst/utils";
import { DataLoader } from "./data";
import type { Dataset } from "./data";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";

const BATCH = 40;

// ---- 学习率调度（余弦退火，沿用 v4）----
// 前期 lr 大 → 大步快降；后期 lr 小 → 小步防震荡。
const LR0 = 0.005;   // 初始学习率（调度起点；曲面比 v4 的 sin 更尖锐，扛不住 0.01）
const LR_MIN = 0;    // 最终学习率（调度终点）
const EPOCHS = 6000; // 训练总步数，与 server.ts 的 maxEpochs 一致

function cosineLr(t: number): number {
  const p = Math.min(t / EPOCHS, 1);  // 进度 0→1，超出后钳到 1
  return LR_MIN + (LR0 - LR_MIN) * 0.5 * (1 + Math.cos(Math.PI * p));
}

export class Trainer extends BaseTrainer<{ trainLoss: number; valLoss: number }> {
  readonly model: Sequential;
  private _opt: Adam;
  private _train: DataLoader;
  private _val: DataLoader;

  constructor(train: Dataset, val: Dataset, numNeurons = 16) {
    super();
    this._train = new DataLoader(train, BATCH);
    this._val = new DataLoader(val, val.features.length);
    this.model = new Sequential([new Linear(train.features[0].length, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new Adam(this.model.parameters(), LR0);
  }

  protected _step() {
    this._opt.lr = cosineLr(this.epoch);   // 学习率调度：每个 epoch 重算 lr
    const { X, Y } = this._train.generate();   // 一批随机数据（矩阵）
    const B = X.cols;

    this.model.zeroGrad();
    const yPred = this.model.forward(X);         // [1 × B]
    const diff = yPred.sub(Y);

    // MSE: L = Σ(ŷ - y)² / B
    const trainLoss = diff.dotmul(diff).sum().data[0] / B;

    // 初始梯度 ∂L/∂ŷ = 2(ŷ - y) / B
    this.model.backward(diff.scale(2 / B));
    this._opt.step();

    // 验证集：一次全量评估
    const { X: vX, Y: vY } = this._val.generate();
    const vPred = this.model.forward(vX);
    const vDiff = vPred.sub(vY);
    const valLoss = vDiff.dotmul(vDiff).sum().data[0] / vX.cols;

    return {
      trainLoss: Number(trainLoss.toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
    };
  }
}
