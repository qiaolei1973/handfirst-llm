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

const BATCH = 40, LR = 0.005;

export class Trainer extends BaseTrainer {
  readonly model: Sequential;
  private _opt: Adam;
  private _train: DataLoader;
  private _val: DataLoader;

  constructor(train: Dataset, val: Dataset, numNeurons = 16) {
    super();
    this._train = new DataLoader(train, BATCH);
    this._val = new DataLoader(val, val.features.length);
    this.model = new Sequential([new Linear(train.features[0].length, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new Adam(this.model.parameters(), LR);
  }

  step() {
    const { X, Y } = this._train.nextBatch();   // X: [d × B], Y: [1 × B]
    const B = X.cols;

    this.model.zeroGrad();
    const yPred = this.model.forward(X);         // [1 × B]

    // MSE: L = Σ(ŷ - y)² / B
    const diff = yPred.sub(Y);
    const trainLoss = diff.dotmul(diff).sum().data[0] / B;

    // 初始梯度 ∂L/∂ŷ = 2(ŷ - y) / B
    this.model.backward(diff.scale(2 / B));
    this._opt.step();

    // 验证集：一次 forward，全部样本
    const { X: vX, Y: vY } = this._val.nextBatch();
    const vPred = this.model.forward(vX);
    const vDiff = vPred.sub(vY);
    const valLoss = vDiff.dotmul(vDiff).sum().data[0] / vX.cols;

    return {
      trainLoss: Number(trainLoss.toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
    };
  }
}
