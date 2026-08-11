import { linearData } from './data';


let W: number = 1;
let bias: number = 0;
let gradW: number = 0;
let gradBias: number = 0;

/** 学习率 */
const learnRate = 0.1;
const MAX_EPOCHS = 600;

const linear = (x: number, W: number, bias: number) => {
  return x * W + bias;
}


const train = () => {
  const dataset = linearData(12, 20); // 生成训练数据

  for (let epoch = 0; epoch <= MAX_EPOCHS; epoch++) {

    gradW = 0;
    gradBias = 0;

    const size = dataset.features.length;

    for (let i = 0; i < size; i++) {
      const x = dataset.features[i];
      // 目标值
      const target = dataset.labels[i];
      // 计算预测值
      const yPred = linear(x, W, bias);
      // 计算预测值与真实值的差
      const diff = yPred - target;
      // MAE 损失:    L = (1/n) Σ |yPred - y|
      const sign = diff > 0 ? 1 : diff < 0 ? -1 : 0;
      // W 单次梯度 sign(yPred - y)·x
      gradW += sign * x;
      // bias 单次梯度 sign(yPred - y)
      gradBias += sign;
    }

    // 取平均得到真实梯度
    // W 的梯度:    ∂L/∂W    = (1/n) Σ sign(yPred - y)·x
    gradW /= size;
    // bias 的梯度: ∂L/∂bias = (1/n) Σ sign(yPred - y)
    gradBias /= size;

    // 参数更新（梯度下降）
    W -= learnRate * gradW;
    bias -= learnRate * gradBias;
  }
}


train();
console.log(`训练完成，W: ${W}, bias: ${bias}`);