import { sampleBatch, mockData } from "./mock_data";

let W: number = 1;
let bias: number = 0;
let gradW: number = 0;
let gradBias: number = 0;

/** 学习率 */
const learnRate = 0.01;

const linear = (x: number, W: number, bias: number) => {
  return x * W + bias;
}


const train = () => {
  const { features, labels } = mockData(10); // 生成训练数据

  for (let epoch = 0; epoch <= 30; epoch++) {
    const batch = sampleBatch({ features, labels }, 10);

    gradW = 0;
    gradBias = 0;

    for (const { feature, label } of batch) {
      const yPred = linear(feature, W, bias);
      const diff = yPred - label;
      const loss = diff * diff; // MSE loss
      const gradLoss = 2 * diff; // loss 对输出的偏导数

      gradW += feature * gradLoss;
      gradBias += gradLoss;
    }
    W -= learnRate * gradW;
    bias -= learnRate * gradBias;
  }
}


train();
console.log(`训练完成，W: ${W}, bias: ${bias}`);