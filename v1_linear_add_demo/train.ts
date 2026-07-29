import { Mat } from "../utils/mat";
import { mockData } from "../data/mock_data";
import { SGDOptimizer } from "../utils/optimizer";
import { Module, LinearLayer } from "./model";
import { sampleBatch } from "../utils/dataLoader";

class LinearRegressionModel extends Module {
  fc: LinearLayer
  constructor() {
    super();
    this.fc = new LinearLayer(2, 1);
  }

  forward(X: Mat) {
    return this.fc.forward(X);
  }

  backward(dY: Mat) {
    return this.fc.backward(dY);
  }

  getAllParams(): { data: Mat; grad: Mat }[] {
    return this.getAllParams();
  }
}

export const train = () => {
  const { features, labels } = mockData(10); // 生成训练数据

  const model = new LinearRegressionModel();
  const optimizer = new SGDOptimizer(model.getAllParams(), 0.01);

  for (let epoch = 0; epoch <= 30; epoch++) {
    const batch = sampleBatch({ features, labels }, 10);

    for (const { feature, label } of batch) {
      // 梯度清零
      optimizer.zeroGrad();

      const xMat = Mat.from(feature);
      const yMat = Mat.from(label);

      // 前向传播
      const yPred = model.forward(xMat);
      const diff = yPred.sub(yMat);
      // MSE loss = (1/n) * sum(diff^2)
      const loss = diff.dotmul(diff).sum();

      model.backward(diff.scale(2)); // 反向传播

      optimizer.step(); // 更新参数
    }
  }
}