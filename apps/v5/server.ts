import { surfaceData, prepare } from "./data";
import { Trainer } from "./train";

const { features, labels } = surfaceData(200);
const d = prepare(features, labels);

Trainer.server({
  port: 3105,
  maxEpochs: 6000,
  features,
  labels,
  trueFnLabel: "f(x1,x2) = sin(sqrt(x1^2+x2^2) * 2pi)",
  factory: () => {
    const t = new Trainer(d.train, d.val, 16);
    return {
      step: () => t.step(),
      get epoch() { return t.epoch; },
    };
  },
});
