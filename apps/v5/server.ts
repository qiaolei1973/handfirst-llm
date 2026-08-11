import { surfaceData, prepare } from "./data";
import { Trainer } from "./train";

const { features, labels } = surfaceData(200);
const d = prepare(features, labels);

Trainer.server({
  port: 3105,
  maxEpochs: 2000,
  features,
  labels,
  trueFnLabel: "f(x1,x2) = sin(sqrt(x1^2+x2^2) * 2pi)",
  factory: () => new Trainer(d.trainF, d.trainL, d.valF, d.valL, d.means, d.stds, 16),
});
