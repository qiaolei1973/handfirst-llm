import { surfaceData } from "@handfirst/datasets";
import { Trainer } from "./train";

const { features, labels } = surfaceData(200);

Trainer.server({
  port: 3105,
  maxEpochs: 2000,
  features,
  labels,
  trueFnLabel: "f(x1,x2) = sin(sqrt(x1^2+x2^2) * 2pi)",
  factory: () => new Trainer(features, labels, 16),
});
