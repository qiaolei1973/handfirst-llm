import { surfaceData, prepare } from "./data";
import { Trainer } from "./train";
import { Linear } from "./nn/linear";

const { features, labels } = surfaceData(200);
const d = prepare(features, labels);

function dumpModel(t: Trainer) {
  const [hw, ow] = t.model.layers.filter(l => l instanceof Linear) as Linear[];
  return { inputDim: hw.inDim, numNeurons: hw.outDim, hiddenW: Array.from(hw.w), hiddenB: Array.from(hw.b), outputW: Array.from(ow.w), outputB: ow.b[0] };
}

Trainer.server({
  port: 3105,
  maxEpochs: 2000,
  features,
  labels,
  trueFnLabel: "f(x1,x2) = sin(sqrt(x1^2+x2^2) * 2pi)",
  factory: () => {
    const t = new Trainer(d.trainF, d.trainL, d.valF, d.valL, 16);
    return { step: () => ({ ...t.step(), params: dumpModel(t) }), reset: t.reset.bind(t), history: t.history, params: dumpModel(t) };
  },
});
