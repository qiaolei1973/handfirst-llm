import { surfaceData, prepare } from "./data";
import { Trainer } from "./train";
import { Linear } from "./nn/linear";

const { features, labels } = surfaceData(200);
const d = prepare(features, labels);

function dumpModel(t: Trainer) {
  const [hw, ow] = t.model.layers.filter(l => l instanceof Linear) as Linear[];
  return { inputDim: hw.inDim, numNeurons: hw.outDim, hiddenW: Array.from(hw.weight.data), hiddenB: Array.from(hw.bias.data), outputW: Array.from(ow.weight.data), outputB: ow.bias.data[0] };
}

Trainer.server({
  port: 3105,
  maxEpochs: 2000,
  features,
  labels,
  trueFnLabel: "f(x1,x2) = sin(sqrt(x1^2+x2^2) * 2pi)",
  factory: () => {
    const t = new Trainer(d.train, d.val, 16);
    return {
      step: () => ({ ...t.step(), params: dumpModel(t) }),
      history: t.history, params: dumpModel(t),
    };
  },
});
