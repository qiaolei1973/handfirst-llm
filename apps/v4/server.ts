import { sinData, prepare } from "./data";
import { Trainer } from "./train";
import { Linear } from "./nn/linear";

const { features, labels, trueFn } = sinData(60);
const d = prepare(features, labels);

function dumpModel(t: Trainer) {
  const [hw, ow] = t.model.layers.filter(l => l instanceof Linear) as Linear[];
  return { numNeurons: hw.outDim, hiddenW: Array.from(hw.w), hiddenB: Array.from(hw.b), outputW: Array.from(ow.w), outputB: ow.b[0] };
}

Trainer.server({
  port: 3104,
  maxEpochs: 3000,
  features,
  labels,
  trueFnLabel: "y = sin(x)",
  factory: () => {
    const t = new Trainer(d.trainF, d.trainL, d.valF, d.valL, 16);
    return { step: () => ({ ...t.step(), params: dumpModel(t) }), reset: t.reset.bind(t), history: t.history, params: dumpModel(t) };
  },
});
