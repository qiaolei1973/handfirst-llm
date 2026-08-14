import { sinData, prepare } from "./data";
import { Trainer } from "./train";
import { Linear } from "./nn/linear";

const { features, labels, trueFn } = sinData(60);
const d = prepare(features, labels);

function dumpModel(t: Trainer) {
  const [hw, ow] = t.model.layers.filter(l => l instanceof Linear) as Linear[];
  return { numNeurons: hw.outDim, hiddenW: Array.from(hw.weight.data), hiddenB: Array.from(hw.bias.data), outputW: Array.from(ow.weight.data), outputB: ow.bias.data[0] };
}

Trainer.server({
  port: 3104,
  maxEpochs: 3000,
  features,
  labels,
  trueFnLabel: "y = sin(x)",
  factory: () => {
    const t = new Trainer(d.train, d.val, 16);
    return {
      step: () => ({ ...t.step(), params: dumpModel(t) }),
      get epoch() { return t.epoch; }, params: dumpModel(t),
    };
  },
});
