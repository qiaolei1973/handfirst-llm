import { sinData } from "./data";
import { Trainer } from "./train";
import { Linear } from "./nn/linear";

const { features, labels, trueFn } = sinData(60);

function dumpModel(t: Trainer) {
  const [hw, ow] = t.model.layers.filter(l => l instanceof Linear) as Linear[];
  return { numNeurons: hw.outDim, hiddenW: Array.from(hw.weight.data), hiddenB: Array.from(hw.bias.data), outputW: Array.from(ow.weight.data), outputB: ow.bias.data[0] };
}

Trainer.server({
  port: 3103,
  maxEpochs: 3000,
  features,
  labels,
  trueFnLabel: "y = sin(x)",
  factory: () => {
    const t = new Trainer(features, labels, 16);
    const opt = (t as any)._opt;
    return {
      step: () => ({ ...t.step(), params: dumpModel(t) }),
      reset: () => { t.history.length = 0; t.model.resetParameters(); opt.reset(); },
      history: t.history, params: dumpModel(t),
    };
  },
});
