import { sinData, prepare } from './data';
import { Trainer } from './train';

const { features, labels, trueFn } = sinData(60);
const d = prepare(features, labels);

Trainer.server({
  port: 3104,
  maxEpochs: 3000,
  features,
  labels,
  trueFnLabel: 'y = sin(x)',
  factory: () => new Trainer(d.trainF, d.trainL, d.valF, d.valL, d.mean, d.std, 16),
});
