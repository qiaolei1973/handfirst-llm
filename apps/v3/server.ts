import { sinData } from './data';
import { Trainer } from './train';

const { features, labels, trueFn } = sinData(60);

Trainer.server({
  port: 3103,
  maxEpochs: 3000,
  features,
  labels,
  trueFnLabel: 'y = sin(x)',
  factory: () => new Trainer(features, labels, 16),
});
