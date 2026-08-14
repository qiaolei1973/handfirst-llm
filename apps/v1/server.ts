import { linearData } from './data';
import { Trainer } from './train';

const { features, labels, trueFn } = linearData(12, 20);

Trainer.server({
  port: 3101,
  maxEpochs: 600,
  features, labels,
  trueFnLabel: `y=2x+10`,
  factory: () => new Trainer({ features, labels }),
});
