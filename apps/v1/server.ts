import { linearData } from '@handfirst/datasets';
import { Trainer } from './train_class';

const { features, labels, trueFn } = linearData(12, 20);

Trainer.server({
  maxEpochs: 600,
  features, labels,
  trueFnLabel: `y=2x+10`,
  factory: () => new Trainer({ features, labels }),
});
