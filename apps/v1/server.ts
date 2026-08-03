import { linearData } from '@handfirst/datasets';
import { Trainer } from './train';

const { features, labels } = linearData(12);

Trainer.server({
  port: 3004,
  maxEpochs: 600,
  features,
  labels,
  trueFnLabel: 'y=2x+10',
  factory: () => new Trainer({ features, labels }),
});
