import { linearData } from './data';
import { Trainer, normalize } from './train';

const ds = normalize(linearData(12, 20));

Trainer.server({
  port: 3102,
  maxEpochs: 600,
  features: ds.features,
  labels: ds.labels,
  trueFnLabel: `y=2x+10`,
  factory: () => new Trainer({ features: ds.features, labels: ds.labels }),
});
