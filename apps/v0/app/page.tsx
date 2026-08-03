'use client';

import { useMemo } from 'react';
import { SurgeryDashboard } from '@handfirst/viz';
import { linearData } from '@handfirst/datasets';
import { Trainer } from '@/lib/train';
import { mse, lossCoeffs } from '@handfirst/utils';

const { features, labels, trueFn } = linearData(12);

export default function Page() {
  const trainer = useMemo(
    () => new Trainer({ features, labels }),
    [],
  );

  return (
    <SurgeryDashboard
      trainer={trainer}
      trueFn={trueFn}
      dataset={{ features, labels }}
      mse={mse}
      lossCoeffs={lossCoeffs}
    />
  );
}
