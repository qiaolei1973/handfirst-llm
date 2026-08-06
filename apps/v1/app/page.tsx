'use client';

import { SurgeryDashboard } from '@handfirst/viz';
import { useWsTrainer } from '@handfirst/utils';

const trueFn = (x: number) => 2 * x + 10;

export default function Page() {
  const trainer = useWsTrainer('ws://localhost:3101');

  return (
    <SurgeryDashboard
      trainer={trainer}
      trueFn={trueFn}
      title="猜一条直线"
    />
  );
}
