'use client';

import { SurgeryDashboard } from '@handfirst/viz';
import { useWsTrainer } from '@handfirst/utils';

// 均值中心化后 x 偏移了 -10，trueFn 需要匹配中心化空间
// y = 2*(x+10) + 10 = 2x + 30
const trueFn = (x: number) => 2 * x + 30;

export default function Page() {
  const trainer = useWsTrainer('ws://localhost:3102');

  return (
    <SurgeryDashboard
      trainer={trainer}
      trueFn={trueFn}
    />
  );
}
