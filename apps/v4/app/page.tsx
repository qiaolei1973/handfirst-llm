'use client';

import { useWsTrainer } from '@handfirst/utils';
import { Dashboard } from './dashboard';

const trueFn = (x: number) => Math.sin(x);

export default function Page() {
  const trainer = useWsTrainer('ws://localhost:3104');

  return (
    <Dashboard trainer={trainer} trueFn={trueFn} title="优化曲线" />
  );
}
