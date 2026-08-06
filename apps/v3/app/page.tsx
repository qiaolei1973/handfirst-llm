'use client';

import { useWsTrainer } from '@handfirst/utils';
import { Dashboard } from './dashboard';

const trueFn = (x: number) => Math.sin(x);

export default function Page() {
  const trainer = useWsTrainer('ws://localhost:3103');

  return (
    <Dashboard trainer={trainer} trueFn={trueFn} title="画曲线" />
  );
}
