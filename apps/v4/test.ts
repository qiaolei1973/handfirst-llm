// 终端验证：pnpm exec tsx apps/v4/test.ts
import { sinData, prepare } from "./data";
import { Trainer } from "./train";

const { features, labels } = sinData(60);
const d = prepare(features, labels);
const t = new Trainer(d.trainF, d.trainL, d.valF, d.valL, 16);

let bestVal = Infinity, patience = 0;
for (let e = 0; e < 3000; e++) {
  const ev = t.step();
  if (ev.valLoss < bestVal) { bestVal = ev.valLoss; patience = 0; }
  else { patience++; }
  if (e % 100 === 0 || e === 2999 || patience >= 300) console.log(`epoch ${e+1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}`);
  if (patience >= 300) break;
}
