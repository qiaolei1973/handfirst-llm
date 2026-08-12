// 终端验证：pnpm exec tsx apps/v5/test.ts
import { surfaceData, prepare } from "./data";
import { Trainer } from "./train";

const { features, labels } = surfaceData(200);
const d = prepare(features, labels);
const t = new Trainer(d.trainF, d.trainL, d.valF, d.valL, d.means, d.stds, 16);

let bestVal = Infinity, patience = 0;
for (let e = 0; e < 2000; e++) {
  const ev = t.step();
  if (ev.valLoss < bestVal) { bestVal = ev.valLoss; patience = 0; }
  else { patience++; }
  if (e % 100 === 0 || patience >= 200) console.log(`epoch ${e+1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}`);
  if (patience >= 200) break;
}
