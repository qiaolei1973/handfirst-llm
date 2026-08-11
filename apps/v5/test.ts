// 终端验证：pnpm exec tsx apps/v5/test.ts
import { fileURLToPath } from "node:url";
import { surfaceData, prepare } from "./data";
import { Trainer } from "./train";

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels } = surfaceData(200);
  const d = prepare(features, labels);
  const t = new Trainer(d.trainF, d.trainL, d.valF, d.valL, d.means, d.stds, 16);
  for (let e = 0; e < 2000; e++) {
    const ev = t.step();
    if (e % 100 === 0 || ev.stopped) console.log(`epoch ${e+1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}`);
    if (ev.stopped) break;
  }
}
