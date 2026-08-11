// 终端验证：pnpm exec tsx apps/v5/test.ts
import { fileURLToPath } from "node:url";
import { surfaceData } from "@handfirst/datasets";
import { Trainer } from "./train";

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels } = surfaceData(200);
  const t = new Trainer(features, labels, 16);
  for (let e = 0; e < 2000; e++) {
    const ev = t.step();
    if (e % 100 === 0 || ev.stopped) console.log(`epoch ${e+1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}`);
    if (ev.stopped) break;
  }
}
