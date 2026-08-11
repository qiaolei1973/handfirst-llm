// 终端验证：pnpm exec tsx apps/v3/test.ts
import { fileURLToPath } from "node:url";
import { sinData } from "@handfirst/datasets";
import { Trainer } from "./train";

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = sinData(60);
  const t = new Trainer({ features, labels }, 16);
  for (let e = 0; e < 3000; e++) {
    const ev = t.step();
    if (e % 100 === 0 || e === 2999) console.log(`epoch ${e+1}: loss=${ev.loss.toFixed(6)}`);
  }
  console.log("\n预测 vs 真实值:");
  for (const x of [0, 0.25, 0.5, 0.75])
    console.log(`  x̂=${x.toFixed(2)}  predict=${t.predict(x).toFixed(4)}  true=${trueFn(x).toFixed(4)}`);
}
