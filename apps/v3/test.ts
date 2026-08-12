// 终端验证：pnpm exec tsx apps/v3/test.ts
import { sinData } from "./data";
import { Trainer } from "./train";

const { features, labels } = sinData(60);
const t = new Trainer(features, labels, 16);

for (let e = 0; e < 3000; e++) {
  const ev = t.step();
  if (e % 100 === 0 || e === 2999) console.log(`epoch ${e + 1}: loss=${ev.loss.toFixed(6)}`);
}
