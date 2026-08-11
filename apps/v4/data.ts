/**
 * v4 数据预处理：训练/验证分离 + 标准化（Z-score）。
 */

/** 80/20 随机划分 */
export function trainValSplit<T>(
  data: T[],
): { train: T[]; val: T[] } {
  const n = data.length;
  const nTrain = Math.floor(n * 0.8);
  const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
  const trainIdx = new Set(indices.slice(0, nTrain));
  const train: T[] = [], val: T[] = [];
  for (let i = 0; i < n; i++) {
    (trainIdx.has(i) ? train : val).push(data[i]);
  }
  return { train, val };
}

/** 单维标准化统计量 */
export function standardize1D(vals: number[]): { mean: number; std: number } {
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
  return { mean, std: Math.sqrt(variance) || 1 };
}
