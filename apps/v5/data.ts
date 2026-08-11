/**
 * v5 数据预处理：多维训练/验证分离 + 每维度独立标准化（Z-score）。
 */

/** 80/20 随机划分（保持 feature-label 配对） */
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

/** 多维 Z-score 统计量（每维度独立） */
export function standardizeMulti(
  data: number[][],
  dim: number,
): { means: number[]; stds: number[] } {
  const means = Array(dim).fill(0);
  const stds = Array(dim).fill(1);
  const n = data.length;
  for (let d = 0; d < dim; d++) {
    means[d] = data.reduce((s, f) => s + f[d], 0) / n;
    const v = data.reduce((s, f) => s + (f[d] - means[d]) ** 2, 0) / n;
    stds[d] = Math.sqrt(v) || 1;
  }
  return { means, stds };
}
