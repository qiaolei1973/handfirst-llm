/** 2D 曲面数据 — f(x₁,x₂) = sin(√(x₁²+x₂²) · 2π) + noise */
export function surfaceData(size = 200) {
  const scale = 2 * Math.PI;
  const trueFn = (x1: number, x2: number) => Math.sin(Math.sqrt(x1 * x1 + x2 * x2) * scale);
  const features: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < size; i++) {
    const x1 = 2 * Math.random() - 1;
    const x2 = 2 * Math.random() - 1;
    features.push([x1, x2]);
    labels.push(trueFn(x1, x2) + (Math.random() - 0.5) * 0.3);
  }
  return { features, labels, trueFn, scale };
}

/** 随机采样 batch（无放回）— v5 接受 number[][] 特征 */
export function sampleBatch(
  features: number[][], labels: number[], size: number,
): { feature: number[]; label: number }[] {
  const n = features.length;
  const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, Math.min(size, n));
  return indices.map((i) => ({ feature: features[i], label: labels[i] }));
}

function trainValSplit<T>(data: T[]) {
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

function standardizeMulti(data: number[][], dim: number) {
  const n = data.length;
  const means = Array(dim).fill(0), stds = Array(dim).fill(1);
  for (let d = 0; d < dim; d++) {
    means[d] = data.reduce((s, f) => s + f[d], 0) / n;
    const v = data.reduce((s, f) => s + (f[d] - means[d]) ** 2, 0) / n;
    stds[d] = Math.sqrt(v) || 1;
  }
  return { means, stds };
}

/** 完整数据准备 */
export function prepare(
  features: number[][], labels: number[],
): {
  trainF: number[][]; trainL: number[]; valF: number[][]; valL: number[];
  means: number[]; stds: number[];
} {
  const dim = features[0].length;
  const fi = trainValSplit(features.map((f, i) => ({ f, l: labels[i] })));
  let trainF = fi.train.map(d => d.f), trainL = fi.train.map(d => d.l);
  let valF = fi.val.map(d => d.f), valL = fi.val.map(d => d.l);

  const { means, stds } = standardizeMulti(trainF, dim);
  trainF = trainF.map(f => f.map((v, i) => (v - means[i]) / stds[i]));
  valF = valF.map(f => f.map((v, i) => (v - means[i]) / stds[i]));

  return { trainF, trainL, valF, valL };
}
