/** 非线性数据 — y = sin(x) + noise, x ∈ [0,1] 对应 [0, 2π] */
export function sinData(size = 60) {
  const xScale = 2 * Math.PI;
  const trueFn = (x: number) => Math.sin(x * xScale);
  const features = Array.from({ length: size }, (_, i) => i / (size - 1 || 1));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 0.3);
  return { features, labels, trueFn, xScale };
}

/** 随机采样 batch（无放回） */
export function sampleBatch(
  features: number[], labels: number[], size: number,
): { feature: number; label: number }[] {
  const n = features.length;
  const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, Math.min(size, n));
  return indices.map((i) => ({ feature: features[i], label: labels[i] }));
}
