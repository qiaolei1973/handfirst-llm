// 和 v1 完全一样的线性数据
export function linearData(size = 12, maxX = 13.2) {
  const trueFn = (x: number) => 2 * x + 10;
  const features = Array.from({ length: size }, (_, i) => i * (maxX / (size - 1 || 1)));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 6);
  return { features, labels, trueFn };
}

/** 随机采样 batch（无放回） */
export function sampleBatch(
  features: number[], labels: number[], size: number,
): { feature: number; label: number }[] {
  const n = features.length;
  const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, Math.min(size, n));
  return indices.map((i) => ({ feature: features[i], label: labels[i] }));
}
