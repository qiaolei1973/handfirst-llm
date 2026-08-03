/**
 * 线性回归 demo 数据。
 * y = 2x + 10 ± 3 均匀噪声，返回带有真实函数引用的 dataset。
 */
export function linearData(size = 12, maxX = 13.2): {
  features: number[];
  labels: number[];
  trueFn: (x: number) => number;
} {
  const trueFn = (x: number) => 2 * x + 10;
  const features = Array.from({ length: size }, (_, i) => i * (maxX / (size - 1 || 1)));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 6);
  return { features, labels, trueFn };
}

/**
 * Random sample without replacement from a {features, labels} dataset.
 */
export function sampleBatch(
  dataset: { features: number[]; labels: number[] },
  size: number,
): { feature: number; label: number }[] {
  const indices = [...Array(dataset.features.length).keys()]
    .sort(() => Math.random() - 0.5)
    .slice(0, size);
  return indices.map((i) => ({
    feature: dataset.features[i],
    label: dataset.labels[i],
  }));
}
