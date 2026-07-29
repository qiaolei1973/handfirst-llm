/** 随机采样一批数据 */
export function sampleBatch(dataset: { features: number[][]; labels: number[] }, size: number): { feature: number[]; label: number }[] {
  const idx = [...Array(dataset.features.length).keys()].sort(() => Math.random() - 0.5).slice(0, size);
  return idx.map(i => ({ feature: dataset.features[i], label: dataset.labels[i] }));
}