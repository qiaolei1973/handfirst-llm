/**
 * Generate mock linear regression data: y = 2x + 10 with ±3 uniform noise.
 */
export function mockData(size: number): { features: number[]; labels: number[] } {
  const features: number[] = [];
  const labels: number[] = [];

  for (let i = 0; i < size; i++) {
    features.push(i);
    labels.push(2 * i + 10 + (Math.random() - 0.5) * 2 * 3);
  }

  return { features, labels };
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
