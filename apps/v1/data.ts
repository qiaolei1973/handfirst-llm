/**
 * v1 mock data — y = 2x + 10 ± noise
 */

export function linearData(size = 12, maxX = 13.2) {
  const trueFn = (x: number) => 2 * x + 10;
  const features = Array.from({ length: size }, (_, i) => i * (maxX / (size - 1 || 1)));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 6);
  return { features, labels, trueFn };
}
