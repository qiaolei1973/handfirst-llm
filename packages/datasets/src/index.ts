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
 * 非线性 demo 数据 — y = sin(x) + noise。
 * x 已归一化到 [0, 1]（对应原始范围 [0, 2π]）。
 * 线性模型 y=Wx+b 无法拟合，需要引入非线性（ReLU / 隐藏层）。
 */
export function sinData(size = 60): {
  features: number[];
  labels: number[];
  trueFn: (xNorm: number) => number;
  /** 归一化因子：原始 x = xNorm * xScale */
  xScale: number;
} {
  const xScale = 2 * Math.PI;
  const trueFn = (xNorm: number) => Math.sin(xNorm * xScale);
  const features = Array.from(
    { length: size },
    (_, i) => i / (size - 1 || 1),
  );
  const labels = features.map(
    (xNorm) => trueFn(xNorm) + (Math.random() - 0.5) * 0.3,
  );
  return { features, labels, trueFn, xScale };
}

/**
 * 2D 曲面数据 — f(x₁, x₂) = sin(√(x₁²+x₂²)) + noise（中心辐射涟漪）。
 * x₁, x₂ 已归一化到 [0, 1]（对应原始范围 [-2, 2]）。
 */
export function surfaceData(size = 200): {
  features: [number, number][];
  labels: number[];
  trueFn: (x1: number, x2: number) => number;
  scale: number;
} {
  const scale = 2;
  const trueFn = (x1: number, x2: number) => {
    const r = Math.sqrt(x1 * x1 + x2 * x2);
    return Math.sin(r * scale * Math.PI);
  };
  const features: [number, number][] = [];
  const labels: number[] = [];
  for (let i = 0; i < size; i++) {
    const x1 = 2 * Math.random() - 1; // [-1, 1]
    const x2 = 2 * Math.random() - 1; // [-1, 1]
    features.push([x1, x2]);
    const noise = (Math.random() - 0.5) * 0.3;
    labels.push(trueFn(x1, x2) + noise);
  }
  return { features, labels, trueFn, scale };
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
