import { Mat } from "../utils/mat";

/**
 * 提供一个符合 y = 2x + 1 线性关系的 mock 数据集
 * @param size 
 * @returns features（特征）
 * @returns labels（标签）
 */
export const mockData = (size: number) => {
  // ===== 准备数据：features（特征矩阵）+ labels（标签向量）=====
  const features: number[] = [];
  const labels: number[] = [];

  for (let i = 0; i <= size; i++) {
    features.push(i);   // 特征
    labels.push(2 * i + 1);    // 标签
  }
  return { features, labels };
}


/** 随机采样一批数据 */
export function sampleBatch(dataset: { features: number[]; labels: number[] }, size: number): { feature: number; label: number }[] {
  const idx = [...Array(dataset.features.length).keys()].sort(() => Math.random() - 0.5).slice(0, size);
  return idx.map(i => ({ feature: dataset.features[i], label: dataset.labels[i] }));
}