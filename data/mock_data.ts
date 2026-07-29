/**
 * 
 * @param size 
 * @returns features（特征）
 * @returns labels（标签）
 */
export const mockData = (size: number) => {
  // ===== 准备数据：features（特征矩阵）+ labels（标签向量）=====
  const features: number[][] = [];
  const labels: number[] = [];

  for (let a = 0; a <= size; a++)
    for (let b = 0; b <= size; b++) {
      features.push([a, b]);   // 特征: [加数1, 加数2]
      labels.push(a + b);    // 标签: 真实和
    }

  return { features, labels };
}