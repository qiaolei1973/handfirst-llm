/**
 * Matrix
 */

export class Mat {
  data: Float64Array;
  rows: number;
  cols: number;

  constructor(rows: number, cols: number, data?: Float64Array) {
    this.rows = rows;
    this.cols = cols;
    this.data = data || new Float64Array(rows * cols);
  }

  /** 创建全零矩阵 */
  static zeros(rows: number, cols: number): Mat {
    return new Mat(rows, cols);
  }

  /**
   * 将 number / number[] / number[][] 统一转为 Mat。
   *
   * @example
   * Mat.from(5)              → (1,1) [[5]]
   * Mat.from([1, 2, 3])     → (1,3) [[1,2,3]]
   * Mat.from([[1],[2]])     → (2,1) [[1],[2]]
   * Mat.from([[1,2],[3,4]]) → (2,2) [[1,2],[3,4]]
   */
  static from(data: number | number[] | number[][]): Mat {
    if (typeof data === 'number') return new Mat(1, 1, new Float64Array([data]));
    if (Array.isArray(data[0])) {
      const rows = data.length;
      const cols = (data as number[][])[0].length;
      const m = new Mat(rows, cols);
      for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
          m.data[i * cols + j] = (data as number[][])[i][j];
      return m;
    }
    return new Mat(1, (data as number[]).length, new Float64Array(data as number[]));
  }

  /**
   * 广播 + 逐元素二元运算。
   *
   * 先对齐形状（广播），再逐个元素调用 op。
   *
   * @param other - 另一个矩阵，形状满足广播条件
   * @param op    - 二元运算函数 (a, b) => result
   *               其中 a 来自 this，b 来自 other
   *
   * 广播规则（从右往左比 shape）:
   *   (rows, cols) + (rows, cols) → 同形逐元素
   *   (rows, cols) + (1, cols)    → 行广播（每行加相同向量）
   *   (rows, cols) + (rows, 1)    → 列广播（每列加相同向量）
   *   (rows, cols) + (1, 1)       → 标量广播（所有元素加相同数）
   *
   * @example
   * _broadcastOp(Mat.from([10,20]), (a,b) => a+b)  → add
   * _broadcastOp(Mat.from([10,20]), (a,b) => a-b)  → sub
   * _broadcastOp(Mat.from([10,20]), (a,b) => a*b)  → mul
   */
  private _broadcastOp(other: Mat, op: (a: number, b: number) => number): Mat {
    // 1. 确定目标形状（两矩阵的行/列各自取最大）
    const rows = Math.max(this.rows, other.rows);
    const cols = Math.max(this.cols, other.cols);

    // 2. 检查是否可广播：每个维度上，两个形状要么相等，要么有一个是 1
    const aOk = (this.rows === rows || this.rows === 1) && (this.cols === cols || this.cols === 1);
    const bOk = (other.rows === rows || other.rows === 1) && (other.cols === cols || other.cols === 1);
    if (!aOk || !bOk) {
      throw new Error(`Cannot broadcast shapes (${this.rows}, ${this.cols}) and (${other.rows}, ${other.cols})`);
    }

    const result = new Mat(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const av = this.data[(i % this.rows) * this.cols + (j % this.cols)];
        const bv = other.data[(i % other.rows) * other.cols + (j % other.cols)];
        result.data[i * cols + j] = op(av, bv);
      }
    }

    return result;
  }

  /**
   * 重塑形状。不改变数据，只重新解释行列数。
   *
   * @example
   * Mat.from([[1,2,3],[4,5,6]]).reshape(3, 2)
   *   → [[1, 2], [3, 4], [5, 6]]
   *
   */
  reshape(rows: number, cols: number): Mat {
    if (rows * cols !== this.rows * this.cols)
      throw new Error(`reshape: 元素总数 ${this.rows * this.cols} 不匹配 ${rows * cols}`);
    return new Mat(rows, cols, this.data);
  }


  /** 矩阵加法 */
  add(other: Mat): Mat {
    return this._broadcastOp(other, (a, b) => a + b);
  }

  /** 矩阵加法（就地加法） */
  add_(other: Mat): void {
    const res = this._broadcastOp(other, (a, b) => a + b);
    this.data.set(res.data);
  }

  /** 矩阵减法 */
  sub(other: Mat): Mat {
    return this._broadcastOp(other, (a, b) => a - b);
  }

  /** 
   * 逐元素乘法（哈达玛积），（对应位置直接乘） 
   * dropout：mask = [1, 0, 0, 1, 1, 0, 1, ...] -> mat.mul(mask) 掩码（mask）
   * weights = [0.1, 0.7, 0.2] -> mat.mul(weights) 加权（weight）
   * scale = 0.5 -> mat.mul(scale) 缩放（scale）
  */
  dotmul(other: Mat): Mat {
    return this._broadcastOp(other, (a, b) => a * b);
  }

  scale(s: number): Mat {
    return this.dotmul(Mat.from([s]));
  }

  /** 逐元素变换，返回新矩阵 */
  map(fn: (v: number) => number): Mat {
    const result = new Mat(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = fn(this.data[i]);
    }
    return result;
  }

  /** 
   * 矩阵乘法：this @ other（形状 m×n 乘 n×p → m×p）
   * 信息融合
   **/
  matmul(other: Mat): Mat {
    const rows = this.rows, inner = this.cols, cols = other.cols;
    const result = new Mat(rows, cols);
    const a = this.data, b = other.data, c = result.data;

    for (let i = 0; i < rows; i++) {
      const aOffset = i * inner, cOffset = i * cols;
      for (let k = 0; k < inner; k++) {
        const av = a[aOffset + k];
        if (av === 0) continue; // skip zero for efficiency
        const bOffset = k * cols;
        for (let j = 0; j < cols; j++) {
          // result[i][j] = Σ_k  A[i][k]  ×  B[k][j] (k=0..n-1)
          c[cOffset + j] += av * b[bOffset + j];
        }
      }
    }

    return result;
  }

  /** 转置矩阵 m×n → n×m */
  transpose(): Mat {
    const result = new Mat(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j * this.rows + i] = this.data[i * this.cols + j];
      }
    }
    return result;
  }

  /**
   * @param dim - 可选。不传时返回 (1,1) 标量矩阵，
   *              0 表示列求和（返回 1×cols），
   *              1 表示行求和（返回 rows×1）
   *
   * @example
   * Mat.from([[1,2],[3,4]]).sum()    → [[10]]
   * Mat.from([[1,2],[3,4]]).sum(0)   → [[4, 6]]
   * Mat.from([[1,2],[3,4]]).sum(1)   → [[3], [7]]
   */
  sum(dim?: number): Mat {
    if (dim === 0) {
      const R = new Mat(1, this.cols);
      for (let i = 0; i < this.rows; i++)
        for (let j = 0; j < this.cols; j++)
          R.data[j] += this.data[i * this.cols + j];
      return R;
    }
    if (dim === 1) {
      const R = new Mat(this.rows, 1);
      for (let i = 0; i < this.rows; i++) {
        let s = 0;
        for (let j = 0; j < this.cols; j++) s += this.data[i * this.cols + j];
        R.data[i] = s;
      }
      return R;
    }
    // 无 dim: 返回 (1,1) 标量矩阵
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return new Mat(1, 1, new Float64Array([s]));
  }
}