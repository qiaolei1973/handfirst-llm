/**
 * Float64Array ↔ plain array serialization helpers.
 *
 * Trainers store parameters as Float64Array internally,
 * but the WS protocol sends them as JSON number[][][].
 * These helpers reduce the boilerplate of mapping between the two.
 */

/** Float64Array → number[] */
export function arr(data: Float64Array): number[] {
  return Array.from(data);
}

/** number[] → Float64Array (in-place) */
export function setArr(dst: Float64Array, src: number[]): void {
  for (let i = 0; i < src.length; i++) dst[i] = src[i];
}

/** Float64Array (flat row-major storage) → number[][] */
export function mat(data: Float64Array, rows: number, cols: number): number[][] {
  const out: number[][] = [];
  for (let j = 0; j < rows; j++) {
    out.push(Array.from(data.subarray(j * cols, j * cols + cols)));
  }
  return out;
}

/** number[][] → Float64Array (flat row-major, in-place) */
export function setMat(dst: Float64Array, src: number[][], cols: number): void {
  for (let j = 0; j < src.length; j++) {
    for (let i = 0; i < cols; i++) {
      dst[j * cols + i] = src[j][i];
    }
  }
}
