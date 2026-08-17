/**
 * Matmul animation — 矩阵乘法动画。
 *
 * 用小矩阵演示矩阵乘法的机械操作，讲清「行 × 列 → 逐格相乘求和」：
 *
 *   A[2×3] @ B[3×2] = C[2×2]
 *   C[i][j] = Σ_k A[i][k] · B[k][j]
 *
 * 自洽循环：依次遍历 4 个输出格，每格高亮 A 的第 i 行、B 的第 j 列，
 * 逐步点亮 k=0,1,2 的相乘项并累加，最后停在完整求和上。
 * 不依赖任何训练数据（矩阵乘法规则与训练无关）。
 */
import { COLORS, STYLE } from '../colors';
import { createCanvasManager } from '../canvas';
import type { CanvasManager } from '../canvas';

export interface MatmulAnimHandle {
  draw(): void;
  destroy(): void;
}

const FONT = STYLE.font.family;

const A = [
  [1, 2, 3],
  [4, 5, 6],
]; // 2×3
const B = [
  [7, 8],
  [9, 10],
  [11, 12],
]; // 3×2
const K = A[0].length; // 内积维度 = 3

// 预计算 C = A @ B（完整结果，用于已算完的格子保持显示）
const C = A.map((row, i) =>
  B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)),
);

// 4 个输出格 (i, j)
const CELLS: [number, number][] = [
  [0, 0], [0, 1], [1, 0], [1, 1],
];

const cellOrder = (r: number, c: number) => CELLS.findIndex(([rr, cc]) => rr === r && cc === c);

const CELL = 44;   // 格子边长
const STEP_MS = 900;
const SHOW_MS = 1600;

function drawCell(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  text: string,
  fill: string,
  textColor: string,
): void {
  ctx.fillStyle = fill;
  ctx.fillRect(x, y, CELL, CELL);
  ctx.strokeStyle = COLORS.grayLight;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
  ctx.fillStyle = textColor;
  ctx.font = `600 16px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + CELL / 2, y + CELL / 2);
}

export function createMatmulAnim(canvas: HTMLCanvasElement): MatmulAnimHandle {
  const cm: CanvasManager = createCanvasManager(canvas);

  let cellIdx = 0;
  let k = 0;          // 当前相乘项（0..K-1）
  let done = false;   // 该格已算完，展示完整和
  let lastT = 0;
  let elapsed = 0;
  let raf = 0;

  cm.onResize = () => draw();

  function advance(): void {
    if (!done) {
      k++;
      if (k >= K) done = true;
    } else {
      done = false;
      k = 0;
      cellIdx = (cellIdx + 1) % CELLS.length;
    }
  }

  function draw(): void {
    const ctx = cm.ctx;
    const w = cm.w, h = cm.h;
    ctx.clearRect(0, 0, w, h);

    const [i, j] = CELLS[cellIdx];
    const upto = done ? K - 1 : k; // 已累加的最后一个 k
    const cur = done ? -1 : k;     // 当前高亮项

    let partial = 0;
    const terms: string[] = [];
    for (let t = 0; t <= upto; t++) {
      partial += A[i][t] * B[t][j];
      terms.push(`${A[i][t]}·${B[t][j]}`);
    }

    // 布局：A [2×3]  ×  B [3×2]  =  C [2×2]
    const aCols = A[0].length, aRows = A.length;
    const bCols = B[0].length, bRows = B.length;
    const cCols = bCols, cRows = aRows;

    const aW = aCols * CELL, aH = aRows * CELL;
    const bW = bCols * CELL, bH = bRows * CELL;
    const cW = cCols * CELL, cH = cRows * CELL;

    const totalW = aW + 34 + bW + 34 + cW;
    const x0 = (w - totalW) / 2;
    const yCenter = 150;
    const aX = x0, aY = yCenter - aH / 2;
    const bX = x0 + aW + 34, bY = yCenter - bH / 2;
    const cX = x0 + aW + 34 + bW + 34, cY = yCenter - cH / 2;

    // 标题 + 形状标注
    ctx.fillStyle = COLORS.gray;
    ctx.font = `500 12px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('A', aX + aW / 2, aY - 20);
    ctx.fillText('[2×3]', aX + aW / 2, aY - 8);
    ctx.fillText('B', bX + bW / 2, bY - 20);
    ctx.fillText('[3×2]', bX + bW / 2, bY - 8);
    ctx.fillText('C', cX + cW / 2, cY - 20);
    ctx.fillText('[2×2]', cX + cW / 2, cY - 8);

    // 运算符
    ctx.fillStyle = '#475569';
    ctx.font = `600 26px ${FONT}`;
    ctx.fillText('×', aX + aW + 17, yCenter);
    ctx.fillText('=', bX + bW + 17, yCenter);

    // A 网格：第 i 行浅蓝，当前项 (i,cur) 深蓝
    for (let r = 0; r < aRows; r++) {
      for (let c = 0; c < aCols; c++) {
        const inRow = r === i;
        const isCur = r === i && c === cur;
        const fill = isCur ? COLORS.blue : inRow ? 'rgba(59,130,246,0.22)' : '#ffffff';
        drawCell(ctx, aX + c * CELL, aY + r * CELL, String(A[r][c]), fill, inRow ? COLORS.white : '#334155');
      }
    }

    // B 网格：第 j 列浅琥珀，当前项 (cur,j) 深琥珀
    for (let r = 0; r < bRows; r++) {
      for (let c = 0; c < bCols; c++) {
        const inCol = c === j;
        const isCur = r === cur && c === j;
        const fill = isCur ? COLORS.amber : inCol ? 'rgba(245,158,11,0.22)' : '#ffffff';
        drawCell(ctx, bX + c * CELL, bY + r * CELL, String(B[r][c]), fill, inCol ? COLORS.white : '#334155');
      }
    }

    // C 网格：当前格红色（部分和），已算完的保持显示，其余留空
    for (let r = 0; r < cRows; r++) {
      for (let c = 0; c < cCols; c++) {
        const idx = cellOrder(r, c);
        if (idx === cellIdx) {
          drawCell(ctx, cX + c * CELL, cY + r * CELL, String(partial), COLORS.red, COLORS.white);
        } else if (idx >= 0 && idx < cellIdx) {
          drawCell(ctx, cX + c * CELL, cY + r * CELL, String(C[r][c]), '#f1f5f9', '#334155');
        } else {
          drawCell(ctx, cX + c * CELL, cY + r * CELL, '', '#ffffff', '#334155');
        }
      }
    }

    // 下方公式 + 步骤提示
    const eqY = yCenter + Math.max(bH, aH) / 2 + 34;
    const eq = upto >= K - 1
      ? `C[${i}][${j}] = ${terms.join(' + ')} = ${partial}`
      : `C[${i}][${j}] = ${terms.join(' + ')} + … = ${partial}`;
    ctx.fillStyle = '#0f172a';
    ctx.font = `600 15px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(eq, w / 2, eqY);

    // 步骤提示：明确「同一个格要加 K 次」是 3 项点积，而非重复播放
    const hint = done
      ? `C[${i}][${j}] 完成 ✓`
      : `第 ${k + 1}/${K} 项：A[${i}][${k}] · B[${k}][${j}]`;
    ctx.fillStyle = COLORS.gray;
    ctx.font = `500 13px ${FONT}`;
    ctx.fillText(hint, w / 2, eqY + 24);
  }

  function tick(now: number): void {
    if (lastT === 0) lastT = now;
    elapsed += now - lastT;
    lastT = now;
    const threshold = done ? SHOW_MS : STEP_MS;
    while (elapsed >= threshold) {
      elapsed -= threshold;
      advance();
    }
    draw();
    raf = requestAnimationFrame(tick);
  }

  raf = requestAnimationFrame(tick);

  return {
    draw(): void {
      draw();
    },
    destroy(): void {
      cancelAnimationFrame(raf);
      cm.destroy();
    },
  };
}
