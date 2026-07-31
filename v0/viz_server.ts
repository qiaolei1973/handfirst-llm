import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import { exec } from "node:child_process";
import { mockData, sampleBatch } from "./mock_data";

const PORT = 3000;

// 服务器启动时只生成一次数据，多次刷新不会变
const TRAINING_DATA = mockData(15);

// ===== 训练逻辑（异步生成器，每个 epoch yield 一次）=====
async function* trainGenerator() {
  let W = 1;
  let bias = 0;
  const learnRate = 0.01;
  const { features: rawX, labels } = TRAINING_DATA;

  // 归一化：x' = (x - mean) / std，让 feature 围绕 0 对称
  const n = rawX.length;
  const mean = rawX.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(rawX.reduce((s, x) => s + (x - mean) ** 2, 0) / n);
  const xNorm = rawX.map((x) => (x - mean) / std);

  // 原始 x 用于可视化拟合线
  const rawXs = Array.from({ length: 15 }, (_, i) => i);

  // 训练用归一化 feature，但每个 epoch 把参数和梯度转回原始空间再 yield
  for (let epoch = 0; epoch <= 1200; epoch++) {
    const batch = sampleBatch({ features: xNorm, labels }, 8);
    let gradW_norm = 0;
    let gradBias_norm = 0;
    let totalLoss = 0;

    for (const { feature, label } of batch) {
      const yPred = feature * W + bias;
      const diff = yPred - label;
      totalLoss += diff * diff;
      const gradLoss = 2 * diff;
      gradW_norm += feature * gradLoss;
      gradBias_norm += gradLoss;
    }

    gradW_norm /= batch.length;
    gradBias_norm /= batch.length;
    const avgLoss = totalLoss / batch.length;

    W -= learnRate * gradW_norm;
    bias -= learnRate * gradBias_norm;

    // 转换到原始空间:  y = W_raw × x + bias_raw
    const W_raw = W / std;
    const bias_raw = bias - W * mean / std;

    // 梯度也转回原始空间
    const gradW_raw = gradW_norm * std;
    const gradBias_raw = gradBias_norm;

    // 拟合线（原始空间）
    const lineInterval = epoch <= 30 ? 2 : 10;
    const lineData =
      epoch % lineInterval === 0
        ? { xs: rawXs, ys: rawXs.map((x) => W_raw * x + bias_raw) }
        : undefined;

    yield {
      epoch,
      loss: avgLoss,
      W: W_raw,
      bias: bias_raw,
      gradW: gradW_raw,
      gradBias: gradBias_raw,
      lineData,
    };

    // 延迟让训练过程可观察
    await new Promise((r) => setTimeout(r, 50));
  }
}

// ===== SSE 辅助 =====
function sseHeader(res: ServerResponse) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": "*",
  });
}

function sseEvent(
  res: ServerResponse,
  event: string,
  data: unknown
) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

// ===== HTML 仪表盘 =====
function getHtml(): string {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧠 v0 训练可视化</title>
<script src="/chart.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #f8fafc;
    color: #1e293b;
    min-height: 100vh;
    padding: 24px;
  }
  .header {
    text-align: center;
    margin-bottom: 20px;
  }
  .header h1 { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
  .header .subtitle { font-size: 0.875rem; color: #64748b; margin-top: 4px; }

  /* 状态栏 */
  .status-bar {
    display: flex;
    justify-content: center;
    gap: 32px;
    flex-wrap: wrap;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .stat-item { text-align: center; }
  .stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
  .stat-value { font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat-value.epoch { color: #6366f1; }
  .stat-value.loss  { color: #ef4444; }
  .stat-value.w     { color: #3b82f6; }
  .stat-value.bias  { color: #f59e0b; }

  /* 图表网格 */
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }
  .chart-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .chart-card h2 {
    font-size: 0.875rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 8px;
  }
  .chart-card canvas { width: 540px; height: 280px; max-width: 100%; }

  /* 响应式 */
  @media (max-width: 768px) {
    .chart-grid { grid-template-columns: 1fr; }
    .status-bar { gap: 16px; }
  }

  /* 加载遮罩 */
  .loading-overlay {
    position: fixed; inset: 0;
    background: rgba(248,250,252,0.85);
    display: flex; align-items: center; justify-content: center;
    z-index: 10;
    transition: opacity 0.4s;
  }
  .loading-overlay.hidden { opacity: 0; pointer-events: none; }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid #e2e8f0;
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="loading-overlay" id="loading">
  <div class="spinner"></div>
</div>

<div class="header">
  <h1>🧠 线性回归训练可视化</h1>
  <p class="subtitle">模型 y = W·x + bias &nbsp;|&nbsp; 优化器 SGD</p>
</div>

<div class="status-bar">
  <div class="stat-item">
    <div class="stat-label">Epoch</div>
    <div class="stat-value epoch" id="statEpoch">0 / 1200</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Loss (MSE)</div>
    <div class="stat-value loss" id="statLoss">—</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">W</div>
    <div class="stat-value w" id="statW">1.000</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">Bias</div>
    <div class="stat-value bias" id="statBias">0.000</div>
  </div>
</div>

<div class="chart-grid">
  <div class="chart-card">
    <h2>📉 Loss 曲线</h2>
    <canvas id="chartLoss"></canvas>
  </div>
  <div class="chart-card">
    <h2>📈 参数收敛</h2>
    <canvas id="chartParams"></canvas>
  </div>
  <div class="chart-card">
    <h2>📊 梯度衰减</h2>
    <canvas id="chartGrad"></canvas>
  </div>
  <div class="chart-card">
    <h2>🎯 拟合直线 vs 真实数据</h2>
    <canvas id="chartFit"></canvas>
  </div>
</div>

<script>
// ===== 图表颜色 =====
const BLUE   = '#3b82f6';
const AMBER  = '#f59e0b';
const RED    = '#ef4444';
const PURPLE = '#6366f1';
const GREEN  = '#22c55e';
const GRAY   = '#94a3b8';
const GRAY_LIGHT = '#cbd5e1';

// ===== 工具函数 =====
function makeChart(canvasId, config) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    ...config,
    options: {
      ...config.options,
      responsive: false,          // 避免持续 update 时无限撑高
      maintainAspectRatio: false,
      animation: false,           // 全关动画，数据太快跟不上
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: {
          labels: {
            usePointStyle: true,
            pointStyleWidth: 8,
            padding: 20,
            font: { size: 11 },
          },
        },
        ...(config.options?.plugins || {}),
      },
      scales: {
        x: {
          type: 'linear',         // 强制数值轴
          grid: { color: '#f1f5f9' },
          ticks: { font: { size: 10 }, color: '#94a3b8' },
          ...config.options?.scales?.x,
        },
        y: {
          grid: { color: '#f1f5f9' },
          ticks: { font: { size: 10 }, color: '#94a3b8' },
          ...config.options?.scales?.y,
        },
        ...Object.fromEntries(
          Object.entries(config.options?.scales || {}).filter(([k]) => k !== 'x' && k !== 'y')
        ),
      },
    },
  });
}

// ===== Chart 1: Loss 曲线 =====
const chartLoss = makeChart('chartLoss', {
  type: 'line',
  data: {
    datasets: [{
      label: 'MSE Loss',
      data: [],
      borderColor: RED,
      backgroundColor: RED + '18',
      borderWidth: 2,
      pointRadius: 0,
      fill: true,
      tension: 0.3,
    }],
  },
  options: {
    scales: {
      x: { min: 0, max: 1200, title: { display: true, text: 'Epoch', color: GRAY, font: { size: 11 } } },
      y: { title: { display: true, text: 'MSE', color: GRAY, font: { size: 11 } }, beginAtZero: true },
    },
  },
});

// ===== Chart 2: 参数收敛 =====
const chartParams = makeChart('chartParams', {
  type: 'line',
  data: {
    datasets: [
      {
        label: 'W',
        data: [],
        borderColor: BLUE,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
      },
      {
        label: 'bias',
        data: [],
        borderColor: AMBER,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
      },
    ],
  },
  options: {
    scales: {
      x: { min: 0, max: 1200, title: { display: true, text: 'Epoch', color: GRAY, font: { size: 11 } } },
      y: { title: { display: true, text: '值', color: GRAY, font: { size: 11 } } },
    },
  },
});

// ===== Chart 3: 梯度衰减 =====
const chartGrad = makeChart('chartGrad', {
  type: 'line',
  data: {
    datasets: [
      {
        label: '|gradW|',
        data: [],
        borderColor: BLUE,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: 'y',
      },
      {
        label: '|gradBias|',
        data: [],
        borderColor: AMBER,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: 'y1',
      },
    ],
  },
  options: {
    scales: {
      x: { min: 0, max: 1200, title: { display: true, text: 'Epoch', color: GRAY, font: { size: 11 } } },
      y: {
        type: 'linear',
        position: 'left',
        title: { display: true, text: '|gradW|', color: BLUE, font: { size: 11 } },
        ticks: { color: BLUE },
      },
      y1: {
        type: 'linear',
        position: 'right',
        title: { display: true, text: '|gradBias|', color: AMBER, font: { size: 11 } },
        ticks: { color: AMBER },
        grid: { drawOnChartArea: false },
      },
    },
  },
});

// ===== Chart 4: 拟合 =====
let trueScatterData = [];
const chartFit = makeChart('chartFit', {
  type: 'scatter',
  data: {
    datasets: [
      {
        label: '训练数据',
        data: [],
        backgroundColor: GRAY,
        borderColor: GRAY,
        pointRadius: 5,
        pointHoverRadius: 7,
      },
      {
        label: '模型拟合线',
        data: [],
        borderColor: PURPLE,
        borderWidth: 2.5,
        pointRadius: 0,
        showLine: true,
        tension: 0,
      },
    ],
  },
  options: {
    scales: {
      x: { title: { display: true, text: 'x', color: GRAY, font: { size: 11 } } },
      y: { title: { display: true, text: 'y', color: GRAY, font: { size: 11 } } },
    },
  },
});

// ===== SSE 连接 =====
const evtSource = new EventSource('/events');

evtSource.addEventListener('init', (e) => {
  const { features, labels } = JSON.parse(e.data);

  // 填充真实数据散点
  const scatterPoints = features.map((x, i) => ({ x, y: labels[i] }));
  chartFit.data.datasets[0].data = scatterPoints;

  // 立即隐藏 loading，让用户看到实时训练
  document.getElementById('loading').classList.add('hidden');

  chartFit.update();
});

evtSource.addEventListener('epoch', (e) => {
  const d = JSON.parse(e.data);

  // Chart 1: Loss
  chartLoss.data.datasets[0].data.push({ x: d.epoch, y: d.loss });

  // Chart 2: Params
  chartParams.data.datasets[0].data.push({ x: d.epoch, y: d.W });
  chartParams.data.datasets[1].data.push({ x: d.epoch, y: d.bias });

  // Chart 3: Gradients
  chartGrad.data.datasets[0].data.push({ x: d.epoch, y: Math.abs(d.gradW) });
  chartGrad.data.datasets[1].data.push({ x: d.epoch, y: Math.abs(d.gradBias) });

  // Chart 4: 拟合线更新
  if (d.lineData) {
    const linePoints = d.lineData.xs.map((x, i) => ({ x, y: d.lineData.ys[i] }));
    chartFit.data.datasets[1].data = linePoints;
  }

  // 前 30 个 epoch 每个都刷（带动画），之后每 5 个 epoch 批量刷
  const throttle = d.epoch <= 30 ? 1 : 5;
  if (d.epoch % throttle === 0) {
    chartLoss.update('none');
    chartParams.update('none');
    chartGrad.update('none');
    chartFit.update('none');
  }

  // 状态栏
  document.getElementById('statEpoch').textContent = d.epoch + ' / 1200';
  document.getElementById('statLoss').textContent = d.loss.toFixed(4);
  document.getElementById('statW').textContent = d.W.toFixed(3);
  document.getElementById('statBias').textContent = d.bias.toFixed(3);
});

evtSource.addEventListener('done', (e) => {
  const d = JSON.parse(e.data);

  // 最终全量更新
  chartLoss.update();
  chartParams.update();
  chartGrad.update();
  chartFit.update();

  document.getElementById('statEpoch').textContent = '1200 / 1200 ✅';
  document.getElementById('statLoss').textContent = d.finalLoss.toFixed(6);
  document.getElementById('statW').textContent = d.finalW.toFixed(4);
  document.getElementById('statBias').textContent = d.finalBias.toFixed(4);

  evtSource.close();
});

evtSource.onerror = () => {
  document.getElementById('loading').classList.add('hidden');
  console.warn('SSE 连接中断（训练可能已完成）');
};
</script>
</body>
</html>`;
}

// ===== HTTP 服务器 =====
const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  const url = req.url || "/";

  if (url === "/" || url === "/index.html") {
    // 首页 —— 返回仪表盘 HTML
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(getHtml());
  } else if (url === "/events") {
    // SSE 端点 —— 流式推送训练指标
    sseHeader(res);

    // 先发送初始数据（真实数据点用于散点图）
    const { features, labels } = TRAINING_DATA;
    sseEvent(res, "init", { features, labels });

    // 跑训练，捕获最终的参数值
    let lastEvent: { W: number; bias: number; loss: number } | null = null;
    try {
      for await (const ev of trainGenerator()) {
        if (res.destroyed) break;
        lastEvent = ev;
        sseEvent(res, "epoch", ev);
      }
    } catch (err) {
      console.error("训练出错:", err);
    }

    // 发送完成事件，带上最终参数
    if (lastEvent) {
      sseEvent(res, "done", {
        finalW: lastEvent.W,
        finalBias: lastEvent.bias,
        finalLoss: lastEvent.loss,
      });
    }

    res.end();
  } else if (url === "/chart.js") {
    res.writeHead(200, { "Content-Type": "application/javascript" });
    res.end(readFileSync("node_modules/chart.js/dist/chart.umd.js", "utf-8"));
  } else if (url === "/surgery") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(readFileSync("surgery.html", "utf-8"));
  } else if (url === "/norm-math") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(readFileSync("normalization_math.html", "utf-8"));
  } else if (url === "/norm") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(readFileSync("normalization_viz.html", "utf-8"));
  } else {
    res.writeHead(404);
    res.end("Not Found");
  }
});

server.listen(PORT, () => {
  const url = `http://localhost:${PORT}`;
  console.log(`\n  🧠 训练可视化仪表盘已启动\n`);
  console.log(`  👉 浏览器访问:  ${url}\n`);
  console.log(`  按 Ctrl+C 停止服务器\n`);

  // 自动打开浏览器
  const cmd = process.platform === "darwin" ? "open" : process.platform === "win32" ? "start" : "xdg-open";
  exec(`${cmd} ${url}`, (err) => {
    if (err) console.log("  (未能自动打开浏览器，请手动访问)\n");
  });
});
