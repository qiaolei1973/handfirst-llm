// ---- palette ----

export const COLORS = {
  // semantic
  blue:   '#3b82f6',   // W 参数
  amber:  '#f59e0b',   // bias 参数
  red:    '#ef4444',   // loss / 当前点 / 误差
  purple: '#6366f1',   // 模型线
  green:  '#22c55e',   // 真实函数 / 谷底

  // structural
  gray:       '#94a3b8',   // 数据点 / 刻度标签 / 默认
  grayLight:  '#cbd5e1',   // 坐标轴描边
  grayBg:     '#f1f5f9',   // 网格线
  white:      '#ffffff',   // 散点描边

  // tooltip
  tooltipBg:  'rgba(255,255,255,0.92)',
  tooltipText:'#64748b',

  // crosshair
  crosshair:  'rgba(100,116,139,0.45)',

  // 真实函数（半透明）
  trueFn:     'rgba(34,197,94,0.45)',
  trueFnLabel:'rgba(34,197,95,0.75)',

  // 误差线（半透明）
  errorLine:  'rgba(239,68,68,0.35)',

  // 轨迹（LossLandscape）
  trajectory:       'rgba(59,130,246,0.25)',
  trajectoryDot:    'rgba(59,130,246,0.35)',

  // 谷底（半透明）
  valley:     'rgba(34,197,94,0.45)',
  valleyLabel:'rgba(34,197,95,0.7)',
} as const;

// ---- style constants (matplotlib-like rcParams) ----

export const STYLE = {
  font: {
    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
    tick: 8,
    axisLabel: 10,
    annotation: 9,
  },
  spine: {
    lineWidth: 1,
    topRight: 'rgba(203,213,225,0.45)',
  },
  legend: {
    fontSize: 9,
    padding: 8,
    swatchSize: 11,
    gap: 6,
  },
} as const;
