export { COLORS } from './colors';
export { niceTicks, drawGrid, drawAxes, drawLegend, resizeCanvas } from './primitives';
export { createAxes } from './axes';
export type { AxesHandle, AxesConfig, Layer } from './axes';
export { createCanvasManager } from './canvas';
export type { CanvasManager } from './canvas';

// biz charts
export { createLineChart } from './biz/line-chart';
export { createLossLandscape } from './biz/loss-landscape';
export { createModelFit } from './biz/model-fit';
export { createModelCurve } from './biz/model-curve';
export { createReLUActivations } from './biz/relu-activations';
export type { LineChartOpts, LineChartHandle } from './biz/line-chart';
export type { LossLandscapeOpts, LossLandscapeConfig, LossLandscapeHandle } from './biz/loss-landscape';
export type { ModelFitConfig, ModelFitHandle } from './biz/model-fit';
export type { ModelCurveConfig, ModelCurveHandle } from './biz/model-curve';
export type { ReLUActivationsConfig, ReLUActivationsHandle } from './biz/relu-activations';

export type { Padding, Coord, Point, LineSeries } from './types';
