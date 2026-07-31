export interface Padding {
  t: number; // top
  r: number; // right
  b: number; // bottom
  l: number; // left
}

export interface Coord {
  toX: (v: number) => number;
  toY: (v: number) => number;
}

export interface Point {
  x: number;
  y: number;
}

export interface LineSeries {
  label: string;
  color: string;
  points: Point[];
}
