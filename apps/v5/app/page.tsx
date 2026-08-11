"use client";

import { useWsTrainer } from "@handfirst/utils";
import { Dashboard } from "./dashboard";

export default function Page() {
  const trainer = useWsTrainer("ws://localhost:3105");

  return <Dashboard trainer={trainer} title="v5: 多维输入 — 权重矩阵 & 3D 曲面" />;
}
