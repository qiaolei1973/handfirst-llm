#!/usr/bin/env bash
# ============================================================
# generate-all.sh — 遍历每个 vn 目录，跑 py + d2 生成图片
#
# 结构:
#   scripts/images/
#     generate-all.sh       ← 入口
#     v1/main.py            ← matplotlib
#     v2/main.py            ← matplotlib
#     v3/main.py,arch.py,backprop.py  ← matplotlib
#     v4/main.py            ← matplotlib
#     recap/main.py         ← matplotlib
#     recap/neural-network.d2  ← D2
#
# 产物: apps/_docs/public/<vn>/*.png
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PUBLIC="$ROOT/apps/_docs/public"
SRC_DIR="$ROOT/scripts/images"
FONT="$HOME/.fonts/NotoSansSC-Regular.ttf"

export PATH="$HOME/.local/bin:$PATH"

echo "==> Generate all images..."

shopt -s nullglob

for vn in v1 v2 v3 v4 recap; do
  dir="$SRC_DIR/$vn"
  out="$PUBLIC/$vn"
  mkdir -p "$out"

  # ---- Python / matplotlib ----
  for py in "$dir"/*.py; do
    base="$(basename "$py")"
    echo "  [$vn] py/$base"
    python3 "$py"
  done

  # ---- D2 ----
  for d2file in "$dir"/*.d2; do
    base="$(basename "$d2file" .d2)"
    echo "  [$vn] d2/$base.d2"

    # D2 → SVG
    d2 --font-regular "$FONT" --font-bold "$FONT" "$d2file" "$out/$base.svg"

    # Fix font-family for resvg
    python3 -c "
import re
p='$out/$base.svg'
with open(p) as f: s=f.read()
s=re.sub(r'@font-face\s*\{[^}]*\}','',s)
s=re.sub(r'font-family:\s*\"[^\"]*d2[^\"]*\"',
         'font-family: \"Noto Sans SC Thin\"', s)
with open(p,'w') as f: f.write(s)
"

    # SVG → PNG
    resvg "$out/$base.svg" "$out/$base.png" 2>&1 | grep -v '^Warning' || true
  done
done

shopt -u nullglob

echo ""
echo "==> Done."
