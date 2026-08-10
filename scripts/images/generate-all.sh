#!/usr/bin/env bash
# ============================================================
# generate-all.sh — 遍历每个 vn 目录，跑 py + d2 生成图片
#
# 缓存策略：产物已存在 && 源文件没变 → 跳过
# 强制重生成: GENERATE_ALL_FORCE=1 bash generate-all.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PUBLIC="$ROOT/apps/_docs/public"
SRC_DIR="$ROOT/scripts/images"
FONT="$HOME/.fonts/NotoSansSC-Regular.ttf"

export PATH="$HOME/.local/bin:$PATH"
FORCE="${GENERATE_ALL_FORCE:-0}"

echo "==> Generate images..."

shopt -s nullglob

for vn in v1 v2 v3 v4 recap; do
  dir="$SRC_DIR/$vn"
  out="$PUBLIC/$vn"
  mkdir -p "$out"

  # ---- Python / matplotlib ----
  for py in "$dir"/*.py; do
    # 源文件改过时间作为 checksum
    base="$(basename "$py" .py)"
    [[ "$base" == "precompute" ]] && continue
    src_ts=$(stat -c %Y "$py")
    # 找这个脚本对应的输出（第一个生成的 PNG）
    first_out=$(ls "$out"/*.png 2>/dev/null | head -1)

    need_run=1
    if [[ "$FORCE" -ne 1 && -n "$first_out" ]]; then
      out_ts=$(stat -c %Y "$first_out" 2>/dev/null || echo 0)
      if [[ "$src_ts" -le "$out_ts" ]]; then
        need_run=0
      fi
    fi

    if [[ "$need_run" -eq 0 ]]; then
      echo "  [$vn] py/$base  (cached)"
      continue
    fi

    echo "  [$vn] py/$base"
    python3 "$py" > /dev/null 2>&1
  done

  # ---- D2 ----
  for d2file in "$dir"/*.d2; do
    base="$(basename "$d2file" .d2)"
    png="$out/$base.png"
    src_ts=$(stat -c %Y "$d2file")

    if [[ "$FORCE" -ne 1 && -f "$png" ]]; then
      out_ts=$(stat -c %Y "$png")
      if [[ "$src_ts" -le "$out_ts" ]]; then
        echo "  [$vn] d2/$base.d2  (cached)"
        continue
      fi
    fi

    echo "  [$vn] d2/$base.d2"

    d2 --font-regular "$FONT" --font-bold "$FONT" "$d2file" "$out/$base.svg" > /dev/null 2>&1

    python3 -c "
import re
p='$out/$base.svg'
with open(p) as f: s=f.read()
s=re.sub(r'@font-face\s*\{[^}]*\}','',s)
s=re.sub(r'font-family:\s*\"[^\"]*d2[^\"]*\"',
         'font-family: \"Noto Sans SC Thin\"', s)
with open(p,'w') as f: f.write(s)
"

    resvg "$out/$base.svg" "$png" 2>&1 | grep -v '^Warning' || true
  done
done

echo "==> Done."
