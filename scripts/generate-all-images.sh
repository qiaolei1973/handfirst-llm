#!/usr/bin/env bash
# ============================================================
# generate-all-images.sh
# 一键生成所有文档图片（Python + D2）
#
# 源文件（git 追踪）→ 产物 PNG/SVG（git 忽略）
#
# 依赖: python3, d2, resvg, NotoSansSC-Regular.ttf
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
DIAGRAMS="$ROOT/diagrams"
OUT_DIAGRAMS="$DIAGRAMS/out"
PUBLIC="$ROOT/apps/_docs/public"

# Ensure PATH includes common user-local install dirs
export PATH="$HOME/.local/bin:$PATH"

FONT="$HOME/.fonts/NotoSansSC-Regular.ttf"
RESVG="resvg"

echo "==> Generate all images..."

# ---- Python / matplotlib ----
echo ""
echo "--- matplotlib ---"
python3 "$SCRIPTS/gen_v1_images.py"
python3 "$SCRIPTS/gen_v2_images.py"
python3 "$SCRIPTS/gen_v3_images.py"
python3 "$SCRIPTS/gen_v3_arch.py"
python3 "$SCRIPTS/gen_v3_backprop.py"
python3 "$SCRIPTS/gen_v4_images.py"
python3 "$SCRIPTS/gen_recap_images.py"

# ---- D2 diagrams ----
echo ""
echo "--- d2 ---"
mkdir -p "$OUT_DIAGRAMS"

for d2file in "$DIAGRAMS"/*.d2; do
  base="$(basename "$d2file" .d2)"
  svg="$OUT_DIAGRAMS/$base.svg"
  png="$OUT_DIAGRAMS/$base.png"

  echo "  $base ..."

  # D2 → SVG
  d2 \
    --font-regular "$FONT" \
    --font-bold "$FONT" \
    --font-italic "$FONT" \
    "$d2file" "$svg"

  # Fix font-family for resvg (D2 embeds custom names)
  python3 -c "
import re
with open('$svg','r') as f: s=f.read()
s=re.sub(r'@font-face\s*\{[^}]*\}','',s)
s=re.sub(r'font-family:\s*\"[^\"]*d2[^\"]*\"',
         'font-family: \"Noto Sans SC Thin\"', s)
with open('$svg','w') as f: f.write(s)
"

  # SVG → PNG
  $RESVG "$svg" "$png" 2>&1 | grep -v '^Warning' || true

  # Copy to public for doc referencing
  cp "$png" "$PUBLIC/recap/$base.png"
done

echo ""
echo "==> Done."
