#!/usr/bin/env python3
"""Generate step variants from the master backprop-steps.excalidraw.

Reads the master file and produces step-0 (full) through step-5 variants.
Elements tagged with the current step's groupId stay at opacity 100;
all others drop to opacity 15.
"""

import json
import copy
import sys
from pathlib import Path

STEPS = ["step-1", "step-2", "step-3", "step-4", "step-5"]
DIM_OPACITY = 15

TITLES = {
    "step-0": "全貌",
    "step-1": "① 前向：x → z → h → ŷ → L",
    "step-2": "② 反向：L → ∂L/∂ŷ = 2(ŷ−y)",
    "step-3": "③ 反向：∂L/∂h = ∂L/∂ŷ · w_out",
    "step-4": "④ 反向：∂L/∂z = ∂L/∂h · ReLU'(z)",
    "step-5": "⑤ 反向：∂L/∂W = ∂L/∂z · x",
}


def set_step_title(elements, text):
    for el in elements:
        if el.get("id") == "step-title":
            el["text"] = text
            return


def main():
    master_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/qingquan/handfirst-llm/apps/_docs/scripts/v3/backprop-steps.excalidraw"
    )
    out_dir = master_path.parent

    master = json.loads(master_path.read_text())

    # Step 0: full view — all elements at 100%
    step0 = copy.deepcopy(master)
    for el in step0["elements"]:
        el["opacity"] = 100
    set_step_title(step0["elements"], TITLES["step-0"])
    (out_dir / "backprop-step0.excalidraw").write_text(
        json.dumps(step0, indent=2, ensure_ascii=False)
    )
    print(f"  ✓ backprop-step0.excalidraw (full view)")

    # Steps 1–5: dim everything except current step
    for step_id in STEPS:
        variant = copy.deepcopy(master)
        for el in variant["elements"]:
            if "all" in el.get("groupIds", []):
                el["opacity"] = 100
            elif step_id in el.get("groupIds", []):
                el["opacity"] = 100
            else:
                el["opacity"] = DIM_OPACITY
        set_step_title(variant["elements"], TITLES[step_id])
        fn = f"backprop-{step_id}.excalidraw"
        (out_dir / fn).write_text(
            json.dumps(variant, indent=2, ensure_ascii=False)
        )
        print(f"  ✓ {fn}")

    print("Done.")


if __name__ == "__main__":
    main()
