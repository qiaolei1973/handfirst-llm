"""
matplotlib CJK font setup — import after matplotlib.use('Agg') but before pyplot.
Registers Noto Sans SC to render Chinese text in plots.
"""
import matplotlib.font_manager as fm
from pathlib import Path

_font_path = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fm.fontManager.addfont(_font_path)

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False
