"""
matplotlib CJK font setup — import after matplotlib.use('Agg') but before pyplot.

The Noto Sans SC font at ~/.fonts registers internally as "Noto Sans SC Thin".
We provide a `font_prop` function that returns a FontProperties pointing directly
to the TTF file, bypassing rcParams font resolution issues.
"""
import matplotlib.font_manager as fm

_font_path = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fm.fontManager.addfont(_font_path)
fm._load_fontmanager(try_read_cache=False)

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["axes.unicode_minus"] = False

# Use this directly in set_title/set_xlabel calls: fontproperties=cjk_fp
cjk_fp = FontProperties(fname=_font_path)
