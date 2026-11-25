"""各版本留出集结果对比：python -m lanegen.versions 基线v1 自适应v6 ..."""
import sys

import pandas as pd

rows = []
for t in sys.argv[1:]:
    d = pd.read_csv(f"输出/评测/{t}/metrics.csv")
    g = lambda k: round(d[k].mean(), 3) if k in d else None
    rows.append(dict(版本=t, 报错=int(d.get("error", pd.Series(dtype=str)).notna().sum()),
                     交叉路口=int((d.crossings > 0).sum()), 重复路口=int((d.duplicates > 0).sum()),
                     方向未定路口=int((d.unknown_dir > 0).sum()), 方向未定总数=int(d.unknown_dir.sum()),
                     断裂=g("frag_per_lane"), 平移稳定=g("shift_stable"),
                     平移稳定P10=round(d.shift_stable.quantile(.1), 3) if "shift_stable" in d else None,
                     recall80=g("arrow_recall80"), ge=g("stop_lane_ge"), eq=g("stop_lane_eq"),
                     车道数=round(d.n_lanes.mean(), 1), 秒=round(d.sec.mean(), 1)))
print(pd.DataFrame(rows).to_string(index=False))
