"""汇总校准结果：python -m lanegen.calib_report [目录]"""
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "输出/评测/校准"
rows = []
for tag in sorted(os.listdir(root)):
    f = os.path.join(root, tag, "metrics.csv")
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    n = len(d)
    r = dict(变体=tag, recall80=d.arrow_recall80.mean(), ge=d.stop_lane_ge.mean(), eq=d.stop_lane_eq.mean(),
             cross=(d.crossings > 0).sum() / n, dup=(d.duplicates > 0).sum() / n, frag=d.frag_per_lane.mean(),
             lanes=d.n_lanes.mean(), err=int(d.get("error", pd.Series(dtype=str)).notna().sum()))
    # 综合分：召回类相加，减去错误类（交叉、重复的路口占比，断裂）
    r["综合"] = r["recall80"] + r["ge"] + r["eq"] - r["cross"] - r["dup"] - 0.5 * r["frag"]
    rows.append(r)
df = pd.DataFrame(rows).sort_values("综合", ascending=False).round(3)
print(df.to_string(index=False))
df.to_csv(os.path.join(root, "汇总.csv"), index=False, encoding="utf-8-sig")
