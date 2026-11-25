"""汇总敏感性：python -m lanegen.sens_report"""
import os

import pandas as pd

KEYS = ["arrow_recall80", "stop_lane_eq", "stop_lane_ge", "crossings", "duplicates", "frag_per_lane", "n_lanes"]
root = "输出/评测/敏感性"
base = pd.read_csv(os.path.join(root, "base", "metrics.csv"))
b = {k: base[k].astype(float).mean() for k in KEYS}
rows = []
for tag in sorted(os.listdir(root)):
    f = os.path.join(root, tag, "metrics.csv")
    if tag == "base" or not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    r = {"变体": tag}
    for k in KEYS:
        r[k] = round(d[k].astype(float).mean() - b[k], 3)
    rows.append(r)
df = pd.DataFrame(rows)
print("基准：", {k: round(v, 3) for k, v in b.items()})
print("各变体相对基准的变化（均值差）：")
print(df.to_string(index=False))
df.to_csv(os.path.join(root, "汇总.csv"), index=False, encoding="utf-8-sig")
