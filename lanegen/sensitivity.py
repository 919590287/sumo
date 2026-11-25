"""参数敏感性：每个无量纲系数分别乘 0.8 / 1.2，在子集上重跑评测。
python -m lanegen.sensitivity"""
import json
import os
import subprocess
import sys

from .params import F

BASE = dict(F)
KEYS = list(BASE.keys())

if __name__ == "__main__":
    py = sys.executable
    runs = [("base", {})] + [(f"{k}x{m}", {k: BASE[k] * m}) for k in KEYS for m in (0.8, 1.2)]
    for tag, ov in runs:
        out = os.path.join("输出/评测/敏感性", tag)
        if os.path.exists(os.path.join(out, "summary.json")):
            continue
        env = dict(os.environ, LANEGEN_F=json.dumps(ov))
        code = ("import sys; from lanegen import evaluate; "
                f"evaluate.main('敏感性/{tag}', shift=False, cases_path='data/评测/敏感性子集.json')")
        subprocess.run([py, "-B", "-W", "ignore", "-c", code], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(tag, "done", flush=True)
