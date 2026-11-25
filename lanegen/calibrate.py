"""在校准集上网格搜索（不碰留出集）：python -m lanegen.calibrate"""
import itertools
import json
import os
import subprocess
import sys
import time

GRID = dict(w_min_k=[0.65, 0.75], w_max_k=[1.2, 1.35, 1.5], clear_out_k=[1.5, 2.0, 3.0])
CASES = "data/评测/校准集.json"
PAR = 3

if __name__ == "__main__":
    py = sys.executable
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    todo = []
    for c in combos:
        tag = "_".join(f"{k}{v}" for k, v in c.items())
        if not os.path.exists(os.path.join("输出/评测/校准", tag, "summary.json")):
            todo.append((tag, c))
    running = []
    while todo or running:
        while todo and len(running) < PAR:
            tag, c = todo.pop(0)
            env = dict(os.environ, LANEGEN_F=json.dumps(c))
            code = f"from lanegen import evaluate; evaluate.main('校准/{tag}', shift=False, cases_path='{CASES}')"
            running.append((tag, subprocess.Popen([py, "-B", "-W", "ignore", "-c", code], env=env,
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))
        time.sleep(2)
        for t, p in list(running):
            if p.poll() is not None:
                print(t, "done", p.returncode, flush=True)
                running.remove((t, p))
