"""批量运行：python -m lanegen.batch"""
import json
import sys
import traceback

from . import run as runner

CASES = [
    ("样例1_珠江新城_隧道上方渠化右转", 113.32263, 23.12359),
    ("样例2_标准十字", 113.32536, 23.12337),
    ("样例3_高架下斜交Y形分叉", 113.28580, 23.12500),
    ("随机1_番禺", 113.37898, 22.84003),
    ("随机2_南沙", 113.49683, 22.78056),
    ("随机3_越秀高架旁", 113.27373, 23.14179),
    ("随机4_天河高架旁", 113.30196, 23.13005),
]

if __name__ == "__main__":
    summary = {}
    for name, lon, lat in CASES:
        try:
            _, rep = runner.main(name, lon, lat, "输出")
            summary[name] = rep
        except Exception as e:
            traceback.print_exc()
            summary[name] = {"error": repr(e)}
    with open("输出/汇总.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    sys.exit(0 if all(r.get("status") == "identified" for r in summary.values()) else 2)
