"""命令行：python -m lanegen.run <名称> <经度> <纬度> [输出根目录]"""
import os
import sys

from . import pipeline, render


def main(name, lon, lat, out_root):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    print(f"[{name}] {lon},{lat}")
    res = pipeline.run(lon, lat)
    rep = pipeline.export(res, out)
    cx, cy = res["core"].centroid.x, res["core"].centroid.y
    title = (f"{name}  {rep['status']}  车道 {rep['n_lanes']}  新增 {rep['lane_add']}  减少 {rep['lane_drop']}  "
             f"渠化右转候选 {rep['slip']}  方向未定 {rep['unknown_direction']}")
    render.draw(res, os.path.join(out, "叠加_全貌.png"), title=title)
    render.draw(res, os.path.join(out, "叠加_路口.png"), extent=(cx - 70, cx + 70, cy - 70, cy + 70), title=title)
    render.draw_model(res, os.path.join(out, "debug_step2.png"))
    print("  ", rep)
    return res, rep


if __name__ == "__main__":
    name, lon, lat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    root = sys.argv[4] if len(sys.argv) > 4 else "输出"
    _, report = main(name, lon, lat, root)
    sys.exit(0 if report["status"] == "identified" else 2)
