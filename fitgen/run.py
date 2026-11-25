"""拟合式路口建模原型：识别 → 拟合叠图 → SUMO 成网 → 仿真（单车遍历 + 持续车流）。

python -m fitgen.run <输出目录> <名称>=<经度>,<纬度> ...
"""
import json
import os
import sys
import time

import numpy as np

from lanegen import roi, boundaries, sumo_check
from lanegen.junction import arrows_of
from lanegen.trial import stress
from . import legs as LG, fit as FT, net as NT, render as RD


def run_one(name, lon, lat, root):
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    T = {}
    t = time.time()
    raw, crs = roi.read_roi(lon, lat, 200)
    r = roi.zfilter(raw, roi.estimate_z0(raw["lines"], (0, 0)))
    B = boundaries.build(r)
    W = LG.walls_of(B)
    T["读取"] = time.time() - t
    t = time.time()
    legs, G = LG.find_legs(B["drivable"], W, (0.0, 0.0))
    c = LG.refine_center(legs, (0.0, 0.0))
    legs, G = LG.find_legs(B["drivable"], W, tuple(c))
    T["路臂(测地过滤)"] = time.time() - t
    t = time.time()
    ev = FT.Evidence(B, arrows_of(r["main"]), W)
    fits = FT.fit_all(legs, ev, G, c)
    T["横断面拟合"] = time.time() - t
    t = time.time()
    netd = NT.build(fits, c, G, ev)
    sd = os.path.join(out, "sumo")
    ok, warns, err = NT.compile_net(netd, sd)
    n_rr = 0
    for _ in range(2):
        if not ok:
            break
        k = NT.reroute_blocked(netd, sd, ev, G)
        if not k:
            break
        n_rr += k
        ok, warns, err = NT.compile_net(netd, sd)
    T["SUMO 编译"] = time.time() - t
    sim, normal, internal, bad, bad_len, bad_norm = {}, [], [], [], 0.0, 0.0
    if ok:
        t = time.time()
        sim["单车遍历"] = sumo_check.simulate(sd, netd)
        sim["持续车流"] = stress(sd, netd)
        T["仿真"] = time.time() - t
        normal, internal = NT.net_lines(os.path.join(sd, "net.net.xml"))
        bad, bad_len = NT.obstacle_hits(internal, ev)
        bad2, bad_norm = NT.obstacle_hits(normal, ev)
        bad += bad2
    t = time.time()
    n_in = sum(int(f["in"]["n"][f["in"]["i0"]]) for f in fits)
    n_out = sum(int(f["out"]["n"][f["out"]["i0"]]) for f in fits)
    title = f"{name}  车行道 {len(fits)}  口门进口车道 {n_in}  出口车道 {n_out}  渠化右转 {netd['slips']}"
    RD.draw(ev, fits, c, os.path.join(out, "拟合与成网.png"), title=title, sumo_lines=(normal, internal) if ok else None,
            bad=bad)
    T["出图"] = time.time() - t
    roads = []
    for f in fits:
        q = f["quality"]
        roads.append(dict(id=f["id"], mode=f["mode"], mode_src=f.get("mode_src"), median=f["med_src"],
                          approach_lanes=int(f["in"]["n"][f["in"]["i0"]]), exit_lanes=int(f["out"]["n"][f["out"]["i0"]]),
                          lane_w_in=float(f["in"]["w"][f["in"]["i0"]]), lane_w_out=float(f["out"]["w"][f["out"]["i0"]]),
                          lane_moves=f.get("lane_mv_final", []), moves_src=f.get("mv_src"),
                          stop_line=bool(f["stop"]), divider_support=q["divider_support"],
                          unexplained_marks=q["unexplained_marks"], leftover_width_p50=q["leftover_p50"],
                          inferred=f["inferred"]))
    summary = dict(name=name, lon=lon, lat=lat, center_local=[round(float(c[0]), 1), round(float(c[1]), 1)],
                   roads=roads, slips=netd["slips"], rerouted_connections=n_rr, notes=netd["notes"],
                   sumo=dict(net_ok=ok, netconvert_warnings=len(warns), internal_in_island_m=round(bad_len, 1),
                             normal_in_island_m=round(bad_norm, 1)),
                   sim=sim, seconds={k: round(v, 1) for k, v in T.items()}, total_s=round(sum(T.values()), 1))
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(sd, "netconvert_warnings.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(warns) + ("\n" + err if not ok else ""))
    return summary


if __name__ == "__main__":
    root = sys.argv[1]
    for arg in sys.argv[2:]:
        name, ll = arg.split("=")
        lon, lat = map(float, ll.split(","))
        s = run_one(name, lon, lat, root)
        print(json.dumps({k: s[k] for k in ("name", "slips", "rerouted_connections", "notes", "sumo", "sim", "total_s")}, ensure_ascii=False,
                         default=float), flush=True)
