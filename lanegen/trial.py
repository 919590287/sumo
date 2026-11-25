"""四类路口试跑：识别 → 叠图 → SUMO 成网 → 仿真（单车遍历 + 持续车流压力），分步计时。

python -m lanegen.trial <输出目录> <名称>=<经度>,<纬度> ...
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from . import pipeline, render, sumo_check, evaluate
from .turns import cpsat_connections


def _sumo_lines(net_path):
    import sumolib
    net = sumolib.net.readNet(net_path, withInternal=True)
    normal, internal = [], []
    for e in net.getEdges(withInternal=True):
        for ln in e.getLanes():
            (internal if e.getFunction() == "internal" else normal).append(np.asarray(ln.getShape()))
    return normal, internal


def stress(d, netd, veh_per_h=300, dur=900):
    """每个可达 进口→出口 组合一条持续车流，跑 dur 秒，统计到达、瞬移、碰撞、平均行程时间。"""
    import sumolib
    net = sumolib.net.readNet(os.path.join(d, "net.net.xml"), withInternal=False)
    routes = []
    for a in netd["starts"]:
        for b in netd["ends"]:
            if net.hasEdge(a) and net.hasEdge(b):
                path, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b))
                if path:
                    routes.append(" ".join(e.getID() for e in path))
    with open(os.path.join(d, "stress.rou.xml"), "w", encoding="utf-8") as f:
        f.write('<routes>\n<vType id="car" length="4.5" minGap="2.0" sigma="0.5"/>\n')
        for k, r in enumerate(routes):
            f.write(f'<route id="r{k}" edges="{r}"/>\n')
        for k in range(len(routes)):
            f.write(f'<flow id="f{k}" type="car" route="r{k}" begin="0" end="{dur}" vehsPerHour="{veh_per_h}" '
                    f'departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")
    import subprocess
    cmd = [sumo_check._tool("sumo"), "-n", "net.net.xml", "-r", "stress.rou.xml", "--no-step-log", "true",
           "--duration-log.disable", "true", "--statistic-output", "stress_stat.xml", "--tripinfo-output", "trips.xml",
           "--collision.action", "warn", "--time-to-teleport", "120", "--end", str(dur + 600), "--no-warnings", "true"]
    subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=900)
    r = ET.parse(os.path.join(d, "stress_stat.xml")).getroot()
    vh, tp, sf, vt = r.find("vehicles"), r.find("teleports"), r.find("safety"), r.find("vehicleTripStatistics")
    return dict(flows=len(routes), veh_per_h_each=veh_per_h, inserted=int(vh.get("inserted")),
                arrived=int(vh.get("inserted")) - int(vh.get("running")) - int(vh.get("waiting", 0)),
                waiting_not_inserted=int(vh.get("waiting", 0)), teleports=int(tp.get("total")),
                collisions=int(sf.get("collisions")), mean_travel_s=round(float(vt.get("duration")), 1),
                mean_wait_s=round(float(vt.get("waitingTime")), 1))


def run_one(name, lon, lat, root):
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    T = {}
    t = time.time()
    res = pipeline.run(lon, lat, log=lambda *a: None)
    T["识别(车道+结构)"] = time.time() - t
    t = time.time()
    rep = pipeline.export(res, out)
    m = evaluate.metrics(res)
    T["导出+自检指标"] = time.time() - t
    cx, cy = res["core"].centroid.x, res["core"].centroid.y
    half = 70 if res["core"].area < 1500 else 95
    ext = (cx - half, cx + half, cy - half, cy + half)
    t = time.time()
    title = f"{name}  车道 {rep['n_lanes']}  路臂 {len(res['model']['arms'])}  渠化右转 {len(res['model']['slips'])}"
    render.draw(res, os.path.join(out, "1_识别车道_路口.png"), extent=ext, title=title)
    render.draw(res, os.path.join(out, "2_识别车道_全貌.png"), title=title)
    T["出叠图"] = time.time() - t

    t = time.time()
    netd = sumo_check.build_network(res)
    diag = {"fallback_approaches": 0}
    jc = cpsat_connections(res, netd, diagnostics=diag)
    T["SUMO 编译(含 CP-SAT 转向)"] = time.time() - t
    sd = os.path.join(out, "sumo")
    os.makedirs(sd, exist_ok=True)
    t = time.time()
    ok, warns, err = sumo_check.compile_net(netd, jc, sd, res["lanes"])
    T["netconvert 成网"] = time.time() - t
    sim = {}
    if ok:
        t = time.time()
        cov = sumo_check.coverage(res, netd, sd)
        sim["单车遍历"] = sumo_check.simulate(sd, netd)
        T["单车遍历仿真"] = time.time() - t
        t = time.time()
        sim["持续车流"] = stress(sd, netd)
        T["持续车流仿真(900 s)"] = time.time() - t
        normal, internal = _sumo_lines(os.path.join(sd, "net.net.xml"))
        r2 = dict(res)
        r2["lanes"] = []
        render.draw(r2, os.path.join(out, "3_SUMO路网_路口.png"), extent=ext, show_events=False,
                    title=f"{name}  SUMO 路网（蓝：路段车道，黄：路口内部转向连接）",
                    extra_lines=[(normal, "#4db8ff", 1.4), (internal, "#ffd23f", 0.9)])
    else:
        cov = {}
    summary = dict(name=name, lon=lon, lat=lat, status=res.get("status"),
                   n_lanes=rep["n_lanes"], n_arms=len(res["model"]["arms"]), n_slips=len(res["model"]["slips"]),
                   arms=[round(a["bearing"]) for a in res["model"]["arms"]],
                   self_check={k: m.get(k) for k in ("on_mark_m", "stop_lane_eq", "arrow_recall80", "dp_fallback_rate",
                                                      "jump_transitions", "crossings", "n_add_drop")},
                   sumo=dict(net_ok=ok, netconvert_warnings=len(warns), **{k: v for k, v in cov.items() if k != "mv_missing"},
                             mv_missing=[list(x) for x in cov.get("mv_missing", [])]),
                   sim=sim, seconds={k: round(v, 1) for k, v in T.items()},
                   total_s=round(sum(T.values()), 1))
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(sd, "netconvert_warnings.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(warns))
    return summary


if __name__ == "__main__":
    root = sys.argv[1]
    for arg in sys.argv[2:]:
        name, ll = arg.split("=")
        lon, lat = map(float, ll.split(","))
        s = run_one(name, lon, lat, root)
        print(json.dumps(s, ensure_ascii=False, default=float), flush=True)
