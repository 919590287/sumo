"""M0 验收第 2 条：手工构造的 20+ 种复杂拓扑标准图 → SUMO 编译 / 往返核验 / 仿真。

用法（在仓库根目录）：
    /opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.run_sumo_suite [--out runs/m0/sumo_suite]

门禁：
- netconvert 零错误、零意外告警（G 显式声明的死胡同/无源车道提示单列统计）；
- 往返连接集合 100% 一致；编译前后车道 / 内部连接 / 斑马线几何 Hausdorff ≤ 0.15 m；
- 3 个固定种子下：交付网稀疏单车 + 测试台探测流 + 测试台中等冲突流 全部
  collisions == 0、teleports == 0、到达率 100%；
- changeLeft/changeRight="authority" A/B 实测阻断民用车变道。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.common.config import REPO_ROOT, cache_dir, load_config
from src.data.standard_graphs import STANDARD_GRAPHS, GraphBuilder
from src.sumo_io.pipeline import compile_graph
from src.sumo_io.roundtrip import classify_warnings
from src.sumo_io.simulate import run_sumo, write_demand

TOL = load_config("sumo_export")["roundtrip"]["hausdorff_tol_m"]


def run_graph(name: str, g, work: Path, seeds: list[int]) -> dict:
    t0 = time.time()
    r = compile_graph(g, work / name)
    rt = r.roundtrip
    declared, unexpected = classify_warnings(g, r.plain.ids, r.site.warnings)
    _, unexpected_ev = classify_warnings(g, r.plain.ids, r.eval_sim.warnings)
    rec = {
        "name": name,
        "lanes": len(g.lanes), "edges": len(g.edges), "connections": len(g.connections),
        "crosswalks": len(g.crosswalks), "nodes": {t: sum(n.node_type == t for n in g.nodes)
                                                  for t in sorted({n.node_type for n in g.nodes})},
        "compile_ok": r.site.ok and r.eval_sim.ok,
        "errors": r.site.errors + r.eval_sim.errors,
        "declared_topology_notices": declared,
        "unexpected_warnings": unexpected + unexpected_ev,
        "roundtrip": rt.summary() if rt else None,
        "roundtrip_failures": rt.failures if rt else ["未编译"],
        "sim": {},
    }
    if rt is not None:
        for mode, net in (("sparse_single", r.site.net_path), ("probe", r.eval_sim.net_path),
                          ("conflict", r.eval_sim.net_path)):
            per = []
            for seed in seeds:
                rou = write_demand(g, r.plain.ids, work / name / f"{mode}_{seed}.rou.xml", mode, seed)
                s = run_sumo(net, rou, work / name, seed, mode)
                per.append({"seed": seed, "loaded": s.loaded, "arrived": s.arrived, "collisions": s.collisions,
                            "teleports": s.teleports, "errors": s.errors[:3]})
            rec["sim"][mode] = per
    sim_ok = all(p["loaded"] > 0 and p["arrived"] == p["loaded"] and p["collisions"] == 0 and
                 p["teleports"] == 0 and not p["errors"] for ps in rec["sim"].values() for p in ps) \
        if rec["sim"] else False
    rec["pass"] = bool(rec["compile_ok"] and not rec["errors"] and not rec["unexpected_warnings"]
                       and rt is not None and rt.ok and sim_ok)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def lane_change_ab(work: Path, seeds: list[int]) -> dict:
    """A/B：同一 2 车道 400 m 路段，快车跟随慢车；对照组允许变道，实验组 authority 禁变道。"""
    out = {}
    for tag, allow in (("allow", True), ("authority", False)):
        b = GraphBuilder(f"lc_{tag}")
        p_in = b.node("roi_entry_port", (0, -3.5))
        mid = b.node("road_continuation", (400, -3.5))
        p_out = b.node("roi_exit_port", (500, -3.5))
        e1 = b.edge(p_in, mid, np.array([[0.0, 0.0], [398.0, 0.0]]), [3.5, 3.5],
                    change_left=[allow, allow], change_right=[allow, allow], roles=["mid_segment"] * 2)
        e2 = b.edge(mid, p_out, np.array([[402.0, 0.0], [500.0, 0.0]]), [3.5, 3.5], roles=["mid_segment"] * 2)
        b.connect_edges(e1, e2, [(0, 0), (1, 1)])
        g = b.build()
        r = compile_graph(g, work / f"lc_{tag}", with_eval_sim=False)
        ids = r.plain.ids
        rou = work / f"lc_{tag}" / "lc.rou.xml"
        veh = []
        for k in range(10):
            t = k * 20.0
            veh.append(f'    <vehicle id="slow{k}" type="slow" depart="{t:.1f}" departLane="0" departPos="60">\n'
                       f'        <route edges="{ids.edge[e1]} {ids.edge[e2]}"/>\n    </vehicle>\n')
            veh.append(f'    <vehicle id="fast{k}" type="fast" depart="{t + 2:.1f}" departLane="0" departPos="0">\n'
                       f'        <route edges="{ids.edge[e1]} {ids.edge[e2]}"/>\n    </vehicle>\n')
        rou.write_text('<routes>\n    <vType id="slow" vClass="passenger" maxSpeed="4"/>\n'
                       '    <vType id="fast" vClass="passenger" maxSpeed="20"/>\n' + "".join(veh) + "</routes>\n",
                       encoding="utf-8")
        per = []
        for seed in seeds:
            s = run_sumo(r.site.net_path, rou, work / f"lc_{tag}", seed, "lc", lanechange=True)
            on_e1 = sum(f.startswith(ids.edge[e1] + "_") for f in s.lane_change_from)
            per.append({"seed": seed, "lane_changes": on_e1, "lane_changes_total": s.lane_changes,
                        "arrived": s.arrived, "loaded": s.loaded,
                        "collisions": s.collisions, "teleports": s.teleports})
        out[tag] = {"roundtrip_ok": bool(r.roundtrip and r.roundtrip.ok), "runs": per}
    out["pass"] = (all(p["lane_changes"] > 0 for p in out["allow"]["runs"])
                   and all(p["lane_changes"] == 0 for p in out["authority"]["runs"])
                   and out["authority"]["roundtrip_ok"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/m0/sumo_suite")
    args = ap.parse_args()
    seeds = load_config("sim")["seeds"]
    work = cache_dir("m0_sumo_suite")
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for name, f in STANDARD_GRAPHS.items():
        g = f()
        rec = run_graph(name, g, work, seeds)
        recs.append(rec)
        rt = rec["roundtrip"] or {}
        sims = " ".join(f"{m[:5]}:{sum(p['arrived'] for p in ps)}/{sum(p['loaded'] for p in ps)}"
                        f" c{sum(p['collisions'] for p in ps)} t{sum(p['teleports'] for p in ps)}"
                        for m, ps in rec["sim"].items())
        print(f"{'PASS' if rec['pass'] else 'FAIL'} {name:22s} conn={rt.get('conn_compiled')}/{rt.get('conn_expected')} "
              f"H_lane={rt.get('lane_hausdorff_max', float('nan')):.4f} H_conn={rt.get('conn_hausdorff_max', float('nan')):.4f} "
              f"cw={rt.get('crossings_compiled')}/{rt.get('crossings_expected')} warn={len(rec['unexpected_warnings'])} "
              f"notice={len(rec['declared_topology_notices'])} | {sims}", flush=True)
    lc = lane_change_ab(work, seeds)
    print(f"{'PASS' if lc['pass'] else 'FAIL'} changeLeft/Right=authority A/B：允许组变道次数 "
          f"{[p['lane_changes'] for p in lc['allow']['runs']]}，禁变道组 "
          f"{[p['lane_changes'] for p in lc['authority']['runs']]}")
    hl = [r["roundtrip"]["lane_hausdorff_max"] for r in recs if r["roundtrip"]]
    hc = [r["roundtrip"]["conn_hausdorff_max"] for r in recs if r["roundtrip"]]
    summary = {
        "n_graphs": len(recs),
        "n_pass": sum(r["pass"] for r in recs),
        "compile_zero_error_zero_unexpected_warning_rate": float(np.mean(
            [r["compile_ok"] and not r["errors"] and not r["unexpected_warnings"] for r in recs])),
        "roundtrip_connection_exact_rate": float(np.mean(
            [bool(r["roundtrip"]) and r["roundtrip"]["conn_missing"] == 0 and r["roundtrip"]["conn_extra"] == 0
             for r in recs])),
        "max_lane_hausdorff_m": max(hl), "max_conn_hausdorff_m": max(hc), "hausdorff_tol_m": TOL,
        "lane_change_authority_ab_pass": lc["pass"],
        "seeds": seeds,
        "all_pass": all(r["pass"] for r in recs) and lc["pass"] and len(recs) >= 20,
    }
    report = {"summary": summary, "graphs": recs, "lane_change_ab": lc,
              "config": {"sumo_export": load_config("sumo_export"), "sim": load_config("sim")}}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
