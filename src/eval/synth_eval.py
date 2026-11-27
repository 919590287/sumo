"""M1 合成留出测试集评测（docs/DESIGN.md §7.1 / §7.2 M1）。

逐样本：约束解码图 G 与 ROI 真值子图 G* 比较——
- 车道中心线横向误差（P50 / P90，TP 车道全部控制点到真值中心线距离）、有效车道 F1、人行横道 F1；
- 无标线支路 / 非机动车道 / 停车 / 港湾式公交站误建率 FPR（预测 L ∪ K_conn 包络落入负样本区域的面积比）；
- 导向箭头转向 F1（箭头所在真值车道 ↔ 匹配预测车道的出向连接转向集合，多标签）；
- 双轨拓扑：约束解码前原始阈值输出 vs 解码后（死胡同 / 无源 / 同进口交叉）；
- SUMO：往返一致编译（零错 + 连接集合 100% 一致 + Hausdorff ≤ 0.20 m + 零意外告警）、交付网稀疏单 OD 与
  测试台（--tls.set）全 OD 探测流在 3 个固定种子下的到达率与 collisions == 0。
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString

from src.common.config import load_config
from src.common.geom import tangents
from src.eval.quick import lateral_errors
from src.graph.lane_graph import LaneGraph

TURNS = ("L", "S", "R", "U", "FL", "FR")


def turn_class(psi_deg: float) -> str:
    tc = load_config("synth")["turn_class_deg"]
    a = abs(psi_deg)
    if a <= tc["straight"]:
        return "S"
    if a <= tc["slight"]:
        return "FL" if psi_deg > 0 else "FR"
    if a <= tc["uturn"]:
        return "L" if psi_deg > 0 else "R"
    return "U"


def lane_turns(g: LaneGraph, lid: int) -> set[str]:
    lanes = {l.id: l for l in g.lanes}
    out = set()
    for c in g.connections:
        if c.from_lane != lid:
            continue
        ti = tangents(lanes[lid].points)[-1]
        tj = tangents(lanes[c.to_lane].points)[0]
        out.add(turn_class(math.degrees(math.atan2(ti[0] * tj[1] - ti[1] * tj[0], float(ti @ tj)))))
    return out


def match_lanes(pred: LaneGraph, gt: LaneGraph, thr: float) -> dict[int, int]:
    if not pred.lanes or not gt.lanes:
        return {}
    P = np.stack([l.points[:, :2] for l in pred.lanes])
    G = np.stack([l.points[:, :2] for l in gt.lanes])
    C = np.abs(P[:, None] - G[None]).sum(-1).mean(-1) / 2
    r, c = linear_sum_assignment(C)
    return {pred.lanes[i].id: gt.lanes[j].id for i, j in zip(r, c) if C[i, j] < thr}


def match_crosswalks(pred: LaneGraph, gt: LaneGraph, thr: float) -> int:
    if not pred.crosswalks or not gt.crosswalks:
        return 0
    P = np.stack([c.endpoints[:, :2] for c in pred.crosswalks])
    G = np.stack([c.endpoints[:, :2] for c in gt.crosswalks])
    d1 = np.abs(P[:, None] - G[None]).sum(-1).mean(-1) / 2
    d2 = np.abs(P[:, None] - G[None, :, ::-1]).sum(-1).mean(-1) / 2
    C = np.minimum(d1, d2)
    r, c = linear_sum_assignment(C)
    return int((C[r, c] < thr).sum())


def envelope_mask(g: LaneGraph, H: int, W: int, res: float) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    lanes = {l.id: l for l in g.lanes}

    def fill(poly):
        for p in getattr(poly, "geoms", [poly]):
            if p.is_empty or p.geom_type != "Polygon":
                continue
            xy = np.asarray(p.exterior.coords)
            ring = np.stack([xy[:, 0] / res - 0.5, H - xy[:, 1] / res - 0.5], 1)
            cv2.fillPoly(m, [np.round(ring * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    for l in g.lanes:
        fill(LineString(l.points[:, :2]).buffer(l.width / 2, cap_style="flat"))
    for c in g.connections:
        fill(LineString(c.points[:, :2]).buffer(lanes[c.from_lane].width / 2))
    return m


def poly_mask(polys, H: int, W: int, res: float) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for q in polys:
        for p in getattr(q, "geoms", [q]):
            if p.is_empty or p.geom_type != "Polygon":
                continue
            xy = np.asarray(p.exterior.coords)
            ring = np.stack([xy[:, 0] / res - 0.5, H - xy[:, 1] / res - 0.5], 1)
            cv2.fillPoly(m, [np.round(ring * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    return m


def eval_geometry(pred: LaneGraph | None, gt: LaneGraph, meta: dict, hw, res: float) -> dict:
    ev = load_config("eval")["match"]
    H, W = int(hw[0]), int(hw[1])
    rec = {"n_gt_lane": len(gt.lanes), "n_gt_cw": len(gt.crosswalks), "lat": np.zeros(0)}
    negs = meta.get("neg_regions", [])
    # 负样本区域取内部（腐蚀 1 像素），与之相邻接壤的机动车道包络边界像素不计入误建
    neg_m = cv2.erode(poly_mask(negs, H, W, res), np.ones((3, 3), np.uint8)) if negs else None
    rec["neg_area"] = float(neg_m.sum()) * res * res if neg_m is not None else 0.0
    gt_turns = []
    for a in meta.get("arrows", []):
        if a.get("lane") is not None:
            gt_turns.append((a["lane"], set(a["turns"])))
    if pred is None:
        rec.update({"n_pred_lane": 0, "tp_lane": 0, "n_pred_cw": 0, "tp_cw": 0, "neg_hit": 0.0,
                    "arrow_tp": 0, "arrow_fp": 0, "arrow_fn": sum(len(t) for _, t in gt_turns)})
        return rec
    m = match_lanes(pred, gt, ev["lane_m"])
    pl = {l.id: l for l in pred.lanes}
    gl = {l.id: l for l in gt.lanes}
    errs = [lateral_errors(pl[p].points[:, :2], gl[g].points[:, :2]) for p, g in m.items()]
    rec.update({"n_pred_lane": len(pred.lanes), "tp_lane": len(m),
                "lat": np.concatenate(errs) if errs else np.zeros(0),
                "n_pred_cw": len(pred.crosswalks), "tp_cw": match_crosswalks(pred, gt, ev["crosswalk_m"])})
    if neg_m is not None:
        pm = envelope_mask(pred, H, W, res)
        rec["neg_hit"] = float((pm & neg_m).sum()) * res * res
    else:
        rec["neg_hit"] = 0.0
    g2p = {g: p for p, g in m.items()}
    tp = fp = fn = 0
    for lid, turns in gt_turns:
        pt = lane_turns(pred, g2p[lid]) if lid in g2p else set()
        tp += len(pt & turns)
        fp += len(pt - turns)
        fn += len(turns - pt)
    rec.update({"arrow_tp": tp, "arrow_fp": fp, "arrow_fn": fn})
    return rec


def eval_sumo(g: LaneGraph, work: Path, seeds: list[int], tol: float) -> dict:
    from src.sumo_io.pipeline import compile_graph
    from src.sumo_io.roundtrip import classify_warnings
    from src.sumo_io.simulate import run_sumo, write_demand
    rec = {"compile_ok": False}
    try:
        r = compile_graph(g, work, tol=tol)
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:200]
        return rec
    _, unexpected = classify_warnings(g, r.plain.ids, r.site.warnings)
    _, unexpected_ev = classify_warnings(g, r.plain.ids, r.eval_sim.warnings) if r.eval_sim else ([], [])
    # 测试台副本中 netconvert 自动生成防死锁配时的周期适配提示只属于测试台，不是 G 的缺陷：单列记录
    tb = [w for w in unexpected_ev if w.startswith("Warning: The traffic light")]
    unexpected_ev = [w for w in unexpected_ev if w not in tb]
    rec["testbed_tls_notices"] = len(tb)
    rec["unexpected_warnings"] = len(unexpected) + len(unexpected_ev)
    rec["warning_samples"] = (unexpected + unexpected_ev)[:3]
    rec["roundtrip"] = r.roundtrip.summary() if r.roundtrip else None
    rec["compile_ok"] = bool(r.ok and not unexpected and not unexpected_ev)
    if not rec["compile_ok"]:
        rec["failures"] = (r.roundtrip.failures if r.roundtrip else r.site.errors)[:3]
        return rec
    sims = {}
    for mode, net in (("sparse_single", r.site.net_path), ("probe", r.eval_sim.net_path)):
        per = []
        for seed in seeds:
            rou = write_demand(g, r.plain.ids, work / f"{mode}_{seed}.rou.xml", mode, seed)
            s = run_sumo(net, rou, work, seed, mode)
            per.append({"seed": seed, "loaded": s.loaded, "arrived": s.arrived, "collisions": s.collisions,
                        "teleports": s.teleports})
        sims[mode] = per
    rec["sim"] = sims
    return rec


def aggregate(recs: list[dict]) -> dict:
    lat = np.concatenate([r["lat"] for r in recs if len(r["lat"])]) if any(len(r["lat"]) for r in recs) else np.array([np.nan])
    f1 = lambda tp, npred, ngt: 2 * tp / max(npred + ngt, 1)  # noqa: E731
    s = lambda k: sum(r.get(k, 0) for r in recs)  # noqa: E731
    out = {
        "n_samples": len(recs),
        "decode_ok_rate": float(np.mean([r.get("decode_ok", False) for r in recs])),
        "lat_p50_m": float(np.nanpercentile(lat, 50)), "lat_p90_m": float(np.nanpercentile(lat, 90)),
        "lane_f1": f1(s("tp_lane"), s("n_pred_lane"), s("n_gt_lane")),
        "crosswalk_f1": f1(s("tp_cw"), s("n_pred_cw"), s("n_gt_cw")),
        "neg_fpr": s("neg_hit") / max(s("neg_area"), 1e-9),
        "arrow_turn_f1": (2 * s("arrow_tp") / (2 * s("arrow_tp") + s("arrow_fp") + s("arrow_fn"))
                          if (s("arrow_tp") + s("arrow_fp") + s("arrow_fn")) else float("nan")),
        "neg_area_m2": s("neg_area"),
    }
    for track in ("raw", "decoded"):
        de = s(f"{track}_dead_end")
        un = s(f"{track}_unfed")
        out[f"{track}_dead_end_rate"] = de / max(s("n_pred_lane"), 1)
        out[f"{track}_unfed_rate"] = un / max(s("n_pred_lane"), 1)
        out[f"{track}_same_approach_crossings"] = s(f"{track}_crossings")
    sumo = [r for r in recs if "sumo" in r]
    if sumo:
        ok = [r["sumo"].get("compile_ok", False) for r in sumo]
        out["roundtrip_rate"] = float(np.mean([r.get("decode_ok", False) and r["sumo"].get("compile_ok", False)
                                               for r in sumo]))
        comp = [r["sumo"] for r in sumo if r["sumo"].get("compile_ok")]
        for mode in ("sparse_single", "probe"):
            rates = [p["arrived"] / p["loaded"] for c in comp for p in c["sim"][mode] if p["loaded"]]
            out[f"{mode}_arrival_rate"] = float(np.mean(rates)) if rates else float("nan")
        out["collision_free_rate"] = float(np.mean([all(p["collisions"] == 0 for p in c["sim"]["probe"])
                                                    for c in comp])) if comp else float("nan")
        out["teleport_free_rate"] = float(np.mean([all(p["teleports"] == 0 for m in c["sim"].values() for p in m)
                                                   for c in comp])) if comp else float("nan")
        out["n_compiled"] = int(sum(ok))
    gates = load_config("eval")["gates_m1"]
    checks = {
        "lat_p50": out["lat_p50_m"] <= gates["lat_p50_m"], "lat_p90": out["lat_p90_m"] <= gates["lat_p90_m"],
        "lane_f1": out["lane_f1"] >= gates["lane_f1"], "crosswalk_f1": out["crosswalk_f1"] >= gates["crosswalk_f1"],
    }
    if s("neg_area") > 0:
        checks["neg_fpr"] = out["neg_fpr"] <= gates["neg_fpr"]
    if out["arrow_turn_f1"] == out["arrow_turn_f1"]:
        checks["arrow_turn_f1"] = out["arrow_turn_f1"] >= gates["arrow_turn_f1"]
    if sumo:
        checks.update({"roundtrip_rate": out["roundtrip_rate"] >= gates["roundtrip_rate"],
                       "sparse_arrival": out["sparse_single_arrival_rate"] >= gates["arrival_rate"],
                       "probe_arrival": out["probe_arrival_rate"] >= gates["arrival_rate"],
                       "collision_free": out["collision_free_rate"] >= gates["collision_free_rate"]})
    out["gates"] = {k: bool(v) for k, v in checks.items()}
    out["all_gates_pass"] = all(checks.values())
    return out


def clean(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)
