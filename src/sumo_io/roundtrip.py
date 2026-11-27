"""编译后往返一致性核验（docs/DESIGN.md §4.3-2）。

三项刚性比对，任一项不通过即判编译失败：
1. 非内部 <connection> 集合 (from, to, fromLane, toLane) 与 G 的 E_succ 完全一致；
2. 死胡同 / 无源车道 / 被丢弃斑马线直接基于 G 与解析后的 net.xml 精确统计；
3. 车道中心线与 K_conn 内部轨迹的编译前后 Hausdorff 漂移与端点漂移。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.config import load_config
from src.common.geom import endpoint_drift, hausdorff
from src.graph.lane_graph import LaneGraph
from src.sumo_io.serializer import SumoIds


def parse_shape(s: str) -> np.ndarray:
    return np.array([[float(v) for v in p.split(",")[:2]] for p in s.split()], dtype=np.float64)


@dataclass
class ParsedNet:
    lane_shapes: dict[str, np.ndarray]
    lane_allow: dict[str, str]
    lane_attrs: dict[str, dict]
    connections: list[dict]
    crossings: list[dict]
    junction_types: dict[str, str]
    net_offset: tuple[float, float]


def parse_net(path: Path) -> ParsedNet:
    root = ET.parse(path).getroot()
    lane_shapes, lane_allow, lane_attrs = {}, {}, {}
    crossings = []
    loc = root.find("location")
    off = tuple(float(v) for v in loc.get("netOffset").split(",")) if loc is not None else (0.0, 0.0)
    for e in root.findall("edge"):
        fn = e.get("function", "normal")
        for ln in e.findall("lane"):
            lane_shapes[ln.get("id")] = parse_shape(ln.get("shape"))
            lane_allow[ln.get("id")] = ln.get("allow", "")
            lane_attrs[ln.get("id")] = dict(ln.attrib)
        if fn == "crossing":
            ln = e.find("lane")
            crossings.append({"id": e.get("id"), "edges": e.get("crossingEdges", "").split(),
                              "shape": parse_shape(ln.get("shape")), "width": float(ln.get("width"))})
    conns = [dict(c.attrib) for c in root.findall("connection")]
    jt = {j.get("id"): j.get("type") for j in root.findall("junction")}
    return ParsedNet(lane_shapes, lane_allow, lane_attrs, conns, crossings, jt, off)


@dataclass
class RoundTripReport:
    ok: bool
    conn_expected: int
    conn_compiled: int
    conn_missing: list[tuple] = field(default_factory=list)
    conn_extra: list[tuple] = field(default_factory=list)
    dead_end_g: int = 0
    unfed_g: int = 0
    dead_end_net: int = 0
    unfed_net: int = 0
    crossings_expected: int = 0
    crossings_compiled: int = 0
    lane_hausdorff_max: float = 0.0
    lane_endpoint_drift_max: float = 0.0
    conn_hausdorff_max: float = 0.0
    conn_endpoint_drift_max: float = 0.0
    crossing_hausdorff_max: float = 0.0
    lane_change_attrs_ok: bool = True
    permission_ok: bool = True
    failures: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {k: (v if not isinstance(v, list) else len(v)) for k, v in self.__dict__.items()}


def classify_warnings(g: LaneGraph, ids: SumoIds, warnings: list[str]) -> tuple[list[str], list[str]]:
    """把 netconvert 告警分为 (G 显式声明的拓扑事实, 意外告警)。

    仅以下两类告警与 G 精确对应、不属编译缺陷：
    - ``Edge 'e' is not connected to outgoing edges``：e 的全部车道在 G 中均为死胡同（零出边）；
    - ``Lane 'x' is not connected from any incoming edge``：x 在 G 中为无源车道。
    其余一律视为意外告警（门禁要求为 0）。
    """
    import re
    nodes = {n.id: n for n in g.nodes}
    out_lanes = {c.from_lane for c in g.connections}
    in_lanes = {c.to_lane for c in g.connections}
    dead_edges = {ids.edge[e.id] for e in g.edges
                  if nodes[e.to_node].node_type != "roi_exit_port" and not (set(e.lanes) & out_lanes)}
    unfed = {ids.lane_sumo_id(g, l.id) for l in g.lanes
             if nodes[g.lane_from(l.id)].node_type != "roi_entry_port" and l.id not in in_lanes}
    declared, unexpected = [], []
    for w in warnings:
        m1 = re.search(r"Edge '([^']+)' is not connected to outgoing edges", w)
        m2 = re.search(r"Lane '([^']+)' is not connected from any incoming edge", w)
        if (m1 and m1.group(1) in dead_edges) or (m2 and m2.group(1) in unfed):
            declared.append(w)
        else:
            unexpected.append(w)
    return declared, unexpected


def _internal_chain(net: ParsedNet, conn: dict) -> list[str]:
    """沿 via 追踪一条连接的内部车道序列（可能被内部路口拆成多段）。"""
    chain = []
    via = conn.get("via")
    target = conn["to"]
    guard = 0
    while via and guard < 8:
        chain.append(via)
        iedge, _, ilane = via.rpartition("_")
        nxt = [c for c in net.connections if c["from"] == iedge and c["fromLane"] == ilane and c["to"] == target]
        via = nxt[0].get("via") if nxt else None
        guard += 1
    return chain


def verify_roundtrip(g: LaneGraph, ids: SumoIds, net_path: Path, tol: float | None = None) -> RoundTripReport:
    cfg = load_config("sumo_export")["roundtrip"]
    tol = cfg["hausdorff_tol_m"] if tol is None else tol
    step = cfg["densify_step_m"]
    net = parse_net(net_path)
    e_of = g.edge_of_lane()
    li = g.lane_index()
    nodes = {n.id: n for n in g.nodes}
    sumo_edges = set(ids.edge.values())
    rev_edge = {v: k for k, v in ids.edge.items()}

    # ---------------------------------------------------------------- 1. 连接拓扑
    def key(c):
        return (ids.edge[e_of[c.from_lane].id], ids.edge[e_of[c.to_lane].id],
                li[c.from_lane] + ids.lane_offset[e_of[c.from_lane].id],
                li[c.to_lane] + ids.lane_offset[e_of[c.to_lane].id])
    expected = {key(c): c for c in g.connections}
    compiled = {}
    for c in net.connections:
        if c["from"] in sumo_edges and c["to"] in sumo_edges:
            compiled[(c["from"], c["to"], int(c["fromLane"]), int(c["toLane"]))] = c
    missing = sorted(set(expected) - set(compiled))
    extra = sorted(set(compiled) - set(expected))
    rep = RoundTripReport(ok=True, conn_expected=len(expected), conn_compiled=len(compiled),
                          conn_missing=missing, conn_extra=extra)
    if missing or extra:
        rep.failures.append(f"连接往返不一致：缺失 {len(missing)}，多出 {len(extra)}")

    # ---------------------------------------------------------------- 2. 死胡同 / 无源 / 斑马线
    out_g = {c.from_lane for c in g.connections}
    in_g = {c.to_lane for c in g.connections}
    lane_sid = {l.id: ids.lane_sumo_id(g, l.id) for l in g.lanes}
    out_net = {f"{k[0]}_{k[2]}" for k in compiled}
    in_net = {f"{k[1]}_{k[3]}" for k in compiled}
    for l in g.lanes:
        e = e_of[l.id]
        if nodes[e.to_node].node_type != "roi_exit_port":
            rep.dead_end_g += l.id not in out_g
            rep.dead_end_net += lane_sid[l.id] not in out_net
        if nodes[e.from_node].node_type != "roi_entry_port":
            rep.unfed_g += l.id not in in_g
            rep.unfed_net += lane_sid[l.id] not in in_net
    if rep.dead_end_net != rep.dead_end_g or rep.unfed_net != rep.unfed_g:
        rep.failures.append("死胡同/无源车道统计与 G 不一致")
    rep.crossings_expected = len(g.crosswalks)
    rep.crossings_compiled = len(net.crossings)
    if rep.crossings_compiled != rep.crossings_expected:
        rep.failures.append(f"斑马线被丢弃：期望 {rep.crossings_expected}，编译 {rep.crossings_compiled}")

    # ---------------------------------------------------------------- 3. 几何保真
    forbid = load_config("sumo_export")["lane_change_forbidden"]
    perm_cfg = load_config("sumo_export")["permission"]
    for l in g.lanes:
        sid = lane_sid[l.id]
        if sid not in net.lane_shapes:
            rep.failures.append(f"车道 {sid} 在 net.xml 中缺失")
            continue
        shp = net.lane_shapes[sid]
        rep.lane_hausdorff_max = max(rep.lane_hausdorff_max, hausdorff(l.points, shp, step))
        rep.lane_endpoint_drift_max = max(rep.lane_endpoint_drift_max, endpoint_drift(l.points, shp))
        attrs = net.lane_attrs[sid]
        want_l = None if l.allow_change_left else forbid
        want_r = None if l.allow_change_right else forbid
        if attrs.get("changeLeft") != want_l or attrs.get("changeRight") != want_r:
            rep.lane_change_attrs_ok = False
        perm = perm_cfg[l.permission]
        if "allow" in perm and attrs.get("allow") != perm["allow"]:
            rep.permission_ok = False
    for k, c in expected.items():
        if k not in compiled:
            continue
        chain = _internal_chain(net, compiled[k])
        if not chain:
            rep.failures.append(f"连接 {k} 无内部车道")
            continue
        shp = np.concatenate([net.lane_shapes[v] for v in chain])
        rep.conn_hausdorff_max = max(rep.conn_hausdorff_max, hausdorff(c.points, shp, step))
        rep.conn_endpoint_drift_max = max(rep.conn_endpoint_drift_max, endpoint_drift(c.points, shp))
    for cw in g.crosswalks:
        want = {ids.edge[e] for e in cw.edges}
        hit = [x for x in net.crossings if set(x["edges"]) == want]
        if hit:
            # 同一 edge 集合可能在不同节点各有一条斑马线（如走廊连接段两端），取几何最近者
            rep.crossing_hausdorff_max = max(rep.crossing_hausdorff_max,
                                             min(hausdorff(cw.endpoints, x["shape"], step) for x in hit))
        else:
            rep.failures.append(f"斑马线 {cw.id} 未按 edges={sorted(want)} 编译")
    if rep.lane_hausdorff_max > tol or rep.conn_hausdorff_max > tol or rep.crossing_hausdorff_max > tol:
        rep.failures.append(f"几何 Hausdorff 漂移超限（车道 {rep.lane_hausdorff_max:.3f} m，"
                            f"连接 {rep.conn_hausdorff_max:.3f} m，斑马线 {rep.crossing_hausdorff_max:.3f} m）")
    if not rep.lane_change_attrs_ok:
        rep.failures.append("禁变道属性未按 G 保留")
    if not rep.permission_ok:
        rep.failures.append("车道权限未按 G 保留")
    _ = rev_edge
    rep.ok = not rep.failures
    return rep
