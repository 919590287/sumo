"""M0 手工构造的复杂拓扑标准图库（docs/DESIGN.md §7.2 M0 验收第 2 条）。

这些图是**测试夹具 / 合成真值样例**（Layer 1 参数化合成引擎的雏形），用于验证
SUMO 序列化器、netconvert 往返一致性与仿真测试台，不参与任何推理期裁决。

构造约定（与 §5 Layer 1 规范化分段协议一致）：
- 车道段 L 恒宽、32 个等弧长控制点；edge 内车道自右（lane_0）向左排列；
- junction_core 轮廓精确经过所有接入 edge 端点横断面角点（按角序，填充至 POLY_K 点）；
- K_conn 由端点位置 + 切向锁定的三次 Hermite 曲线生成 8 点轨迹。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from shapely.geometry import Polygon

from src.common.geom import hermite_curve, offset_polyline, resample_polyline, tangents
from src.graph.lane_graph import (K_CONN, K_LANE, POLY_K, Connection, Crosswalk, Edge, Lane,
                                  LaneGraph, Node)


def unit(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def pad_ring(ring: np.ndarray, k: int) -> np.ndarray:
    """在最长边上插入中点直至 k 个顶点（不改变多边形形状）。"""
    pts = [np.asarray(p, float) for p in ring]
    if len(pts) > k:
        raise ValueError(f"轮廓角点 {len(pts)} 超过 POLY_K={k}")
    while len(pts) < k:
        L = [np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts))]
        i = int(np.argmax(L))
        pts.insert(i + 1, (pts[i] + pts[(i + 1) % len(pts)]) / 2)
    return np.asarray(pts)


class GraphBuilder:
    def __init__(self, name: str):
        self.name = name
        self.nodes: list[Node] = []
        self.lanes: list[Lane] = []
        self.edges: list[Edge] = []
        self.conns: list[Connection] = []
        self.cws: list[Crosswalk] = []

    # ------------------------------------------------------------------ 实体
    def node(self, node_type: str, pos) -> int:
        nid = len(self.nodes)
        self.nodes.append(Node(nid, np.asarray(pos, float), node_type))
        return nid

    def edge(self, frm: int, to: int, left_boundary, widths, permissions=None, change_left=None,
             change_right=None, roles=None, stop_line_end=False) -> int:
        """由 edge 左边界折线（行驶方向）与自右向左的车道宽度生成恒宽车道。"""
        lb = resample_polyline(np.asarray(left_boundary, float), max(64, len(left_boundary)))
        n = len(widths)
        lane_ids = []
        for k in range(n):
            off = -(sum(widths[k + 1:]) + widths[k] / 2)
            pts = resample_polyline(offset_polyline(lb, off), K_LANE)
            lid = len(self.lanes)
            role = roles[k] if roles else ("approach" if stop_line_end else "exit")
            self.lanes.append(Lane(
                lid, pts, float(widths[k]), role=role,
                permission=(permissions[k] if permissions else "general"),
                allow_change_left=(change_left[k] if change_left else True),
                allow_change_right=(change_right[k] if change_right else True),
                is_stop_line_end=stop_line_end))
            lane_ids.append(lid)
        eid = len(self.edges)
        self.edges.append(Edge(eid, lane_ids, frm, to))
        return eid

    def lanes_of(self, eid: int) -> list[int]:
        return self.edges[eid].lanes

    def connect(self, li: int, lj: int, tension: float | None = None) -> None:
        a, b = self.lanes[li].points, self.lanes[lj].points
        ta, tb = tangents(a)[-1], tangents(b)[0]
        if np.linalg.norm(b[0, :2] - a[-1, :2]) < 1e-6:
            raise ValueError("连接首尾重合，需留出节点过渡区")
        pts = hermite_curve(a[-1, :2], ta, b[0, :2], tb, K_CONN, tension)
        pts[0], pts[-1] = a[-1, :2], b[0, :2]
        self.conns.append(Connection(li, lj, pts))

    def connect_edges(self, ei: int, ej: int, pairs: list[tuple[int, int]], tension=None) -> None:
        for a, b in pairs:
            self.connect(self.edges[ei].lanes[a], self.edges[ej].lanes[b], tension)

    def crosswalk(self, node: int, edges: list[int], p0, p1, width: float) -> None:
        self.cws.append(Crosswalk(len(self.cws), np.asarray([p0, p1], float), width, node, list(edges)))

    # ------------------------------------------------------------------ 轮廓
    def edge_corners(self, eid: int, at_node: int) -> list[np.ndarray]:
        e = self.edges[eid]
        at_end = e.to_node == at_node
        k = -1 if at_end else 0
        r, l = self.lanes[e.lanes[0]], self.lanes[e.lanes[-1]]
        nr = rot90(tangents(r.points)[k])
        nl = rot90(tangents(l.points)[k])
        return [r.points[k, :2] - nr * r.width / 2, l.points[k, :2] + nl * l.width / 2]

    def core_polygon(self, nid: int) -> np.ndarray:
        pts = []
        for e in self.edges:
            if nid in (e.from_node, e.to_node):
                pts.extend(self.edge_corners(e.id, nid))
        pts = np.asarray(pts)
        ctr = pts.mean(0)
        order = np.argsort(np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0]), kind="stable")
        ring = [pts[order[0]]]
        for i in order[1:]:
            if np.linalg.norm(pts[i] - ring[-1]) > 1e-6 and np.linalg.norm(pts[i] - ring[0]) > 1e-6:
                ring.append(pts[i])
        ring = np.asarray(ring)
        if not Polygon(ring).is_valid:
            raise ValueError(f"{self.name}: 节点 {nid} 轮廓自交")
        return pad_ring(ring, POLY_K)

    def build(self, origin=(0.0, 0.0), meta=None) -> LaneGraph:
        for n in self.nodes:
            if n.node_type == "junction_core" and n.polygon is None:
                n.polygon = self.core_polygon(n.id)
        g = LaneGraph(self.nodes, self.lanes, self.edges, self.conns, self.cws, origin=tuple(origin),
                      meta={"name": self.name, **(meta or {})})
        errs = g.validate()
        if errs:
            raise ValueError(f"{self.name}: " + "; ".join(errs))
        return g


# ====================================================================== 路口臂工厂
@dataclass
class Arm:
    angle: float                       # 臂朝外方位角（度，x 轴逆时针）
    w_in: list[float] = field(default_factory=lambda: [3.5, 3.5])    # 进口车道宽（右→左）
    w_out: list[float] = field(default_factory=lambda: [3.5, 3.5])   # 出口车道宽（右→左）
    median: float = 0.0
    r_stop: float = 16.0               # 进出口车道在路口端的半径
    length: float = 80.0
    bend: float = 0.0                  # 远端横向弯曲量（米），0 为直线引道
    lateral: float = 0.0               # 臂轴线横向错位（错位路口）
    perm_in: list[str] | None = None
    lc_in: list[bool] | None = None    # 进口车道是否允许变道（False → 实线禁变道）
    crosswalk: bool = True
    cw_width: float = 5.0

    def axis(self, s0: float, s1: float, n: int = 64) -> np.ndarray:
        u = unit(self.angle)
        nrm = rot90(u)
        s = np.linspace(s0, s1, n)
        span = max(self.length - self.r_stop, 1e-6)
        lat = self.lateral + self.bend * np.clip((s - self.r_stop) / span, 0, None) ** 2
        return s[:, None] * u + lat[:, None] * nrm


@dataclass
class JunctionHandles:
    core: int
    arms: list[Arm]
    in_edge: dict[int, int]
    out_edge: dict[int, int]


def build_arms(b: GraphBuilder, arms: list[Arm], center=(0.0, 0.0), in_s0=None, out_s0=None,
               in_skip: set | None = None, out_skip: set | None = None) -> JunctionHandles:
    """为每条臂生成进口 edge（入口端口 → 路口核心）与出口 edge（路口核心 → 出口端口）。"""
    c = np.asarray(center, float)
    core = b.node("junction_core", c)
    in_edge, out_edge = {}, {}
    for k, a in enumerate(arms):
        u = unit(a.angle)
        if a.w_in and k not in (in_skip or set()):
            s_far = (in_s0 or {}).get(k, a.length)
            ax = c + a.axis(a.r_stop, s_far)
            lb = offset_polyline(np.c_[ax, np.zeros(len(ax))], a.median / 2)[::-1, :2]
            port = b.node("roi_entry_port", lb[0] + rot90(-u) * (-sum(a.w_in) / 2))
            in_edge[k] = b.edge(port, core, lb, a.w_in, permissions=a.perm_in,
                                change_left=a.lc_in, change_right=a.lc_in, stop_line_end=True)
        if a.w_out and k not in (out_skip or set()):
            s_far = (out_s0 or {}).get(k, a.length)
            ax = c + a.axis(a.r_stop, s_far)
            lb = offset_polyline(np.c_[ax, np.zeros(len(ax))], -a.median / 2)[:, :2]
            port = b.node("roi_exit_port", lb[-1] + rot90(u) * (-sum(a.w_out) / 2))
            out_edge[k] = b.edge(core, port, lb, a.w_out)
    return JunctionHandles(core, arms, in_edge, out_edge)


def turn_class(a_in: Arm, a_out: Arm, same: bool) -> str:
    if same:
        return "U"
    d_in = -unit(a_in.angle)
    d_out = unit(a_out.angle)
    ang = math.degrees(math.atan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], float(d_in @ d_out)))
    if ang > 45:
        return "L"
    if ang < -45:
        return "R"
    return "S"


def connect_movements(b: GraphBuilder, h: JunctionHandles, moves: dict[int, list[list[int]]],
                      tension_scale: float = 1.0) -> None:
    """moves[臂 k] = 每条进口车道（右→左）可去往的出口臂列表。

    同一 (进口臂, 出口臂) 的进口车道按序与出口车道单调配对：右转/直行靠右对齐，
    左转/掉头靠左对齐（纯夹具构造约定）。
    """
    for k, per_lane in moves.items():
        ein = h.in_edge[k]
        targets = sorted({t for lst in per_lane for t in lst})
        for t in targets:
            lanes = [i for i, lst in enumerate(per_lane) if t in lst]
            eout = h.out_edge[t]
            n_out = len(b.edges[eout].lanes)
            cls = turn_class(h.arms[k], h.arms[t], k == t)
            if cls in ("L", "U"):
                outs = list(range(max(n_out - len(lanes), 0), n_out))
            else:
                outs = list(range(min(len(lanes), n_out)))
            if len(outs) < len(lanes):
                raise ValueError(f"{b.name}: 臂 {k}→{t} 有 {len(lanes)} 条进口车道并入 {n_out} 条出口车道（夹具不合理）")
            for i, j in zip(lanes, outs):
                a = b.lanes[b.edges[ein].lanes[i]].points[-1, :2]
                bb = b.lanes[b.edges[eout].lanes[j]].points[0, :2]
                ten = np.linalg.norm(bb - a) * tension_scale
                if cls == "U":
                    ten = max(ten, 8.0) * 1.6
                b.connect(b.edges[ein].lanes[i], b.edges[eout].lanes[j], ten)
    fill_unfed(b, h)


def fill_unfed(b: GraphBuilder, h: JunctionHandles) -> None:
    """为路口出口 edge 中无源的车道补连（单调保序扇出）：取向相邻出口车道 j-1 供车的
    进口车道中序号最大者（j=0 时取向 j+1 供车者中序号最小者）增连 → j。"""
    for t, eout in h.out_edge.items():
        outs = b.edges[eout].lanes
        for j, lj in enumerate(outs):
            if any(c.to_lane == lj for c in b.conns):
                continue
            nb = outs[j - 1] if j > 0 else (outs[j + 1] if len(outs) > 1 else None)
            feeders = [c.from_lane for c in b.conns if c.to_lane == nb]
            if not feeders:
                continue
            idx = {lid: i for e in b.edges for i, lid in enumerate(e.lanes)}
            li = max(feeders, key=lambda x: idx[x]) if j > 0 else min(feeders, key=lambda x: idx[x])
            b.connect(li, lj, np.linalg.norm(b.lanes[lj].points[0, :2] - b.lanes[li].points[-1, :2]))


def arm_crosswalks(b: GraphBuilder, h: JunctionHandles, two_stage: set[int] | None = None) -> None:
    """在各臂停止线内侧布置人行横道（two_stage 中的臂拆成以中央分隔为界的两段）。"""
    c = np.asarray(b.nodes[h.core].pos[:2])
    for k, a in enumerate(h.arms):
        if not a.crosswalk:
            continue
        es = [e for e in (h.in_edge.get(k), h.out_edge.get(k)) if e is not None]
        if not es:
            continue
        u = unit(a.angle)
        nrm = rot90(u)
        rc = a.r_stop - a.cw_width / 2 - 0.5
        base = c + rc * u + a.lateral * nrm
        w_in = sum(a.w_in) if k in h.in_edge else 0.0
        w_out = sum(a.w_out) if k in h.out_edge else 0.0
        p_in = base + nrm * (a.median / 2 + w_in)
        p_out = base - nrm * (a.median / 2 + w_out)
        if two_stage and k in two_stage and k in h.in_edge and k in h.out_edge:
            b.crosswalk(h.core, [h.in_edge[k]], p_in, base + nrm * a.median / 2, a.cw_width)
            b.crosswalk(h.core, [h.out_edge[k]], p_out, base - nrm * a.median / 2, a.cw_width)
        else:
            b.crosswalk(h.core, es, p_out, p_in, a.cw_width)


def turn_angle(a_in: Arm, a_out: Arm) -> float:
    d_in = -unit(a_in.angle)
    d_out = unit(a_out.angle)
    return math.degrees(math.atan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], float(d_in @ d_out)))


def default_moves(arms: list[Arm], uturn: set[int] | None = None) -> dict[int, list[list[int]]]:
    """同进口单调无交叉的车道-去向分配：去向按偏转角自右向左排序（掉头最左），
    车道 i 承担连续去向区间 [⌊i·k/m⌋, ⌈(i+1)·k/m⌉)，保证同进口转向互不交叉。"""
    moves = {}
    n = len(arms)
    for k, a in enumerate(arms):
        if not a.w_in:
            continue
        tg = sorted((t for t in range(n) if t != k and arms[t].w_out), key=lambda t: turn_angle(a, arms[t]))
        if uturn and k in uturn and a.w_out:
            tg.append(k)
        m, kk = len(a.w_in), len(tg)
        moves[k] = [tg[(i * kk) // m: max(-(-(i + 1) * kk // m), (i * kk) // m + 1)] for i in range(m)]
    return moves


# ====================================================================== 标准图
def g_cross4_basic():
    b = GraphBuilder("cross4_basic")
    arms = [Arm(a) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_cross4_median_3lane():
    b = GraphBuilder("cross4_median_3lane")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25], w_out=[3.5, 3.25, 3.25], median=2.0, r_stop=22) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_t_junction():
    b = GraphBuilder("t_junction")
    arms = [Arm(0), Arm(180), Arm(270, w_in=[3.5, 3.25], w_out=[3.5])]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_y_junction():
    b = GraphBuilder("y_junction")
    arms = [Arm(0, r_stop=18), Arm(140, r_stop=20), Arm(225, r_stop=20)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_skew4():
    b = GraphBuilder("skew4")
    arms = [Arm(0, r_stop=20), Arm(65, r_stop=22), Arm(180, r_stop=20), Arm(245, r_stop=22)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_five_arm():
    b = GraphBuilder("five_arm")
    arms = [Arm(a, r_stop=24) for a in (0, 72, 144, 216, 288)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_six_arm_median():
    b = GraphBuilder("six_arm_median")
    arms = [Arm(a, w_in=[3.5, 3.25], w_out=[3.5, 3.25], median=1.5, r_stop=30) for a in (0, 60, 120, 180, 240, 300)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_offset_4arm():
    b = GraphBuilder("offset_4arm")
    arms = [Arm(0, lateral=3.0, r_stop=18), Arm(90), Arm(180, lateral=3.0, r_stop=18), Arm(270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_curved_approach():
    b = GraphBuilder("curved_approach")
    arms = [Arm(0, bend=12.0), Arm(90, bend=-10.0), Arm(180, bend=8.0), Arm(270, bend=-14.0)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_width_asym():
    b = GraphBuilder("width_asym")
    arms = [Arm(0, w_in=[2.4, 2.8, 3.0, 3.25], w_out=[3.75, 3.5]),
            Arm(90, w_in=[4.0], w_out=[3.0, 2.6, 2.3]),
            Arm(180, w_in=[3.25, 3.0, 2.8], w_out=[3.5, 3.5], median=1.0),
            Arm(270, w_in=[3.5, 3.5], w_out=[2.9, 3.0])]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_oneway_arms():
    b = GraphBuilder("oneway_arms")
    arms = [Arm(0, w_in=[3.5, 3.5], w_out=[]), Arm(90, w_in=[], w_out=[3.5, 3.25]),
            Arm(180, w_in=[3.5, 3.25]), Arm(270, w_in=[3.5], w_out=[3.5])]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_uturn_at_junction():
    b = GraphBuilder("uturn_at_junction")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25], w_out=[3.5, 3.25, 3.25], median=4.0, r_stop=22) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms, uturn={0, 2}))
    arm_crosswalks(b, h)
    return b.build()


def g_double_left():
    b = GraphBuilder("double_left")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25, 3.25], w_out=[3.5, 3.25, 3.25], median=2.0, r_stop=26) for a in (0, 90, 180, 270)]
    moves = {}
    for k in range(4):
        straight, right, left = (k + 2) % 4, (k + 1) % 4, (k + 3) % 4
        moves[k] = [[straight, right], [straight], [left], [left]]
    h = build_arms(b, arms)
    connect_movements(b, h, moves)
    arm_crosswalks(b, h)
    return b.build()


def g_solid_no_lane_change():
    b = GraphBuilder("solid_no_lane_change")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25], lc_in=[False, False, False]) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_zero_out_edge():
    """某进口 edge 在路口无任何出向连接：必须写 <connection from=.../> 禁止 netconvert 自动补连。"""
    b = GraphBuilder("zero_out_edge")
    arms = [Arm(a) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    moves = default_moves(arms)
    moves.pop(3)
    connect_movements(b, h, moves)
    arm_crosswalks(b, h)
    return b.build()


def g_two_stage_crossing():
    """中央安全岛二次过街：两段人行横道各跨单向 edge，需岛侧辅助锚点。"""
    b = GraphBuilder("two_stage_crossing")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25], w_out=[3.5, 3.25, 3.25], median=4.0, r_stop=24) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h, two_stage={0, 1, 2, 3})
    return b.build()


def _approach_with_taper(b: GraphBuilder, a: Arm, center, core: int, w_up: list[float], w_dn: list[float],
                         s_taper: tuple[float, float], pairs: list[tuple[int, int]]) -> int:
    """进口方向：端口 →(上游 k 车道)→ split_merge →(下游 k' 车道)→ 路口核心；返回下游 edge。"""
    c = np.asarray(center, float)
    u = unit(a.angle)
    s_near, s_far = s_taper
    ax_up = c + a.axis(s_far, a.length)
    lb_up = offset_polyline(np.c_[ax_up, np.zeros(len(ax_up))], a.median / 2)[::-1, :2]
    ax_dn = c + a.axis(a.r_stop, s_near)
    lb_dn = offset_polyline(np.c_[ax_dn, np.zeros(len(ax_dn))], a.median / 2)[::-1, :2]
    port = b.node("roi_entry_port", lb_up[0] + rot90(-u) * (-sum(w_up) / 2))
    sm = b.node("split_merge", c + u * (s_near + s_far) / 2 + rot90(u) * (a.median / 2 + sum(w_dn) / 2))
    e_up = b.edge(port, sm, lb_up, w_up, stop_line_end=False, roles=["mid_segment"] * len(w_up))
    e_dn = b.edge(sm, core, lb_dn, w_dn, stop_line_end=True)
    b.connect_edges(e_up, e_dn, pairs)
    return e_dn


def g_flare_widening():
    """进口展宽：2 → 3 车道（右侧新增右转专用道），split_merge + K_conn 平滑过渡。"""
    b = GraphBuilder("flare_widening")
    arms = [Arm(a, r_stop=18) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms, in_skip={0})
    a0 = arms[0]
    a0.w_in = [3.25, 3.5, 3.5]
    h.in_edge[0] = _approach_with_taper(b, a0, (0, 0), h.core, [3.5, 3.5], [3.25, 3.5, 3.5],
                                        (45.0, 70.0), [(0, 0), (0, 1), (1, 2)])
    moves = default_moves(arms)
    moves[0] = [[1], [2], [2, 3]]
    connect_movements(b, h, moves)
    arm_crosswalks(b, h)
    return b.build()


def g_taper_reduction():
    """出口缩减：3 → 2 车道（最右车道合流），split_merge 合流过渡。"""
    b = GraphBuilder("taper_reduction")
    arms = [Arm(a, r_stop=18) for a in (0, 90, 180, 270)]
    arms[2].w_out = [3.5, 3.5, 3.5]
    h = build_arms(b, arms, out_skip={2})
    a2 = arms[2]
    u = unit(a2.angle)
    ax_near = a2.axis(a2.r_stop, 45.0)
    ax_far = a2.axis(70.0, a2.length)
    sm = b.node("split_merge", u * 57.5 + rot90(u) * (-5.25))
    port = b.node("roi_exit_port", ax_far[-1] + rot90(u) * (-3.5))
    e_up = b.edge(h.core, sm, ax_near, [3.5, 3.5, 3.5])
    e_dn = b.edge(sm, port, ax_far, [3.5, 3.5], roles=["mid_segment"] * 2)
    b.connect_edges(e_up, e_dn, [(0, 0), (1, 0), (2, 1)])
    h.out_edge[2] = e_up
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_bus_lane():
    """公交专用道：进口最右车道 bus_only，自 road_continuation 断面起始。"""
    b = GraphBuilder("bus_lane")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25], r_stop=20) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms, in_skip={0})
    a0 = arms[0]
    a0.perm_in = ["bus_only", "general", "general"]
    c = np.zeros(2)
    u = unit(a0.angle)
    ax_up = a0.axis(52.0, a0.length)
    lb_up = offset_polyline(np.c_[ax_up, np.zeros(len(ax_up))], 0.0)[::-1, :2]
    ax_dn = a0.axis(a0.r_stop, 50.0)
    lb_dn = offset_polyline(np.c_[ax_dn, np.zeros(len(ax_dn))], 0.0)[::-1, :2]
    port = b.node("roi_entry_port", lb_up[0] + rot90(-u) * (-5.0))
    rc = b.node("road_continuation", c + u * 51.0 + rot90(u) * 5.0)
    e_up = b.edge(port, rc, lb_up, [3.5, 3.25, 3.25], roles=["mid_segment"] * 3)
    e_dn = b.edge(rc, h.core, lb_dn, [3.5, 3.25, 3.25], permissions=["bus_only", "general", "general"],
                  stop_line_end=True)
    b.connect_edges(e_up, e_dn, [(0, 0), (1, 1), (2, 2)])
    h.in_edge[0] = e_dn
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def g_variable_reversible():
    b = GraphBuilder("variable_reversible")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25], w_out=[3.5, 3.25, 3.25]) for a in (0, 90, 180, 270)]
    arms[0].perm_in = ["general", "variable", "general"]
    arms[2].perm_in = ["general", "general", "reversible"]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms))
    arm_crosswalks(b, h)
    return b.build()


def _straight_corridor(b: GraphBuilder, x0: float, x1: float, w_eb, w_wb, median: float):
    """东西向双向路段：返回 (东行 edge 左边界生成器, 西行 ...)。"""
    def eb(xa, xb):
        return np.array([[xa, -median / 2], [xb, -median / 2]])

    def wb(xa, xb):
        return np.array([[xb, median / 2], [xa, median / 2]])
    return eb, wb


def g_midblock_uturn():
    """路段中央分隔带掉头开口：midblock_node 切分双向车道，掉头 K_conn 跨开口。"""
    b = GraphBuilder("midblock_uturn")
    med = 3.0
    w = [3.5, 3.25]
    eb, wb = _straight_corridor(b, -90, 90, w, w, med)
    pw_in = b.node("roi_entry_port", (-90, -med / 2 - 3.4))
    pe_out = b.node("roi_exit_port", (90, -med / 2 - 3.4))
    pe_in = b.node("roi_entry_port", (90, med / 2 + 3.4))
    pw_out = b.node("roi_exit_port", (-90, med / 2 + 3.4))
    m = b.node("midblock_node", (0, 0))
    e1 = b.edge(pw_in, m, eb(-90, -6), w, roles=["mid_segment", "uturn_bay"])
    e2 = b.edge(m, pe_out, eb(6, 90), w, roles=["mid_segment"] * 2)
    e3 = b.edge(pe_in, m, wb(6, 90), w, roles=["mid_segment", "uturn_bay"])
    e4 = b.edge(m, pw_out, wb(-90, -6), w, roles=["mid_segment"] * 2)
    b.connect_edges(e1, e2, [(0, 0), (1, 1)])
    b.connect_edges(e3, e4, [(0, 0), (1, 1)])
    b.connect(b.lanes_of(e1)[1], b.lanes_of(e4)[1], tension=14.0)
    b.connect(b.lanes_of(e3)[1], b.lanes_of(e2)[1], tension=14.0)
    return b.build()


def g_midblock_two_stage():
    """路段中段人行横道 + 中央安全岛二次过街（midblock_node 绑定）。"""
    b = GraphBuilder("midblock_two_stage")
    med = 3.0
    w = [3.5, 3.25]
    eb, wb = _straight_corridor(b, -90, 90, w, w, med)
    pw_in = b.node("roi_entry_port", (-90, -med / 2 - 3.4))
    pe_out = b.node("roi_exit_port", (90, -med / 2 - 3.4))
    pe_in = b.node("roi_entry_port", (90, med / 2 + 3.4))
    pw_out = b.node("roi_exit_port", (-90, med / 2 + 3.4))
    m = b.node("midblock_node", (0, 0))
    e1 = b.edge(pw_in, m, eb(-90, -4), w, roles=["mid_segment"] * 2, stop_line_end=True)
    e2 = b.edge(m, pe_out, eb(4, 90), w, roles=["mid_segment"] * 2)
    e3 = b.edge(pe_in, m, wb(4, 90), w, roles=["mid_segment"] * 2, stop_line_end=True)
    e4 = b.edge(m, pw_out, wb(-90, -4), w, roles=["mid_segment"] * 2)
    b.connect_edges(e1, e2, [(0, 0), (1, 1)])
    b.connect_edges(e3, e4, [(0, 0), (1, 1)])
    W = sum(w)
    b.crosswalk(m, [e1], (-2.0, -med / 2 - W), (-2.0, -med / 2), 3.5)
    b.crosswalk(m, [e4], (-2.0, med / 2 + W), (-2.0, med / 2), 3.5)
    return b.build()


def g_midblock_crosswalk():
    """无中央分隔的路段中段人行横道（双向一次过街）。"""
    b = GraphBuilder("midblock_crosswalk")
    w = [3.5, 3.25]
    eb, wb = _straight_corridor(b, -90, 90, w, w, 0.0)
    pw_in = b.node("roi_entry_port", (-90, -3.4))
    pe_out = b.node("roi_exit_port", (90, -3.4))
    pe_in = b.node("roi_entry_port", (90, 3.4))
    pw_out = b.node("roi_exit_port", (-90, 3.4))
    m = b.node("midblock_node", (0, 0))
    e1 = b.edge(pw_in, m, eb(-90, -3), w, roles=["mid_segment"] * 2, stop_line_end=True)
    e2 = b.edge(m, pe_out, eb(3, 90), w, roles=["mid_segment"] * 2)
    e3 = b.edge(pe_in, m, wb(3, 90), w, roles=["mid_segment"] * 2, stop_line_end=True)
    e4 = b.edge(m, pw_out, wb(-90, -3), w, roles=["mid_segment"] * 2)
    b.connect_edges(e1, e2, [(0, 0), (1, 1)])
    b.connect_edges(e3, e4, [(0, 0), (1, 1)])
    W = sum(w)
    b.crosswalk(m, [e1, e4], (-1.5, -W), (-1.5, W), 3.0)
    return b.build()


def g_slip_right_turn():
    """渠化右转：进口上游 split_merge 分流 → 渠化右转道（中段 midblock 斑马线，单向跨越）→ 出口下游合流。"""
    b = GraphBuilder("slip_right_turn")
    arms = [Arm(a, r_stop=20) for a in (0, 90, 180, 270)]
    c = np.zeros(2)
    h = build_arms(b, arms, in_skip={0}, out_skip={1})
    a0, a1 = arms[0], arms[1]
    u0, u1 = unit(a0.angle), unit(a1.angle)
    ws = 3.5
    # 东臂进口：上游 2 车道 → split_merge → 下游 2 车道（主路）+ 渠化右转道
    s_near, s_far = 40.0, 58.0
    ax_up = c + a0.axis(s_far, a0.length)
    lb_up = ax_up[::-1]
    ax_dn = c + a0.axis(a0.r_stop, s_near)
    lb_dn = ax_dn[::-1]
    port_in = b.node("roi_entry_port", lb_up[0] + rot90(-u0) * (-3.5))
    sm_div = b.node("split_merge", u0 * (s_near + s_far) / 2 + rot90(u0) * 7.0)
    e_up = b.edge(port_in, sm_div, lb_up, [3.5, 3.5], roles=["mid_segment"] * 2)
    e_dn = b.edge(sm_div, h.core, lb_dn, [3.5, 3.5], stop_line_end=True)
    b.connect_edges(e_up, e_dn, [(0, 0), (1, 1)])
    h.in_edge[0] = e_dn
    # 北臂出口：路口核心 → split_merge(合流) → 端口
    ax_o1 = c + a1.axis(a1.r_stop, s_near)
    ax_o2 = c + a1.axis(s_far, a1.length)
    sm_mrg = b.node("split_merge", u1 * (s_near + s_far) / 2 + rot90(u1) * -3.5)
    port_out = b.node("roi_exit_port", ax_o2[-1] + rot90(u1) * -3.5)
    o_dn = b.edge(h.core, sm_mrg, ax_o1, [3.5, 3.5])
    o_up = b.edge(sm_mrg, port_out, ax_o2, [3.5, 3.5])
    b.connect_edges(o_dn, o_up, [(0, 0), (1, 1)])
    h.out_edge[1] = o_dn
    # 渠化右转道中心线：从东臂分流断面外侧到北臂合流断面外侧
    p0 = u0 * s_near + rot90(u0) * (7.0 + ws / 2)
    p1 = u1 * s_near + rot90(u1) * -(7.0 + ws / 2)
    full = hermite_curve(p0, -u0, p1, u1, 200, tension=55.0)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
    half = L[-1] / 2
    gap = 2.0
    i0 = int(np.searchsorted(L, half - gap))
    i1 = int(np.searchsorted(L, half + gap))
    mid = full[int(np.searchsorted(L, half))]
    tmid = full[i1] - full[i0]
    tmid /= np.linalg.norm(tmid)
    mb = b.node("midblock_node", mid)
    s1 = b.edge(sm_div, mb, offset_polyline(np.c_[full[: i0 + 1], np.zeros(i0 + 1)], ws / 2)[:, :2], [ws],
                roles=["slip_turn"], stop_line_end=True)
    s2 = b.edge(mb, sm_mrg, offset_polyline(np.c_[full[i1:], np.zeros(len(full) - i1)], ws / 2)[:, :2], [ws],
                roles=["slip_turn"])
    # 分流 / 合流过渡（从相邻车道平移一条车道宽度）
    b.connect(b.lanes_of(e_up)[0], b.lanes_of(s1)[0])
    b.connect(b.lanes_of(s1)[0], b.lanes_of(s2)[0])
    b.connect(b.lanes_of(s2)[0], b.lanes_of(o_up)[0])
    nrm = rot90(tmid)
    b.crosswalk(mb, [s1], mid - nrm * ws / 2, mid + nrm * ws / 2, 3.0)
    moves = default_moves(arms)
    moves[0] = [[2], [2, 3]]        # 东进口主路：直行 + 左转（右转走渠化道）
    connect_movements(b, h, moves)
    arm_crosswalks(b, h)
    return b.build()


def g_long_left_waiting():
    """大路口长左转（左转待转区走廊被吸收进左转 K_conn）。"""
    b = GraphBuilder("long_left_waiting")
    arms = [Arm(a, w_in=[3.5, 3.25, 3.25, 3.0], w_out=[3.5, 3.25, 3.25], median=3.0, r_stop=34) for a in (0, 90, 180, 270)]
    h = build_arms(b, arms)
    connect_movements(b, h, default_moves(arms), tension_scale=1.15)
    arm_crosswalks(b, h)
    return b.build()


def g_corridor_two_junctions(spacing: float = 300.0):
    """双路口走廊（跨度 > 204.8 m 单窗上限，用作超大选区分块端口拼接的真值图）。"""
    b = GraphBuilder("corridor_two_junctions")
    w = [3.5, 3.25]
    arms1 = [Arm(0, w_in=w, w_out=w, crosswalk=False), Arm(90), Arm(180), Arm(270)]
    arms2 = [Arm(0), Arm(90), Arm(180, w_in=w, w_out=w, crosswalk=False), Arm(270)]
    h1 = build_arms(b, arms1, center=(0, 0), in_skip={0}, out_skip={0})
    h2 = build_arms(b, arms2, center=(spacing, 0), in_skip={2}, out_skip={2})
    r = arms1[0].r_stop
    eb = b.edge(h1.core, h2.core, np.array([[r, 0.0], [spacing - r, 0.0]]), w, roles=["mid_segment", "approach"],
                stop_line_end=True)
    wb = b.edge(h2.core, h1.core, np.array([[spacing - r, 0.0], [r, 0.0]]), w, roles=["mid_segment", "approach"],
                stop_line_end=True)
    h1.out_edge[0], h1.in_edge[0] = eb, wb
    h2.in_edge[2], h2.out_edge[2] = eb, wb
    connect_movements(b, h1, default_moves(arms1))
    connect_movements(b, h2, default_moves(arms2))
    arm_crosswalks(b, h1)
    arm_crosswalks(b, h2)
    return b.build()


STANDARD_GRAPHS: dict[str, Callable[[], LaneGraph]] = {
    "corridor_two_junctions": g_corridor_two_junctions,
    "cross4_basic": g_cross4_basic,
    "cross4_median_3lane": g_cross4_median_3lane,
    "t_junction": g_t_junction,
    "y_junction": g_y_junction,
    "skew4": g_skew4,
    "five_arm": g_five_arm,
    "six_arm_median": g_six_arm_median,
    "offset_4arm": g_offset_4arm,
    "curved_approach": g_curved_approach,
    "width_asym": g_width_asym,
    "oneway_arms": g_oneway_arms,
    "uturn_at_junction": g_uturn_at_junction,
    "double_left": g_double_left,
    "solid_no_lane_change": g_solid_no_lane_change,
    "zero_out_edge": g_zero_out_edge,
    "two_stage_crossing": g_two_stage_crossing,
    "flare_widening": g_flare_widening,
    "taper_reduction": g_taper_reduction,
    "bus_lane": g_bus_lane,
    "variable_reversible": g_variable_reversible,
    "midblock_uturn": g_midblock_uturn,
    "midblock_two_stage": g_midblock_two_stage,
    "midblock_crosswalk": g_midblock_crosswalk,
    "slip_right_turn": g_slip_right_turn,
    "long_left_waiting": g_long_left_waiting,
}


def build_all() -> dict[str, LaneGraph]:
    return {k: f() for k, f in STANDARD_GRAPHS.items()}
