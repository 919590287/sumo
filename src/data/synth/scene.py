"""参数化合成场景生成器：规范化分段协议下的合成真值图 G*（docs/DESIGN.md §5 Layer 1-1/2）。

一个场景 = 单个平面交叉口（3–6 岔）| 双路口走廊 | 无路口路段，产出：
- ``graph``：严格遵守规范化真值分段协议的 LaneGraph（场景局部米制坐标，origin = 随机 UTM 偏移）：
  节点只在 5 类视觉可观测事件处创建——(a) ROI / 场景边界端口；(b) 交叉口核心停止线断面
  （junction_core，pos = 接入横断面角点质心，polygon = 精确经过全部接入横断面角点的 32 点闭合轮廓）；
  (c) 展宽 / 缩减渐变段（split_merge，单节点 + K_conn 桥接，渐变段内不放 L）；
  (d) 实虚禁变道切换 / 公交专用道起止断面（road_continuation）；
  (e) 路段掉头开口与非核心区人行横道（midblock_node）。
  K_conn* 由 ``kinematic.TrajectoryGenerator`` 生成（最小转弯半径、避障、并列保序）。
- ``marks``：供百度风格渲染器使用的高层标线 / 设施语义（与 G* 几何严格一致）；
- ``neg_regions``：显式负样本区域（无标线支路、停车、非机动车道、港湾式公交站），只进输入不进真值。

场景参数全部来自 configs/synth.yaml，只服务于训练数据分布，不参与推理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.common.config import load_config
from src.common.geom import hermite_curve, offset_polyline, resample_polyline, tangents
from src.data.standard_graphs import GraphBuilder, pad_ring
from src.data.synth.kinematic import KinematicFail, Obstacles, TrajectoryGenerator
from src.graph.lane_graph import K_CONN, POLY_K, Connection, LaneGraph

TURN_SETS = {}          # frozenset(turns) → 分类代码（首个匹配）
for _code, _turns in load_config("data")["arrow_turns"].items():
    if _code == "uturn_allowed_symbol":
        continue
    TURN_SETS.setdefault(frozenset(_turns), _code)


class SceneReject(RuntimeError):
    pass


def unit(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def _xyz(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return np.c_[p[:, :2], np.zeros(len(p))]


# ====================================================================== 数据结构
@dataclass
class Event:
    kind: str              # attr_solid | attr_bus | flare_right | flare_left | reduce_right | slip | midblock
    lo: float
    hi: float
    k: int = 0             # 展宽 / 缩减车道数
    node: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Sec:
    s_a: float             # 行驶方向起点轴向坐标
    s_b: float
    widths: list[float]
    left_off: float
    perms: list[str]
    lc: bool
    roles: list[str]
    stop_end: bool
    edge: int | None = None


@dataclass
class Arm:
    idx: int
    angle: float
    center: np.ndarray
    length: float
    r_stop: float = 0.0
    bend: float = 0.0
    lateral: float = 0.0
    w_in: list[float] = field(default_factory=list)     # 核心侧（展宽后）进口车道宽，右→左
    w_out: list[float] = field(default_factory=list)
    median_type: str = "double_yellow"
    median: float = 0.4
    in_events: list[Event] = field(default_factory=list)
    out_events: list[Event] = field(default_factory=list)
    perms_in: list[str] = field(default_factory=list)
    near: str = "core"                       # core | segment（路段两端皆端口）
    far_in_node: int | None = None           # 走廊连接段：进口远端为另一路口核心
    skip_out: bool = False                   # 走廊连接段：出口由对端臂的进口提供
    link_partner: "Arm | None" = None
    is_link: bool = False
    s_min: float = 0.0                       # 轴向坐标下限（路段场景为 -L/2）
    cw: dict | None = None
    edge_line: bool = True
    in_secs: list[Sec] = field(default_factory=list)
    out_secs: list[Sec] = field(default_factory=list)
    solid_core: bool = False
    bus: bool = False

    @property
    def u(self) -> np.ndarray:
        return unit(self.angle)

    @property
    def n(self) -> np.ndarray:
        return rot90(self.u)

    def lat(self, s):
        s = np.asarray(s, float)
        if self.near == "segment":
            half = max(self.length, 1e-6)
            return self.lateral + self.bend * (s / half) ** 2
        span = max(self.length - self.r_stop, 1e-6)
        return self.lateral + self.bend * np.clip((s - self.r_stop) / span, 0, None) ** 2

    def axis(self, s0: float, s1: float, k: int | None = None) -> np.ndarray:
        k = k or max(8, int(abs(s1 - s0) / 1.0) + 2)
        s = np.linspace(s0, s1, k)
        return self.center + s[:, None] * self.u + self.lat(s)[:, None] * self.n

    def point(self, s: float, lateral: float) -> np.ndarray:
        """轴向坐标 s 处、相对轴线左侧（+n）横向 lateral 的点（使用局部法向）。"""
        a = self.axis(s - 0.5, s + 0.5, 3)
        t = a[2] - a[0]
        t /= np.linalg.norm(t)
        return a[1] + rot90(t) * lateral

    def tangent(self, s: float) -> np.ndarray:
        a = self.axis(s - 0.5, s + 0.5, 3)
        t = a[2] - a[0]
        return t / np.linalg.norm(t)


@dataclass
class Scene:
    kind: str
    graph: LaneGraph
    marks: list[dict]
    neg_regions: list[Polygon]
    origin: tuple[float, float]
    seed: int
    meta: dict = field(default_factory=dict)


# ====================================================================== 生成器
class SceneGenerator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config("synth")
        self.traj = TrajectoryGenerator(self.cfg["kinematic"])
        tc = self.cfg["turn_class_deg"]
        self.tc = (tc["straight"], tc["slight"], tc["uturn"])

    # ------------------------------------------------------------------ 随机工具
    def _u(self, rng, a):
        return float(rng.uniform(a[0], a[1]))

    def _count(self, rng, spec) -> int:
        w = np.asarray(spec["weights"], float)
        return int(rng.choice(np.arange(spec["min"], spec["min"] + len(w)), p=w / w.sum()))

    def _widths(self, rng, n: int) -> list[float]:
        lo, hi = self.cfg["lane_width"]
        base = float(rng.triangular(max(lo, 2.6), self.cfg["lane_width_mode"], min(hi, 3.9)))
        w = np.clip(base + rng.normal(0, 0.12, n), lo, hi)
        if n > 1 and rng.random() < 0.1:
            w[rng.integers(n)] = self._u(rng, (lo, hi))
        return [round(float(x), 2) for x in w]

    def turn_class(self, psi_deg: float, same_arm: bool) -> str:
        if same_arm:
            return "U"
        a = abs(psi_deg)
        s, sl, ut = self.tc
        if a <= s:
            return "S"
        if a <= sl:
            return "FL" if psi_deg > 0 else "FR"
        if a <= ut:
            return "L" if psi_deg > 0 else "R"
        return "U"

    # ------------------------------------------------------------------ 主入口
    def generate(self, seed: int) -> Scene:
        last = None
        for attempt in range(self.cfg["max_attempts_per_sample"]):
            rng = np.random.default_rng([seed, attempt])
            try:
                sc = self._generate_once(rng)
                sc.seed = seed
                sc.meta["attempt"] = attempt
                return sc
            except (SceneReject, KinematicFail, ValueError) as e:
                last = e
        raise SceneReject(f"种子 {seed} 连续 {self.cfg['max_attempts_per_sample']} 次生成失败：{last}")

    def _generate_once(self, rng) -> Scene:
        c = self.cfg
        mix = c["scene_mix"]
        kind = rng.choice(list(mix), p=np.asarray(list(mix.values())) / sum(mix.values()))
        ox = self._u(rng, c["utm_origin_range"]["x"])
        oy = self._u(rng, c["utm_origin_range"]["y"])
        st = _State(self, rng, kind)
        if kind == "junction":
            st.build_junction(np.zeros(2), self._sample_arms(rng, np.zeros(2)))
        elif kind == "corridor":
            st.build_corridor()
        else:
            st.build_segment()
        g, marks, negs = st.finish()
        return Scene(str(kind), LaneGraph(g.nodes, g.lanes, g.edges, g.connections, g.crosswalks,
                                          origin=(math.floor(ox), math.floor(oy)), meta={"kind": str(kind)}),
                     marks, negs, (math.floor(ox), math.floor(oy)), 0, {"kind": str(kind)})

    # ------------------------------------------------------------------ 臂采样
    def _sample_arms(self, rng, center, n_arms: int | None = None, fixed: list[float] | None = None,
                     max_len: float | None = None) -> list[Arm]:
        c = self.cfg
        n = n_arms or self._count(rng, c["arms"])
        lo, hi = c["arm_gap_deg"]
        templates = {3: [90.0, 90.0, 180.0], 4: [90.0, 90.0, 90.0, 90.0]}
        for _ in range(200):
            if n in templates and rng.random() < c["arm_template_prob"]:
                base = np.roll(np.asarray(templates[n]), int(rng.integers(n)))
                gaps = base + rng.normal(0, c["arm_template_jitter_deg"], n)
                gaps = gaps / gaps.sum() * 360.0
            else:
                gaps = rng.dirichlet(np.ones(n) * c["arm_gap_concentration"]) * 360.0
            if gaps.min() >= lo and gaps.max() <= hi:
                break
        else:
            raise SceneReject("臂夹角采样失败")
        a0 = fixed[0] if fixed else float(rng.uniform(0, 360))
        angles = (a0 + np.concatenate([[0], np.cumsum(gaps[:-1])])) % 360
        arms = []
        for k, ang in enumerate(angles):
            L = self._u(rng, c["arm_length"])
            if max_len:
                L = min(L, max_len)
            a = Arm(k, float(ang), np.asarray(center, float), L)
            n_in = self._count(rng, c["lanes_in"])
            n_out = self._count(rng, c["lanes_out"])
            if rng.random() < c["oneway_prob"]:
                if rng.random() < 0.5:
                    n_in = 0
                else:
                    n_out = 0
            a.w_in = self._widths(rng, n_in) if n_in else []
            a.w_out = self._widths(rng, n_out) if n_out else []
            if rng.random() < c["bend_prob"]:
                a.bend = self._u(rng, c["bend_m"])
            if rng.random() < c["lateral_prob"]:
                a.lateral = self._u(rng, c["lateral_m"])
            self._sample_median(rng, a)
            a.edge_line = bool(rng.random() < c["render"]["edge_line_prob"])
            arms.append(a)
        if sum(bool(a.w_in) for a in arms) < 1 or sum(bool(a.w_out) for a in arms) < 2:
            raise SceneReject("进出口臂不足")
        return arms

    def _sample_median(self, rng, a: Arm) -> None:
        m = self.cfg["median"]
        types = list(m["types"])
        p = np.asarray(list(m["types"].values()), float)
        a.median_type = str(rng.choice(types, p=p / p.sum()))
        if not a.w_in or not a.w_out:
            a.median_type = "none"
            a.median = 0.0
        elif a.median_type == "island":
            a.median = self._u(rng, m["island_width"])
        elif a.median_type == "barrier":
            a.median = self._u(rng, m["barrier_width"])
        else:
            a.median = self._u(rng, m["painted_width"])


# ====================================================================== 构建状态
class _State:
    def __init__(self, gen: SceneGenerator, rng, kind: str):
        self.gen = gen
        self.cfg = gen.cfg
        self.rng = rng
        self.kind = kind
        self.b = GraphBuilder("synth")
        self.marks: list[dict] = []
        self.negs: list[Polygon] = []
        self.islands: list[np.ndarray] = []
        self.channels: list[np.ndarray] = []
        self.dense_conn: dict[tuple[int, int], np.ndarray] = {}
        self.junctions: list[dict] = []
        self.arms_all: list[Arm] = []
        self.lane_edge: dict[int, int] = {}

    # ------------------------------------------------------------------ 公共
    def u(self, a):
        return self.gen._u(self.rng, a)

    def node(self, t: str) -> int:
        return self.b.node(t, np.zeros(2))

    def set_pos(self, nid: int, p) -> None:
        self.b.nodes[nid].pos = np.r_[np.asarray(p, float)[:2], 0.0]

    # ------------------------------------------------------------------ 路口
    def build_junction(self, center, arms: list[Arm], compute_r: bool = True) -> dict:
        rng, c = self.rng, self.cfg
        if compute_r:
            self._r_stop(arms)
        for a in arms:
            if a.r_stop >= a.length - 35:
                if a.is_link or not compute_r:
                    raise SceneReject("引道过短")
                a.length = a.r_stop + self.u((40.0, 80.0))
        core = self.node("junction_core")
        J = {"core": core, "arms": arms, "center": np.asarray(center, float), "in_edge": {}, "out_edge": {},
             "slip": {}}
        self.junctions.append(J)
        self.arms_all += arms
        for a in arms:
            self._plan_events(a, J)
        self._plan_slips(J)
        return J

    def _r_stop(self, arms: list[Arm]) -> None:
        c = self.cfg
        n = len(arms)
        order = sorted(range(n), key=lambda k: arms[k].angle)
        for pos, k in enumerate(order):
            a = arms[k]
            ccw = arms[order[(pos + 1) % n]]
            cw = arms[order[(pos - 1) % n]]
            e_plus = (a.median / 2 + sum(a.w_in)) + max(a.lateral, 0) + 0.5
            e_minus = (a.median / 2 + sum(a.w_out)) + max(-a.lateral, 0) + 0.5
            d1 = math.radians((ccw.angle - a.angle) % 360)
            d2 = math.radians((a.angle - cw.angle) % 360)
            b1 = ccw.median / 2 + sum(ccw.w_out) + max(-ccw.lateral, 0) + 0.5
            b2 = cw.median / 2 + sum(cw.w_in) + max(cw.lateral, 0) + 0.5
            # 相邻两臂路幅外缘线交点距中心的轴向距离（夹角 ≥ 160° 时两臂近似对向，无交点约束）
            x1 = (b1 + e_plus * math.cos(d1)) / max(math.sin(d1), 0.2) if d1 < math.radians(160) else 0.0
            x2 = (b2 + e_minus * math.cos(d2)) / max(math.sin(d2), 0.2) if d2 < math.radians(160) else 0.0
            cw_space = 0.0
            if self.rng.random() < c["crosswalk_prob"]:
                w = self.u(c["crosswalk_width"])
                a.cw = {"width": w}
                cw_space = w + 1.5
            a.r_stop = max(x1, x2, 6.0) + cw_space + self.u(c["stop_margin_m"])
        # 右转切线长约束：最右进口车道线与右转去向臂最右出口车道线的交点到两侧停止线的距离
        # 须容纳半径 R_min 的圆弧（合成数据侧的几何可行性预留，避免整场景拒绝重采）
        R = c["kinematic"]["r_min_turn"] * 1.25
        for a in arms:
            for t in arms:
                if t is a or not a.w_in or not t.w_out or self._psi(a, t) > -self.gen.tc[0]:
                    continue
                ea = a.median / 2 + sum(a.w_in) - a.w_in[0] / 2 + a.lateral
                et = t.median / 2 + sum(t.w_out) - t.w_out[0] / 2 - t.lateral
                A = np.stack([a.u, -t.u], 1)
                rhs = -et * t.n - ea * a.n
                if abs(np.linalg.det(A)) < 1e-6:
                    continue
                s1, s2 = np.linalg.solve(A, rhs)
                theta = math.radians(abs(self._psi(a, t)))
                T = R * math.tan(min(theta, math.radians(150)) / 2)
                a.r_stop = max(a.r_stop, s1 + T + 1.0)
                t.r_stop = max(t.r_stop, s2 + T + 1.0)

    # ------------------------------------------------------------------ 事件规划
    def _plan_events(self, a: Arm, J: dict | None) -> None:
        """按“核心侧 → 远端”的顺序放置进口 / 出口事件并生成分段。"""
        rng, c = self.rng, self.cfg
        s_end = a.length - 12.0
        mid = None
        two_way = bool(a.w_in) and bool(a.w_out) and not a.is_link
        if two_way and a.median_type in ("island", "barrier") and rng.random() < c["midblock_uturn_prob"]:
            mid = ("uturn", self.u(c["uturn_opening_m"]))
        elif two_way and rng.random() < c["midblock_crosswalk_prob"] * (1.0 if a.near == "core" else 3.0):
            mid = ("crosswalk", self.u(c["crosswalk_width"]) + 2.0)
        if mid is not None:
            s_m = self.u((max(a.s_min + 40.0, (a.r_stop if a.near == "core" else a.s_min) + 45.0), s_end - 15.0)) \
                if s_end - 15.0 > max(a.s_min + 40.0, a.r_stop + 45.0) else None
            if s_m is None:
                mid = None
            else:
                ev = Event("midblock", s_m - mid[1] / 2, s_m + mid[1] / 2, extra={"type": mid[0]})
                a.in_events.append(ev)
                a.out_events.append(Event("midblock", ev.lo, ev.hi, extra=ev.extra))
                s_end = ev.lo - 10.0
        s0 = a.r_stop if a.near == "core" else a.s_min + 15.0
        # ---------------- 进口（核心侧 → 远端）
        if a.w_in:
            cur = s0
            n_core = len(a.w_in)
            if a.near == "core" and rng.random() < c["solid_prob"]:
                L = self.u(c["solid_len_m"])
                if cur + L + 20 < s_end:
                    a.solid_core = True
                    a.in_events.append(Event("attr_solid", cur + L, cur + L + c["attr_gap_m"]))
                    cur += L + c["attr_gap_m"]
            if a.near == "core" and n_core >= 2 and rng.random() < c["bus_lane_prob"]:
                L = self.u((15.0, 40.0))
                if cur + L + 20 < s_end:
                    a.bus = True
                    a.in_events.append(Event("attr_bus", cur + L, cur + L + c["attr_gap_m"]))
                    cur += L + c["attr_gap_m"]
            if n_core >= 2 and rng.random() < c["flare_prob"]:
                left = (a.median_type == "island" and rng.random() < c["flare_side_left_prob"])
                k = 1
                L = self.u((15.0, 45.0))
                taper = self.u(c["taper_len_m"])
                if left:
                    wl = a.w_in[-1]
                    if a.median / 2 - wl < 0.6:
                        left = False
                if not (left is False and a.bus):
                    if cur + L + taper + 15 < s_end:
                        a.in_events.append(Event("flare_left" if left else "flare_right", cur + L, cur + L + taper, k))
                        cur += L + taper
            a._in_cursor = cur
        # ---------------- 出口
        if a.w_out and not a.skip_out:
            cur = s0
            if len(a.w_out) >= 2 and rng.random() < c["taper_out_prob"]:
                L = self.u((15.0, 40.0))
                taper = self.u(c["taper_len_m"])
                if cur + L + taper + 15 < s_end:
                    a.out_events.append(Event("reduce_right", cur + L, cur + L + taper, 1))
                    cur += L + taper
            a._out_cursor = cur
        a._s_end = s_end
        # 可变 / 潮汐车道（核心段属性，不新增节点）
        a.perms_in = ["general"] * len(a.w_in)
        if a.bus:
            a.perms_in[0] = "bus_only"
        if len(a.w_in) >= 3 and rng.random() < c["variable_lane_prob"]:
            a.perms_in[int(rng.integers(1, len(a.w_in) - 1))] = "variable"
        if len(a.w_in) >= 2 and a.near == "core" and rng.random() < c["reversible_lane_prob"]:
            a.perms_in[-1] = "reversible"

    def _plan_slips(self, J: dict) -> None:
        rng, c = self.rng, self.cfg
        arms = J["arms"]
        n = len(arms)
        order = sorted(range(n), key=lambda k: arms[k].angle)
        for pos, k in enumerate(order):
            a = arms[k]
            t = arms[order[(pos - 1) % n]]           # 顺时针相邻臂 = 右转去向
            if not a.w_in or not t.w_out or a.is_link or t.is_link or a.bus:
                continue
            psi = self._psi(a, t)
            if self.gen.turn_class(psi, False) != "R" or rng.random() >= c["slip_prob"]:
                continue
            if any(e.kind in ("flare_right",) for e in a.in_events):
                continue
            if t.idx in [x[1] for x in J["slip"].values()]:
                continue
            lo_a = a._in_cursor + self.u((8.0, 20.0))
            lo_t = t._out_cursor + self.u((8.0, 20.0))
            div = self.u((15.0, 28.0))
            if lo_a + div + 12 > a._s_end or lo_t + div + 12 > t._s_end:
                continue
            ws = self.gen._widths(rng, 1)[0]
            a.in_events.append(Event("slip", lo_a, lo_a + div, extra={"target": t.idx, "w": ws}))
            t.out_events.append(Event("slip", lo_t, lo_t + div, extra={"source": a.idx, "w": ws}))
            a._in_cursor = lo_a + div
            t._out_cursor = lo_t + div
            J["slip"][a.idx] = (a.idx, t.idx)

    @staticmethod
    def _psi(a_in: Arm, a_out: Arm) -> float:
        d_in = -a_in.u
        d_out = a_out.u
        return math.degrees(math.atan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], float(d_in @ d_out)))

    # ------------------------------------------------------------------ 分段
    def _sections(self, a: Arm) -> None:
        """由事件生成进口 / 出口恒宽分段（核心侧属性向远端逐事件回推）。"""
        s_near = a.r_stop if a.near == "core" else a.s_min
        # ---------------- 进口
        if a.w_in:
            evs = sorted(a.in_events, key=lambda e: e.lo)
            widths = list(a.w_in)
            left_off = a.median / 2
            for e in evs:
                if e.kind == "flare_left":
                    left_off -= sum(widths[-e.k:])
            perms = list(a.perms_in)
            lc = not a.solid_core
            secs = []
            s_lo = s_near
            props = (list(widths), left_off, list(perms), lc)
            for e in evs:
                secs.append((s_lo, e.lo, props))
                w, lo_, p, l_ = props
                w, p = list(w), list(p)
                if e.kind == "attr_solid":
                    l_ = True
                elif e.kind == "attr_bus":
                    p = ["general"] * len(p)
                elif e.kind == "flare_right":
                    w, p = w[e.k:], p[e.k:]
                elif e.kind == "flare_left":
                    lo_ = lo_ + sum(w[-e.k:])
                    w, p = w[:-e.k], p[:-e.k]
                props = (w, lo_, p, l_)
                s_lo = e.hi
            s_far = a.length
            secs.append((s_lo, s_far, props))
            a.in_secs = []
            for k, (lo, hi, (w, lo_, p, l_)) in enumerate(secs):
                if p and any(x in ("variable", "reversible") for x in p) and k > 0:
                    p = ["general" if x in ("variable", "reversible") else x for x in p]
                if hi - lo < 4.0:
                    raise SceneReject("进口分段过短")
                core_sec = (k == 0)
                stop = core_sec and a.near == "core"
                roles = ["approach" if stop else "mid_segment"] * len(w)
                a.in_secs.append(Sec(hi, lo, w, lo_, p, l_, roles, stop))
            a.in_secs = a.in_secs[::-1]          # 行驶方向：远端 → 核心
            # 路段掉头开口 / 中段人行横道：开口上游车道段终点为停止线（人行横道）或掉头车道（掉头）
            for e in evs:
                if e.kind == "midblock":
                    for s in a.in_secs:
                        if abs(s.s_b - e.hi) < 1e-6:
                            if e.extra["type"] == "crosswalk":
                                s.stop_end = True
            if a.near == "segment":
                for s in a.in_secs:
                    s.roles = ["mid_segment"] * len(s.widths)
        # ---------------- 出口
        if a.w_out and not a.skip_out:
            evs = sorted(a.out_events, key=lambda e: e.lo)
            props = (list(a.w_out), a.median / 2)
            s_lo = s_near
            secs = []
            for e in evs:
                secs.append((s_lo, e.lo, props))
                w, lo_ = props
                if e.kind == "reduce_right":
                    w = w[e.k:]
                props = (list(w), lo_)
                s_lo = e.hi
            secs.append((s_lo, a.length, props))
            a.out_secs = []
            for k, (lo, hi, (w, lo_)) in enumerate(secs):
                if hi - lo < 4.0:
                    raise SceneReject("出口分段过短")
                role = "exit" if (k == 0 and a.near == "core") else "mid_segment"
                a.out_secs.append(Sec(lo, hi, w, lo_, ["general"] * len(w), True, [role] * len(w), False))
            for e in evs:
                if e.kind == "midblock" and e.extra["type"] == "crosswalk":
                    for s in a.out_secs:
                        if abs(s.s_b - e.lo) < 1e-6:
                            s.stop_end = True
                            s.roles = ["mid_segment"] * len(s.widths)

    def _lb(self, a: Arm, sec: Sec, inbound: bool) -> np.ndarray:
        pts = a.axis(sec.s_a, sec.s_b)
        return offset_polyline(_xyz(pts), -sec.left_off)[:, :2]

    def _build_edges(self, a: Arm, J: dict | None) -> None:
        """为一条臂建立进口 / 出口 edge、事件节点与事件跨接连接。"""
        b = self.b
        # ---------------- 进口：远端 → 核心
        if a.w_in:
            secs = a.in_secs
            nodes = []
            if a.far_in_node is not None:
                nodes.append(a.far_in_node)
            else:
                nodes.append(self.node("roi_entry_port"))
            evs = sorted(a.in_events, key=lambda e: -e.hi)       # 远端 → 核心
            for e in evs:
                if e.kind == "midblock":
                    if e.node is None:
                        e.node = self.node("midblock_node")
                    nodes.append(e.node)
                elif e.kind.startswith("attr"):
                    e.node = self.node("road_continuation")
                    nodes.append(e.node)
                else:
                    e.node = self.node("split_merge")
                    nodes.append(e.node)
            nodes.append(J["core"] if a.near == "core" else self.node("roi_exit_port"))
            for k, s in enumerate(secs):
                lc = [s.lc] * len(s.widths)
                s.edge = b.edge(nodes[k], nodes[k + 1], self._lb(a, s, True), s.widths, permissions=s.perms,
                                change_left=lc, change_right=lc, roles=s.roles, stop_line_end=s.stop_end)
            a._in_nodes = nodes
            if J is not None and a.near == "core":
                J["in_edge"][a.idx] = secs[-1].edge
        # ---------------- 出口：核心 → 远端
        if a.w_out and not a.skip_out:
            secs = a.out_secs
            nodes = [J["core"] if a.near == "core" else self.node("roi_entry_port")]
            evs = sorted(a.out_events, key=lambda e: e.lo)
            for e in evs:
                if e.kind == "midblock":
                    partner = next(x for x in a.in_events if x.kind == "midblock")
                    e.node = partner.node
                    nodes.append(e.node)
                else:
                    e.node = self.node("split_merge")
                    nodes.append(e.node)
            nodes.append(self.node("roi_exit_port"))
            for k, s in enumerate(secs):
                s.edge = b.edge(nodes[k], nodes[k + 1], self._lb(a, s, False), s.widths,
                                roles=s.roles, stop_line_end=s.stop_end)
            a._out_nodes = nodes
            if J is not None and a.near == "core":
                J["out_edge"][a.idx] = secs[0].edge

    # ------------------------------------------------------------------ 连接
    def conn(self, li: int, lj: int, uturn: bool = False, siblings=None) -> np.ndarray:
        A = self.b.lanes[li].points
        B = self.b.lanes[lj].points
        key = len(self.islands) + 1000 * len(self.channels)
        if getattr(self, "_obs_key", None) != key:
            self._obs = Obstacles(self.islands + self.channels)
            self._obs_key = key
        w = (self.b.lanes[li].width + self.b.lanes[lj].width) / 2
        pts, dense = self.gen.traj.generate(A[-1, :2], tangents(A)[-1], B[0, :2], tangents(B)[0], K_CONN,
                                            uturn=uturn, obstacles=self._obs, siblings=siblings, lane_width=w)
        pts[0], pts[-1] = A[-1, :2], B[0, :2]
        self.b.conns.append(Connection(li, lj, pts))
        self.dense_conn[(li, lj)] = dense
        return dense

    def _event_conns(self, a: Arm) -> None:
        """事件两侧车道段的过渡连接（展宽分流 / 缩减合流 / 属性续接 / 开口续行）。"""
        def pairs_for(kind, n_up, n_dn, k):
            if kind in ("attr_solid", "attr_bus", "midblock", "slip"):
                return [(i, i) for i in range(min(n_up, n_dn))]
            if kind == "flare_right":
                return [(0, j) for j in range(k + 1)] + [(i, i + k) for i in range(1, n_up)]
            if kind == "flare_left":
                return [(i, i) for i in range(n_up - 1)] + [(n_up - 1, n_up - 1 + j) for j in range(k + 1)]
            if kind == "reduce_right":
                return [(i, 0) for i in range(k + 1)] + [(i, i - k) for i in range(k + 1, n_up)]
            raise ValueError(kind)

        for secs, evs in ((a.in_secs, sorted(a.in_events, key=lambda e: -e.hi)),
                          (a.out_secs, sorted(a.out_events, key=lambda e: e.lo))):
            if not secs:
                continue
            for k, e in enumerate(evs):
                up, dn = secs[k], secs[k + 1]
                lu, ld = self.b.edges[up.edge].lanes, self.b.edges[dn.edge].lanes
                sib: dict[int, list] = {}
                for i, j in pairs_for(e.kind, len(lu), len(ld), e.k):
                    d = self.conn(lu[i], ld[j], siblings=None)
                    sib.setdefault(i, []).append(d)

    # ------------------------------------------------------------------ 渠化右转
    def _build_slip(self, J: dict, a: Arm, t: Arm) -> None:
        b = self.b
        ev_a = next(e for e in a.in_events if e.kind == "slip")
        ev_t = next(e for e in t.out_events if e.kind == "slip")
        ws = ev_a.extra["w"]
        dn_main = next(s for s in a.in_secs if abs(s.s_a - ev_a.lo) < 1e-6)
        up_main = next(s for s in a.in_secs if abs(s.s_b - ev_a.hi) < 1e-6)
        o_up = next(s for s in t.out_secs if abs(s.s_b - ev_t.lo) < 1e-6)       # 核心 → 合流点
        o_dn = next(s for s in t.out_secs if abs(s.s_a - ev_t.hi) < 1e-6)       # 合流点 → 端口
        p0 = a.point(ev_a.lo, dn_main.left_off + sum(dn_main.widths) + ws / 2)
        t0 = -a.tangent(ev_a.lo)
        p1 = t.point(ev_t.lo, -(o_up.left_off + sum(o_up.widths) + ws / 2))
        t1 = t.tangent(ev_t.lo)
        chord = float(np.linalg.norm(p1 - p0))
        full = hermite_curve(p0, t0, p1, t1, 240, tension=chord * float(self.rng.uniform(0.9, 1.3)))
        from src.data.synth.kinematic import min_radius
        if min_radius(full) < self.cfg["kinematic"]["r_min_turn"] + 2.0:
            raise SceneReject("渠化右转道曲率过大")
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
        with_cw = bool(self.rng.random() < 0.7)
        nd_div, nd_mrg = ev_a.node, ev_t.node
        slip_edges = []
        if with_cw:
            cw_w = self.u((3.0, 4.0))
            half = L[-1] / 2
            i0 = int(np.searchsorted(L, half - cw_w / 2 - 0.5))
            i1 = int(np.searchsorted(L, half + cw_w / 2 + 0.5))
            mb = self.node("midblock_node")
            mid = full[int(np.searchsorted(L, half))]
            self.set_pos(mb, mid)
            s1 = b.edge(nd_div, mb, offset_polyline(_xyz(full[: i0 + 1]), ws / 2)[:, :2], [ws], roles=["slip_turn"],
                        stop_line_end=True)
            s2 = b.edge(mb, nd_mrg, offset_polyline(_xyz(full[i1:]), ws / 2)[:, :2], [ws], roles=["slip_turn"])
            slip_edges = [s1, s2]
            tm = full[i1] - full[i0]
            tm /= np.linalg.norm(tm)
            nrm = rot90(tm)
            b.crosswalk(mb, [s1], mid - nrm * ws / 2, mid + nrm * ws / 2, cw_w)
            self.marks.append({"t": "stop_line", "lanes": b.edges[s1].lanes})
        else:
            s1 = b.edge(nd_div, nd_mrg, offset_polyline(_xyz(full), ws / 2)[:, :2], [ws], roles=["slip_turn"])
            slip_edges = [s1]
        lu = b.edges[up_main.edge].lanes
        self.conn(lu[0], b.edges[slip_edges[0]].lanes[0])
        if with_cw:
            self.conn(b.edges[slip_edges[0]].lanes[0], b.edges[slip_edges[1]].lanes[0])
        self.conn(b.edges[slip_edges[-1]].lanes[0], b.edges[o_dn.edge].lanes[0])
        for e in slip_edges:
            self.marks.append({"t": "slip_edge", "edge": e})
        # 转角安全岛：渠化右转道左侧边界 ↔ 两臂右侧路缘 ↔ 路口角
        left = offset_polyline(_xyz(full), ws / 2 + 0.6)[:, :2]
        pa = a.point(a.r_stop + 1.0, dn_main.left_off + sum(dn_main.widths) + 0.6)
        pt = t.point(t.r_stop + 1.0, -(o_up.left_off + sum(o_up.widths) + 0.6))
        ring = np.vstack([left, pt[None], pa[None]])
        isl = Polygon(ring).buffer(0)
        lane_env = unary_union([LineString(l.points[:, :2]).buffer(l.width / 2 + 0.6, cap_style="flat")
                                for l in b.lanes])
        isl = isl.difference(lane_env).buffer(-0.3)
        if isl.geom_type == "MultiPolygon":
            isl = max(isl.geoms, key=lambda q: q.area)
        if isl.geom_type == "Polygon" and isl.area > 8.0 and isl.is_valid:
            self.islands.append(np.asarray(isl.exterior.coords)[:-1])
            self.marks.append({"t": "island", "poly": np.asarray(isl.exterior.coords)[:-1]})
        # 分流鼻端导流斜纹
        main_r = offset_polyline(_xyz(a.axis(ev_a.lo, ev_a.lo - 0.01, 2)), 0)[:, :2]
        _ = main_r
        nose_len = min(12.0, L[-1] * 0.25)
        k_n = int(np.searchsorted(L, nose_len))
        if k_n > 3:
            wedge_l = offset_polyline(_xyz(full[:k_n]), ws / 2 + 0.3)[:, :2]
            pr = a.point(ev_a.lo - nose_len, dn_main.left_off + sum(dn_main.widths) + 0.3)
            wedge = Polygon(np.vstack([wedge_l, pr[None]])).buffer(0)
            wedge = wedge.difference(lane_env).buffer(-0.1)
            if wedge.geom_type == "MultiPolygon":
                wedge = max(wedge.geoms, key=lambda q: q.area)
            if wedge.geom_type == "Polygon" and wedge.area > 1.0:
                self.channels.append(np.asarray(wedge.exterior.coords)[:-1])
                self.marks.append({"t": "channel", "poly": np.asarray(wedge.exterior.coords)[:-1]})

    # ------------------------------------------------------------------ 转向分配
    def _movements(self, J: dict) -> None:
        rng, c = self.rng, self.cfg
        arms = J["arms"]
        b = self.b
        for a in arms:
            if a.idx not in J["in_edge"]:
                continue
            ein = J["in_edge"][a.idx]
            n_in = len(b.edges[ein].lanes)
            mv = []
            for t in arms:
                if t.idx not in J["out_edge"]:
                    continue
                same = t.idx == a.idx
                if same and not (rng.random() < c["junction_uturn_prob"] and a.median_type != "none"):
                    continue
                if J["slip"].get(a.idx, (None, None))[1] == t.idx:
                    continue
                psi = 180.0 if same else self._psi(a, t)
                mv.append((psi, t.idx, self.gen.turn_class(psi, same)))
            if not mv:
                raise SceneReject("进口无任何去向")
            mv.sort()
            if len(mv) > 1:
                keep = [m for m in mv if not (rng.random() < c["movement_drop_prob"] and m[2] != "S")]
                mv = keep or mv
            assign = None
            while assign is None and mv:
                for _ in range(30):
                    cand = self._assign(n_in, len(mv))
                    sets = [frozenset(mv[m][2] for m in lane) for lane in cand]
                    if all(s in TURN_SETS for s in sets):
                        assign = cand
                        break
                if assign is None:
                    if len(mv) == 1:
                        raise SceneReject("进口车道转向集合无法用导向箭头表达")
                    pri = {"U": 0, "FL": 1, "FR": 1, "L": 2, "R": 2, "S": 3}
                    drop = min(range(len(mv)), key=lambda k: (pri[mv[k][2]], rng.random()))
                    mv = mv[:drop] + mv[drop + 1:]
            J.setdefault("assign", {})[a.idx] = (assign, mv)
        fed_arms = {t for _, (asg, mv) in J.get("assign", {}).items() for _, t, _ in mv}
        if set(J["out_edge"]) - fed_arms:
            raise SceneReject("存在无任何来向的出口臂")

    def _assign(self, n: int, m: int) -> list[list[int]]:
        """车道（右→左）承担连续、单调不交叉的去向区间。"""
        rng = self.rng
        if n >= m:
            cuts = np.sort(rng.choice(np.arange(1, n), m - 1, replace=False)) if m > 1 else np.array([], int)
            bounds = np.r_[0, cuts, n]
            lanes = [[] for _ in range(n)]
            for k in range(m):
                for i in range(bounds[k], bounds[k + 1]):
                    lanes[i].append(k)
            for k in range(m - 1):                    # 相邻去向共享边界车道
                i = bounds[k + 1]
                if rng.random() < 0.25 and i - 1 >= 0 and k not in lanes[i]:
                    lanes[i].insert(0, k)
            return lanes
        cuts = np.sort(rng.choice(np.arange(1, m), n - 1, replace=False)) if n > 1 else np.array([], int)
        bounds = np.r_[0, cuts, m]
        return [list(range(bounds[i], bounds[i + 1])) for i in range(n)]

    def _junction_conns(self, J: dict) -> None:
        b = self.b
        arms = {a.idx: a for a in J["arms"]}
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for k, (assign, mv) in J.get("assign", {}).items():
            ein = J["in_edge"][k]
            lin = b.edges[ein].lanes
            for m_idx, (psi, t, cls) in enumerate(mv):
                lanes = [i for i, lst in enumerate(assign) if m_idx in lst]
                if not lanes:
                    continue
                eout = J["out_edge"][t]
                n_out = len(b.edges[eout].lanes)
                if len(lanes) > n_out:
                    outs = [int(i * n_out / len(lanes)) for i in range(len(lanes))]
                elif cls in ("R", "FR"):
                    outs = list(range(len(lanes)))
                elif cls in ("L", "FL", "U"):
                    outs = list(range(n_out - len(lanes), n_out))
                else:
                    off = int(self.rng.integers(0, n_out - len(lanes) + 1))
                    outs = [off + i for i in range(len(lanes))]
                for i, j in zip(lanes, outs):
                    groups.setdefault((ein, eout), []).append((i, j))
        # 出口无源车道：单调保序扇出补连（迭代：相邻已供车车道的边界进口车道增连）
        for eout in set(J["out_edge"].values()):
            n_out = len(b.edges[eout].lanes)
            while True:
                fed = {j for (ei, eo), ps in groups.items() if eo == eout for _, j in ps}
                unfed = [j for j in range(n_out) if j not in fed]
                if not unfed:
                    break
                progress = False
                for j in unfed:
                    for nb, pick in ((j - 1, max), (j + 1, min)):
                        if nb not in fed:
                            continue
                        feeders = [(ei, i) for (ei, eo), ps in groups.items() if eo == eout for i, jj in ps if jj == nb]
                        ei, i = pick(feeders, key=lambda x: x[1])
                        groups[(ei, eout)].append((i, j))
                        progress = True
                        break
                    if progress:
                        break
                if not progress:
                    raise SceneReject("出口车道无法补连")
        # 运动学轨迹：同 (进口, 出口) 组内右 → 左生成，并列保序
        for (ein, eout), ps in groups.items():
            ps = sorted(set(ps), key=lambda x: (x[0], x[1]))
            uturn = any(ein == J["in_edge"].get(k) and eout == J["out_edge"].get(k) for k in arms)
            done = []
            for i, j in ps:
                # 并列保序间隔只约束起终车道均不同的并列转向（同源扇出 / 同汇合流本应交汇）
                sib = [d for (i2, j2, d) in done if i2 != i and j2 != j]
                d = self.conn(b.edges[ein].lanes[i], b.edges[eout].lanes[j], uturn=uturn, siblings=sib or None)
                done.append((i, j, d))

    # ------------------------------------------------------------------ 路段掉头
    def _midblock_uturns(self, a: Arm) -> None:
        b = self.b
        for e in a.in_events:
            if e.kind != "midblock" or e.extra["type"] != "uturn":
                continue
            in_up = next(s for s in a.in_secs if abs(s.s_b - e.hi) < 1e-6)     # 远端 → 开口
            in_dn = next(s for s in a.in_secs if abs(s.s_a - e.lo) < 1e-6)     # 开口 → 核心
            out_up = next(s for s in a.out_secs if abs(s.s_b - e.lo) < 1e-6)   # 核心 → 开口
            out_dn = next(s for s in a.out_secs if abs(s.s_a - e.hi) < 1e-6)   # 开口 → 远端
            dirs = [(in_up, out_dn), (out_up, in_dn)]
            pick = dirs if self.rng.random() < 0.4 else [dirs[int(self.rng.integers(2))]]
            for src, dst in pick:
                li = b.edges[src.edge].lanes[-1]
                lj = b.edges[dst.edge].lanes[-1]
                self.conn(li, lj, uturn=True)
                b.lanes[li].role = "uturn_bay"
                self.marks.append({"t": "uturn_arrow", "lane": li})

    # ------------------------------------------------------------------ 中央分隔与岛
    def _left_profile(self, a: Arm, inbound: bool):
        secs = a.in_secs if inbound else a.out_secs
        xs, ys = [], []
        for s in sorted(secs, key=lambda s: min(s.s_a, s.s_b)):
            lo, hi = min(s.s_a, s.s_b), max(s.s_a, s.s_b)
            xs += [lo, hi]
            ys += [s.left_off, s.left_off]
        return np.asarray(xs), np.asarray(ys)

    def _medians(self, a: Arm) -> None:
        if not a.w_in or not a.w_out or a.median_type == "none":
            return
        if a.skip_out or a.far_in_node is not None:
            partner = a.link_partner
            spacing = a.length + partner.r_stop
            span = (a.r_stop, spacing / 2)
            xi, yi = self._left_profile(a, True)
            xo, yo = self._left_profile(partner, True)
            xo = spacing - xo
            order = np.argsort(xo)
            xo, yo = xo[order], yo[order]
        else:
            span = (a.r_stop if a.near == "core" else a.s_min, a.length)
            xi, yi = self._left_profile(a, True)
            xo, yo = self._left_profile(a, False)
        s0 = span[0]
        if a.median_type == "island" and a.near == "core":
            two = bool(a.cw) and a.median >= self.cfg["two_stage_min_island_m"] and self.rng.random() < 0.5
            s0 = a.r_stop - ((a.cw["width"] + 1.0 + self.u((0.3, 1.0))) if two else self.u((0.0, 1.0)))
        s = np.linspace(s0, span[1], max(8, int((span[1] - s0) / 1.0)))
        li = np.interp(s, xi, yi)
        lo_ = np.interp(s, xo, yo)
        openings = [(e.lo, e.hi) for e in a.in_events if e.kind == "midblock" and e.extra["type"] == "uturn"]
        mtype = a.median_type
        if mtype == "island":
            inset = 0.25
            ok = (li + lo_ - 2 * inset) > 0.5
            for lo, hi in openings:
                ok &= ~((s > lo) & (s < hi))
            runs = _runs(ok)
            for r0, r1 in runs:
                if r1 - r0 < 3:
                    continue
                ss = s[r0:r1]
                left = np.array([a.point(x, y - inset) for x, y in zip(ss, li[r0:r1])])
                right = np.array([a.point(x, -(y - inset)) for x, y in zip(ss, lo_[r0:r1])])
                ring = np.vstack([left, right[::-1]])
                p = Polygon(ring).buffer(0)
                if p.geom_type == "Polygon" and p.area > 2.0:
                    ring = np.asarray(p.exterior.coords)[:-1]
                    self.islands.append(ring)
                    self.marks.append({"t": "island", "poly": ring, "median": True})
        else:
            cen = np.array([a.point(x, (y1 - y2) / 2) for x, y1, y2 in zip(s, li, lo_)])
            keep = np.ones(len(s), bool)
            for lo, hi in openings:
                keep &= ~((s > lo) & (s < hi))
            for r0, r1 in _runs(keep):
                if r1 - r0 < 3:
                    continue
                self.marks.append({"t": "median", "kind": mtype, "line": cen[r0:r1], "width": a.median})

    # ------------------------------------------------------------------ 人行横道
    def _arm_crosswalks(self, J: dict) -> None:
        b = self.b
        for a in J["arms"]:
            if not a.cw:
                continue
            w = a.cw["width"]
            s_c = a.r_stop - w / 2 - self.u((0.5, 1.5))
            ein = J["in_edge"].get(a.idx)
            eout = J["out_edge"].get(a.idx)
            W_in = sum(b.lanes[i].width for i in b.edges[ein].lanes) if ein is not None else 0.0
            W_out = sum(b.lanes[i].width for i in b.edges[eout].lanes) if eout is not None else 0.0
            lin = a.in_secs[-1].left_off if ein is not None else 0.0
            if eout is None:
                lout = 0.0
            elif a.out_secs:
                lout = a.out_secs[0].left_off
            else:
                lout = a.link_partner.in_secs[0].left_off
            two = (a.median_type == "island" and a.median >= self.cfg["two_stage_min_island_m"]
                   and ein is not None and eout is not None)
            if two:
                b.crosswalk(J["core"], [ein], a.point(s_c, lin + W_in), a.point(s_c, lin), w)
                b.crosswalk(J["core"], [eout], a.point(s_c, -(lout + W_out)), a.point(s_c, -lout), w)
            else:
                es = [e for e in (ein, eout) if e is not None]
                p_hi = a.point(s_c, lin + W_in) if ein is not None else a.point(s_c, -lout)
                p_lo = a.point(s_c, -(lout + W_out)) if eout is not None else a.point(s_c, lin)
                b.crosswalk(J["core"], es, p_lo, p_hi, w)

    def _midblock_crosswalks(self, a: Arm) -> None:
        b = self.b
        for e in a.in_events:
            if e.kind != "midblock" or e.extra["type"] != "crosswalk":
                continue
            in_dn = next(s for s in a.in_secs if abs(s.s_a - e.lo) < 1e-6)   # 开口 → 核心
            out_up = next(s for s in a.out_secs if abs(s.s_b - e.lo) < 1e-6)  # 核心 → 开口
            s_c = (e.lo + e.hi) / 2
            w = (e.hi - e.lo) - 2.0
            lin, lout = in_dn.left_off, out_up.left_off
            W_in, W_out = sum(in_dn.widths), sum(out_up.widths)
            if a.median_type == "island" and a.median >= self.cfg["two_stage_min_island_m"]:
                b.crosswalk(e.node, [in_dn.edge], a.point(s_c, lin + W_in), a.point(s_c, lin), w)
                b.crosswalk(e.node, [out_up.edge], a.point(s_c, -(lout + W_out)), a.point(s_c, -lout), w)
            else:
                b.crosswalk(e.node, [in_dn.edge, out_up.edge], a.point(s_c, -(lout + W_out)),
                            a.point(s_c, lin + W_in), w)

    # ------------------------------------------------------------------ 走廊 / 路段
    def build_corridor(self) -> None:
        rng, c = self.rng, self.cfg
        spacing = self.u(c["corridor_spacing_m"])
        arms1 = self.gen._sample_arms(rng, np.zeros(2), max_len=spacing * 0.42)
        cand = [a for a in arms1 if a.w_in and a.w_out]
        if not cand:
            raise SceneReject("走廊无双向臂")
        a1 = cand[int(rng.integers(len(cand)))]
        c2 = a1.u * spacing
        arms2 = self.gen._sample_arms(rng, c2, fixed=[(a1.angle + 180) % 360], max_len=spacing * 0.42)
        a2 = arms2[0]
        a2.w_in = list(a1.w_out)
        a2.w_out = list(a1.w_in)
        a2.median_type, a2.median, a2.bend, a2.lateral = a1.median_type, a1.median, 0.0, 0.0
        a1.bend = a1.lateral = 0.0
        for a in (a1, a2):
            a.is_link = True
            a.skip_out = True
        a1.link_partner, a2.link_partner = a2, a1
        self._r_stop(arms1)
        self._r_stop(arms2)
        a1.length = spacing - a2.r_stop
        a2.length = spacing - a1.r_stop
        J1 = self.build_junction(np.zeros(2), arms1, compute_r=False)
        J2 = self.build_junction(c2, arms2, compute_r=False)
        a1.far_in_node, a2.far_in_node = J2["core"], J1["core"]
        self._finish_junctions([J1, J2], links=[(J1, a1, J2, a2)])

    def build_segment(self) -> None:
        rng, c = self.rng, self.cfg
        L = self.u(c["segment_length_m"])
        a = Arm(0, float(rng.uniform(0, 360)), np.zeros(2), L / 2)
        a.near = "segment"
        a.s_min = -L / 2
        a.w_in = self.gen._widths(rng, self.gen._count(rng, c["lanes_in"]))
        a.w_out = self.gen._widths(rng, self.gen._count(rng, c["lanes_out"]))
        a.bend = self.u(c["bend_m"]) if rng.random() < c["bend_prob"] else 0.0
        self.gen._sample_median(rng, a)
        a.edge_line = bool(rng.random() < c["render"]["edge_line_prob"])
        a.r_stop = a.s_min
        self.arms_all.append(a)
        # 路段至少含一个中段事件
        typ = "uturn" if (a.median_type in ("island", "barrier") and rng.random() < 0.5) else "crosswalk"
        gap = self.u(c["uturn_opening_m"]) if typ == "uturn" else self.u(c["crosswalk_width"]) + 2.0
        s_m = self.u((-L / 2 + 50.0, L / 2 - 50.0))
        ev = Event("midblock", s_m - gap / 2, s_m + gap / 2, extra={"type": typ})
        a.in_events.append(ev)
        a.out_events.append(Event("midblock", ev.lo, ev.hi, extra=ev.extra))
        a.perms_in = ["general"] * len(a.w_in)
        a._s_end = a.length - 12
        self._sections(a)
        self._build_edges(a, None)
        self._event_conns(a)
        self._midblock_uturns(a)
        self._medians(a)
        self._midblock_crosswalks(a)
        self._node_positions()

    def _finish_junctions(self, Js: list[dict], links=()) -> None:
        for J in Js:
            for a in J["arms"]:
                self._sections(a)
        for J in Js:
            for a in J["arms"]:
                self._medians(a)
        for J in Js:
            for a in J["arms"]:
                self._build_edges(a, J)
        for J1, a1, J2, a2 in links:
            J1["out_edge"][a1.idx] = a2.in_secs[0].edge
            J2["out_edge"][a2.idx] = a1.in_secs[0].edge
        for J in Js:
            arms = {x.idx: x for x in J["arms"]}
            for k, (ia, ta) in J["slip"].items():
                self._build_slip(J, arms[ia], arms[ta])
        for J in Js:
            for a in J["arms"]:
                self._event_conns(a)
                if a.out_secs and a.in_secs:
                    self._midblock_uturns(a)
        for J in Js:
            self._movements(J)
            self._junction_conns(J)
            self._arm_crosswalks(J)
            for a in J["arms"]:
                if a.in_secs and a.out_secs:
                    self._midblock_crosswalks(a)
        self._node_positions()

    # ------------------------------------------------------------------ 节点位置与收尾
    def _node_positions(self) -> None:
        """junction_core：接入横断面角点质心；其余节点：接入车道端点质心。"""
        b = self.b
        ends: dict[int, list[np.ndarray]] = {}
        corners: dict[int, list[np.ndarray]] = {}
        for e in b.edges:
            for nid, at_end in ((e.from_node, False), (e.to_node, True)):
                k = -1 if at_end else 0
                for lid in e.lanes:
                    ends.setdefault(nid, []).append(b.lanes[lid].points[k, :2])
                corners.setdefault(nid, []).extend(b.edge_corners(e.id, nid))
        for n in b.nodes:
            if n.id not in ends:
                raise SceneReject(f"节点 {n.id} 无接入车道")
            src = corners if n.node_type == "junction_core" else ends
            self.set_pos(n.id, np.mean(src[n.id], axis=0))

    def finish(self):
        if self.kind == "junction":
            self._finish_junctions(self.junctions)
        g = self.b.build()
        self._check_overlap(g)
        self._negatives(g)
        marks = self._lane_marks(g)
        return g, marks, self.negs

    # ------------------------------------------------------------------ 几何安全网
    def _lane_poly(self, l) -> Polygon:
        return LineString(l.points[:, :2]).buffer(l.width / 2, cap_style="flat")

    def _check_overlap(self, g: LaneGraph) -> None:
        from shapely.strtree import STRtree
        polys = [self._lane_poly(l) for l in g.lanes]
        e_of = g.edge_of_lane()
        tree = STRtree(polys)
        for i, p in enumerate(polys):
            for j in tree.query(p):
                if j <= i or e_of[g.lanes[i].id].id == e_of[g.lanes[j].id].id:
                    continue
                if p.intersection(polys[j]).area > 0.05:
                    raise SceneReject("车道包络相互重叠")
        isl = [Polygon(r) for r in self.islands]
        for q in isl:
            for j in tree.query(q):
                if q.intersection(polys[j]).area > 0.05:
                    raise SceneReject("安全岛与车道重叠")
        for cw in g.crosswalks:
            ep = cw.endpoints[:, :2]
            poly = LineString(ep).buffer(cw.width / 2, cap_style="flat")
            for j in tree.query(poly):
                if poly.intersection(polys[j]).area > 0.05:
                    raise SceneReject("人行横道压车道段")
        cores = [n for n in g.nodes if n.node_type == "junction_core"]
        for n in cores:
            if not Polygon(n.polygon).is_valid:
                raise SceneReject("路口轮廓自交")

    # ------------------------------------------------------------------ 渲染语义
    def _lane_marks(self, g: LaneGraph) -> list[dict]:
        """把 G* 的车道 / edge 语义整理为渲染器输入（与 G* 几何严格一致）。"""
        marks = list(self.marks)
        lanes = {l.id: l for l in g.lanes}
        arm_of_edge = {}
        for a in self.arms_all:
            for s in a.in_secs + a.out_secs:
                if s.edge is not None:
                    arm_of_edge[s.edge] = (a, s)
        slip_edges = {m["edge"] for m in marks if m["t"] == "slip_edge"}
        for e in g.edges:
            ls = [lanes[i] for i in e.lanes]
            a, sec = arm_of_edge.get(e.id, (None, None))
            marks.append({"t": "edge", "edge": e.id, "lanes": e.lanes, "slip": e.id in slip_edges,
                          "edge_line": bool(a.edge_line) if a is not None else True,
                          "left_kind": self._left_kind(a, sec) if a is not None else "white_solid",
                          "stop": bool(ls[0].is_stop_line_end)})
        marks.append({"t": "junctions", "cores": [J["core"] for J in self.junctions]})
        marks.append({"t": "dense_conn", "curves": {f"{i}_{j}": v for (i, j), v in self.dense_conn.items()}})
        return marks

    def _left_kind(self, a: Arm, sec: Sec) -> str:
        if a.median_type in ("double_yellow", "single_yellow", "yellow_dashed", "barrier"):
            return "median_painted"
        if a.median_type == "island":
            return "yellow_edge" if self.rng.random() < 0.6 else "none"
        return "yellow_solid" if (not a.w_in or not a.w_out) and self.rng.random() < 0.3 else "white_solid"

    # ------------------------------------------------------------------ 负样本
    def _negatives(self, g: LaneGraph) -> None:
        from src.data.synth.negatives import place_negatives
        place_negatives(self, g)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out
