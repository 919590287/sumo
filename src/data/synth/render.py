"""百度风格矢量渲染器（docs/DESIGN.md §5 Layer 1-2「百度风格域随机化」的离线部分）。

把规范化合成真值 G* 与场景语义 marks 反向渲染为与真实 GPKG 同模式的要素表：
图层（南沙道路要素 / 斑马线白色条纹 / 隔离设施路径 / 其他原生区域面 / 公交站点位）+ ``分类代码``
+ 箭头 ``heading``（UTM 弧度）/ ``旋转角度`` / ``显示宽度米`` / ``显示长度米``，从而与真实数据共用
同一套光栅化（§3.2 分支 A）与 Token 化（分支 B）代码。

漆宽、虚线段长、箭头尺寸、斑马线条宽、导流斜纹宽等均按开创大道样本实测范围随机化
（configs/synth.yaml ``render``）。磨损缺损 / 瓦片切缝 / 几何抖动 / 箭头缺失在 DataLoader 在线执行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from src.common.geom import cumulative_arclength, offset_polyline, resample_polyline, tangents
from src.data.synth.scene import TURN_SETS, Scene, SceneGenerator, _xyz
from src.graph.lane_graph import LaneGraph

TURN_ANGLE = {"S": 0.0, "L": 90.0, "R": -90.0, "FL": 35.0, "FR": -35.0, "U": 180.0}


@dataclass
class FeatureSet:
    """场景局部坐标下的百度风格要素（另存场景 UTM 偏移）。"""
    road: list[tuple] = field(default_factory=list)      # (code, geom, heading, disp_w, disp_l)
    stripes: list = field(default_factory=list)          # 斑马线白色条纹
    separation: list = field(default_factory=list)       # 隔离设施路径（LineString）
    other_area: list[tuple] = field(default_factory=list)  # (code, geom)
    bus_stop: list = field(default_factory=list)         # Point

    def add(self, code, geom, heading=float("nan"), w=float("nan"), l=float("nan")):
        if geom is None or geom.is_empty:
            return
        if geom.geom_type == "GeometryCollection":
            geom = MultiPolygon([g for g in geom.geoms if g.geom_type == "Polygon"])
            if geom.is_empty:
                return
        self.road.append((code, geom, heading, w, l))


# ====================================================================== 基元
def stroke(line: np.ndarray, width: float):
    if len(line) < 2 or cumulative_arclength(line)[-1] < 1e-3:
        return None
    return LineString(line).buffer(width / 2, cap_style="flat", join_style="mitre", mitre_limit=3.0)


def dashes(line: np.ndarray, width: float, dash: float, gap: float, phase: float) -> list:
    s = cumulative_arclength(line)
    if s[-1] < 0.5:
        return []
    out = []
    t = -phase
    while t < s[-1]:
        a, b = max(t, 0.0), min(t + dash, s[-1])
        if b - a > 0.3:
            k = max(2, int((b - a) / 1.0) + 2)
            tt = np.linspace(a, b, k)
            seg = np.stack([np.interp(tt, s, line[:, 0]), np.interp(tt, s, line[:, 1])], 1)
            g = stroke(seg, width)
            if g is not None:
                out.append(g)
        t += dash + gap
    return out


def lane_left(l, extra: float = 0.0) -> np.ndarray:
    return offset_polyline(l.points, l.width / 2 + extra)[:, :2]


def lane_right(l, extra: float = 0.0) -> np.ndarray:
    return offset_polyline(l.points, -(l.width / 2 + extra))[:, :2]


def arrow_glyph(turns: list[str], L: float, W: float, rng) -> Polygon:
    """局部坐标（x 前进、y 向左）的组合导向箭头多边形。"""
    shaft = max(0.22, W * 0.28)
    parts = []
    back = -L / 2
    head_l = L * 0.28
    for t in turns:
        if t == "S":
            path = np.array([[back, 0.0], [L / 2 - head_l, 0.0]])
            tip, d = np.array([L / 2, 0.0]), np.array([1.0, 0.0])
            hw = W / 2
        elif t == "U":
            x0 = back + L * 0.45
            r = W * 0.55
            path = np.array([[back, 0.0], [x0, 0.0], [x0 + r * 0.6, r * 0.5], [x0, r], [x0 - L * 0.12, r]])
            d = np.array([-1.0, 0.0])
            tip = path[-1] + d * head_l * 0.8
            hw = W * 0.35
        else:
            ang = math.radians(TURN_ANGLE[t])
            x0 = back + L * (0.40 if t in ("L", "R") else 0.35)
            lb = L * 0.22
            d = np.array([math.cos(ang), math.sin(ang)])
            p1 = np.array([x0, 0.0]) + d * lb
            path = np.array([[back, 0.0], [x0, 0.0], p1])
            tip = p1 + d * head_l * 0.8
            hw = W * 0.35
        parts.append(LineString(path).buffer(shaft / 2, cap_style="flat", join_style="mitre"))
        base = tip - d * head_l * 0.8
        nrm = np.array([-d[1], d[0]])
        parts.append(Polygon([tip, base + nrm * hw, base - nrm * hw]))
    return unary_union(parts).buffer(0)


def place(geom, center, heading):
    g = rotate(geom, math.degrees(heading), origin=(0, 0))
    return translate(g, center[0], center[1])


def lane_turn_set(g: LaneGraph, gen: SceneGenerator, lid: int) -> list[str]:
    lanes = {l.id: l for l in g.lanes}
    e_of = g.edge_of_lane()
    out = set()
    for c in g.connections:
        if c.from_lane != lid:
            continue
        ti = tangents(lanes[lid].points)[-1]
        tj = tangents(lanes[c.to_lane].points)[0]
        psi = math.degrees(math.atan2(ti[0] * tj[1] - ti[1] * tj[0], float(ti @ tj)))
        same = False
        out.add(gen.turn_class(psi, same))
    order = ["L", "S", "R", "U", "FL", "FR"]
    _ = e_of
    return [t for t in order if t in out]


# ====================================================================== 主渲染
def render_scene(sc: Scene, gen: SceneGenerator, rng) -> tuple[FeatureSet, dict]:
    cfg = gen.cfg["render"]
    g = sc.graph
    fs = FeatureSet()
    lanes = {l.id: l for l in g.lanes}
    nodes = {n.id: n for n in g.nodes}
    edges = {e.id: e for e in g.edges}
    marks = sc.marks
    dense = next(m for m in marks if m["t"] == "dense_conn")["curves"]
    W_WHITE = lambda: float(rng.uniform(*cfg["white_width"]))  # noqa: E731
    W_YELLOW = lambda: float(rng.uniform(*cfg["yellow_width"]))  # noqa: E731
    dash_len = float(rng.uniform(*cfg["dash_len_m"]))
    dash_gap = float(rng.uniform(*cfg["dash_gap_m"]))
    arrows_meta = []
    surface = []

    # ---------------------------------------------------------------- 车道与 edge 标线
    for m in marks:
        if m["t"] != "edge":
            continue
        e = edges[m["edge"]]
        ls = [lanes[i] for i in e.lanes]
        to_type = nodes[e.to_node].node_type
        ext = 1.0 if to_type == "road_continuation" else 0.0
        for l in ls:
            surface.append(LineString(l.points[:, :2]).buffer(l.width / 2 + 0.05, cap_style="flat"))
        for k in range(len(ls) - 1):
            a, b = ls[k], ls[k + 1]
            line = lane_left(a)
            if ext:
                t = tangents(line)[-1]
                line = np.vstack([line, line[-1] + t * ext])
            if "bus_only" in (a.permission,):
                for d in dashes(line, 0.2, dash_len * 0.6, dash_gap * 0.5, rng.uniform(0, 3)):
                    fs.add("bus_lane_marking", d)
                continue
            if "variable" in (a.permission, b.permission):
                fs.add("variable_lane_marking", zigzag(line, 0.15, 0.35, 3.0))
                continue
            if b.permission == "reversible":
                for off in (-0.12, 0.12):
                    ll = offset_polyline(_xyz(line), off)[:, :2]
                    for d in dashes(ll, 0.15, dash_len, dash_gap * 0.6, rng.uniform(0, dash_len)):
                        fs.add("reversible_lane_marking", d)
                continue
            if a.allow_change_left and b.allow_change_right:
                ph = float(rng.uniform(0, dash_len + dash_gap))
                for d in dashes(line, W_WHITE(), dash_len, dash_gap, ph):
                    fs.add("white_dashed_stroke", d)
            elif rng.random() < 0.08:
                w = W_WHITE()
                gap = float(rng.uniform(0.1, 0.2))
                g2 = unary_union([stroke(offset_polyline(_xyz(line), s * (gap / 2 + w / 2))[:, :2], w)
                                  for s in (-1, 1)])
                fs.add("white_double_solid", g2)
            else:
                fs.add("white_solid_stroke", stroke(line, W_WHITE()))
        r0 = ls[0]
        if m["edge_line"] or m["slip"]:
            fs.add("white_solid_stroke", stroke(lane_right(r0, -0.15), W_WHITE()))
        lk = m["left_kind"]
        lt = ls[-1]
        if m["slip"]:
            fs.add("white_solid_stroke", stroke(lane_left(lt, -0.15), W_WHITE()))
        elif lk == "yellow_edge":
            fs.add("yellow_solid_stroke", stroke(lane_left(lt, -0.12), W_YELLOW()))
        elif lk in ("white_solid", "yellow_solid"):
            code = "white_solid_stroke" if lk == "white_solid" else "yellow_solid_stroke"
            fs.add(code, stroke(lane_left(lt, -0.12), W_WHITE()))
        # 停止线
        if m["stop"] and to_type in ("junction_core", "midblock_node"):
            sw = float(rng.uniform(*cfg["stop_line_width"]))
            p_r = lane_right(r0)[-1]
            p_l = lane_left(lt)[-1]
            t = tangents(lt.points)[-1][:2]
            fs.add("white_solid_stroke", stroke(np.array([p_r - t * sw / 2, p_l - t * sw / 2]), sw))
        # 公交专用道文字
        for l in ls:
            if l.permission == "bus_only" and rng.random() < cfg["bus_text_prob"]:
                fs.add("bus_lane_marking", bus_text(l, rng))

    # ---------------------------------------------------------------- 导向箭头
    cores = {n.id for n in g.nodes if n.node_type == "junction_core"}
    e_of = g.edge_of_lane()
    for l in g.lanes:
        e = e_of[l.id]
        if not ((l.is_stop_line_end and e.to_node in cores) or l.role == "uturn_bay"):
            continue
        turns = lane_turn_set(g, gen, l.id)
        code = TURN_SETS.get(frozenset(turns))
        if code is None:
            continue
        s = cumulative_arclength(l.points)
        AL = float(rng.uniform(*cfg["arrow_len_m"]))
        AW = min(float(rng.uniform(*cfg["arrow_width_m"])), l.width * 0.7)
        d1 = float(rng.uniform(*cfg["arrow_first_m"]))
        offs = [d1]
        if rng.random() < cfg["arrow_second_prob"]:
            offs.append(d1 + AL + float(rng.uniform(*cfg["arrow_second_gap_m"])))
        for d in offs:
            sc_ = s[-1] - d - AL / 2
            if sc_ - AL / 2 < 1.0:
                continue
            p = np.array([np.interp(sc_, s, l.points[:, 0]), np.interp(sc_, s, l.points[:, 1])])
            k = int(np.searchsorted(s, sc_).clip(1, len(s) - 1))
            t = l.points[k, :2] - l.points[k - 1, :2]
            hd = math.atan2(t[1], t[0])
            glyph = place(arrow_glyph(turns, AL, AW, rng), p, hd)
            fs.add(code, glyph, hd, AW + float(rng.uniform(0.3, 0.9)), AL)
            arrows_meta.append({"lane": l.id, "turns": turns, "pos": p.tolist(), "heading": hd})

    # ---------------------------------------------------------------- 内部连接：路面 + 待转区 + 引导虚线
    junction_left = []
    for key, d in dense.items():
        i, j = map(int, key.split("_"))
        w = lanes[i].width
        surface.append(LineString(d).buffer(w / 2 + 0.05))
        if e_of[i].to_node in cores:
            ti = tangents(lanes[i].points)[-1]
            tj = tangents(lanes[j].points)[0]
            psi = math.degrees(math.atan2(ti[0] * tj[1] - ti[1] * tj[0], float(ti @ tj)))
            if gen.turn_class(psi, False) == "L":
                junction_left.append((i, j, d, w))
    wait_on = rng.random() < gen.cfg["waiting_area_prob"]
    guide_on = rng.random() < gen.cfg["guide_line_prob"]
    for i, j, d, w in junction_left:
        s = cumulative_arclength(d)
        f0 = 0.0
        if wait_on and lanes[i].is_stop_line_end:
            f = float(rng.uniform(*cfg["waiting_len_frac"]))
            k1 = int(np.searchsorted(s, s[-1] * f))
            if k1 > 3 and s[k1] > 8.0:
                seg = d[: k1 + 1]
                for side in (-1, 1):
                    ln = offset_polyline(_xyz(seg), side * w / 2)[:, :2]
                    for dd in dashes(ln, W_WHITE(), 1.5, 1.5, 0.0):
                        fs.add("white_dashed_stroke", dd)
                t = (d[k1] - d[k1 - 1]) / max(np.linalg.norm(d[k1] - d[k1 - 1]), 1e-9)
                n = np.array([-t[1], t[0]])
                fs.add("white_solid_stroke", stroke(np.array([d[k1] - n * w / 2, d[k1] + n * w / 2]), 0.25))
                km = int(np.searchsorted(s, s[k1] * 0.5))
                tm = d[min(km + 1, len(d) - 1)] - d[max(km - 1, 0)]
                hd = math.atan2(tm[1], tm[0])
                AL = min(5.0, s[k1] * 0.5)
                if AL > 2.5:
                    fs.add("arrow_left", place(arrow_glyph(["L"], AL, min(1.6, w * 0.6), rng), d[km], hd), hd,
                           min(1.6, w * 0.6) + 0.5, AL)
                f0 = f
        if guide_on:
            a0 = int(np.searchsorted(s, s[-1] * max(f0, 0.12)))
            a1 = int(np.searchsorted(s, s[-1] * 0.88))
            if a1 - a0 > 3:
                seg = d[a0:a1]
                for side in (-1, 1):
                    ln = offset_polyline(_xyz(seg), side * w / 2)[:, :2]
                    for dd in dashes(ln, W_WHITE(), 1.0, 2.0, float(rng.uniform(0, 3))):
                        fs.add("white_dashed_stroke", dd)

    # ---------------------------------------------------------------- 渐变段 / 节点区路面
    for n in g.nodes:
        if n.polygon is not None:
            surface.append(Polygon(n.polygon).buffer(0.05))
    for e in g.edges:
        pass

    # ---------------------------------------------------------------- 中央分隔 / 岛 / 导流
    for m in marks:
        t = m["t"]
        if t == "median":
            line = np.asarray(m["line"])
            W = float(m["width"])
            surface.append(LineString(line).buffer(W / 2 + 0.1, cap_style="flat"))
            kind = m["kind"]
            if kind == "double_yellow" and rng.random() < 0.06:
                w = W_YELLOW()
                fs.add("yellow_left_solid_right_dashed",
                       unary_union([stroke(offset_polyline(_xyz(line), -0.12)[:, :2], w)]
                                   + dashes(offset_polyline(_xyz(line), 0.12)[:, :2], w, 4.0, 4.0, 0.0)))
            elif kind == "double_yellow":
                w = W_YELLOW()
                gap = min(float(rng.uniform(*cfg["double_yellow_gap"])), max(W - 2 * w, 0.05))
                for side in (-1, 1):
                    fs.add("yellow_solid_stroke", stroke(offset_polyline(_xyz(line), side * (gap / 2 + w / 2))[:, :2], w))
            elif kind == "single_yellow":
                fs.add("yellow_solid_stroke", stroke(line, W_YELLOW()))
            elif kind == "yellow_dashed":
                w = W_YELLOW()
                for side in (-1, 1):
                    ln = offset_polyline(_xyz(line), side * 0.12)[:, :2]
                    for d in dashes(ln, w, 4.0, 4.0, 0.0):
                        fs.add("yellow_double_dashed", d)
            elif kind == "barrier":
                fs.separation.append(LineString(line))
                if rng.random() < 0.5:
                    for side in (-1, 1):
                        fs.add("yellow_solid_stroke",
                               stroke(offset_polyline(_xyz(line), side * (W / 2 - 0.15))[:, :2], W_YELLOW()))
        elif t == "island":
            fs.add("traffic_island", Polygon(m["poly"]).buffer(0))
        elif t == "channel":
            poly = Polygon(m["poly"]).buffer(0)
            surface.append(poly)
            for sp in channel_stripes(poly, rng, cfg):
                fs.add("channelization_area", sp)
            if rng.random() < 0.7:
                fs.add("white_solid_stroke", poly.exterior.buffer(0.08, cap_style="flat").intersection(
                    poly.buffer(0.2)))

    # ---------------------------------------------------------------- 人行横道
    for cw in g.crosswalks:
        p0, p1 = cw.endpoints[0, :2], cw.endpoints[1, :2]
        ax = p1 - p0
        Lc = float(np.linalg.norm(ax))
        ax = ax / max(Lc, 1e-9)
        nrm = np.array([-ax[1], ax[0]])
        rect = Polygon([p0 - nrm * cw.width / 2, p1 - nrm * cw.width / 2, p1 + nrm * cw.width / 2,
                        p0 + nrm * cw.width / 2])
        fs.add("crosswalk_area", rect)
        surface.append(rect.buffer(0.1))
        sw = float(rng.uniform(*cfg["stripe_width_m"]))
        gap = sw * float(rng.uniform(*cfg["stripe_gap_ratio"]))
        t = float(rng.uniform(0, gap))
        while t + sw <= Lc:
            a, b = p0 + ax * t, p0 + ax * (t + sw)
            fs.stripes.append(Polygon([a - nrm * cw.width / 2, b - nrm * cw.width / 2, b + nrm * cw.width / 2,
                                       a + nrm * cw.width / 2]))
            t += sw + gap

    # ---------------------------------------------------------------- 负样本
    for m in marks:
        t = m["t"]
        if t == "neg_road":
            surface.append(Polygon(m["poly"]).buffer(0.05))
            if m.get("gate"):
                fs.separation.append(LineString(m["gate_line"]))
        elif t == "neg_parking":
            render_parking(fs, m, rng, surface)
        elif t == "neg_nonmotor":
            poly = Polygon(m["poly"]).buffer(0)
            surface.append(poly.buffer(0.3))
            if m["sep"] == "separation":
                fs.separation.append(LineString(m["sep_line"]))
            else:
                fs.add("white_solid_stroke", stroke(np.asarray(m["sep_line"]), W_WHITE()))
            if m["greenway"]:
                fs.add("greenway_boundary", stroke(np.asarray(m["outer"]), 0.15))
            c = np.asarray(m["center"])
            s = cumulative_arclength(c)
            x = float(rng.uniform(3, 12))
            while x < s[-1] - 2:
                p = np.array([np.interp(x, s, c[:, 0]), np.interp(x, s, c[:, 1])])
                k = int(np.searchsorted(s, x).clip(1, len(s) - 1))
                tt = c[k] - c[k - 1]
                hd = math.atan2(tt[1], tt[0])
                fs.add("bicycle_lane_symbol", place(box(-1.13, -0.93, 1.13, 0.93), p, hd))
                x += float(rng.uniform(18, 40))
        elif t == "neg_bus_bay":
            surface.append(Polygon(m["poly"]).buffer(0.05))
            fs.bus_stop.append(Point(m["stop"]))
            if rng.random() < 0.5:
                for d in dashes(np.asarray(m["mouth"]), 0.15, 2.0, 2.0, 0.0):
                    fs.add("white_dashed_stroke", d)

    # ---------------------------------------------------------------- 路面 → 瓦片碎片
    islands = [Polygon(m["poly"]).buffer(0) for m in marks if m["t"] == "island"]
    road = unary_union(surface)
    if islands:
        road = road.difference(unary_union(islands))
    tile = float(rng.uniform(*cfg["road_tile_m"]))
    xmin, ymin, xmax, ymax = road.bounds
    ox, oy = float(rng.uniform(0, tile)), float(rng.uniform(0, tile))
    cells = [box(x, y, x + tile, y + tile)
             for x in np.arange(xmin - ox, xmax, tile) for y in np.arange(ymin - oy, ymax, tile)]
    pieces = shapely.intersection(road, np.asarray(cells, dtype=object))
    for p in pieces:
        if p is not None and not p.is_empty and p.area > 0.05:
            polys = [q for q in getattr(p, "geoms", [p]) if q.geom_type == "Polygon" and q.area > 0.05]
            if polys:
                fs.road.insert(0, ("road_surface", MultiPolygon(polys), float("nan"), float("nan"), float("nan")))
    meta = {"arrows": arrows_meta, "neg_regions": [np.asarray(p.exterior.coords)[:-1].tolist()
                                                   for p in sc.neg_regions if p.geom_type == "Polygon"]}
    return fs, meta


def zigzag(line: np.ndarray, width: float, amp: float, period: float):
    s = cumulative_arclength(line)
    if s[-1] < 1:
        return None
    k = max(3, int(s[-1] / (period / 2)) + 1)
    tt = np.linspace(0, s[-1], k)
    base = np.stack([np.interp(tt, s, line[:, 0]), np.interp(tt, s, line[:, 1])], 1)
    nrm = np.array([[-t[1], t[0]] for t in tangents(base)])
    sign = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    return stroke(base + nrm * (amp * sign)[:, None], width)


def bus_text(l, rng):
    s = cumulative_arclength(l.points)
    if s[-1] < 12:
        return None
    x = float(rng.uniform(5, s[-1] - 6))
    p = np.array([np.interp(x, s, l.points[:, 0]), np.interp(x, s, l.points[:, 1])])
    k = int(np.searchsorted(s, x).clip(1, len(s) - 1))
    t = l.points[k, :2] - l.points[k - 1, :2]
    hd = math.atan2(t[1], t[0])
    parts = []
    for ci in range(4):
        cx = -4.5 + ci * 3.0
        for _ in range(int(rng.integers(4, 9))):
            w, h = float(rng.uniform(0.15, 0.6)), float(rng.uniform(0.15, 1.6))
            dx, dy = float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8))
            parts.append(box(cx + dx - w / 2, dy - h / 2, cx + dx + w / 2, dy + h / 2))
    return place(unary_union(parts), p, hd)


def channel_stripes(poly: Polygon, rng, cfg) -> list:
    w = float(rng.uniform(*cfg["channel_stripe_w"]))
    gap = float(rng.uniform(*cfg["channel_stripe_gap"]))
    mrr = poly.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)[:4]
    e = c[1] - c[0]
    ang = math.atan2(e[1], e[0]) + math.radians(45 if rng.random() < 0.5 else -45)
    d = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-d[1], d[0]])
    ctr = np.asarray(poly.centroid.coords[0])
    R = math.hypot(*(np.asarray(poly.bounds[2:]) - poly.bounds[:2])) + 1
    out = []
    t = -R
    while t < R:
        a = ctr + n * t
        strip = Polygon([a - d * R, a + d * R, a + d * R + n * w, a - d * R + n * w])
        x = strip.intersection(poly)
        if not x.is_empty and x.area > 0.05:
            out.append(x)
        t += w + gap
    return out


def render_parking(fs: FeatureSet, m: dict, rng, surface: list) -> None:
    lot = Polygon(m["poly"]).buffer(0)
    fs.other_area.append(("parking_background_surface", lot))
    Lx, Ly = m["size"]
    ang = m["angle"]
    c0 = np.asarray(m["center"])
    sw, sl = float(rng.uniform(2.3, 2.7)), float(rng.uniform(4.8, 5.5))
    aisle = float(rng.uniform(5.5, 7.0))
    rows = []
    y = -Ly / 2 + 0.5
    while y + sl < Ly / 2:
        rows.append(y)
        y += sl + (aisle if len(rows) % 2 == 1 else 0.0)
    code = str(rng.choice(["parking_space_surface", "offstreet_parking_space"]))
    for k, y in enumerate(rows):
        x = -Lx / 2 + 0.5
        while x + sw < Lx / 2 - 0.5:
            fs.add(code, place(box(x, y, x + sw, y + sl), c0, math.radians(ang)))
            x += sw + 0.15
        if k % 2 == 1 and y + sl + aisle / 2 < Ly / 2:
            yc = y + sl + aisle / 2
            aisle_poly = place(box(-Lx / 2, yc - aisle / 2, Lx / 2, yc + aisle / 2), c0, math.radians(ang))
            if rng.random() < 0.6:
                fs.add("road_surface", aisle_poly)
            sym = "parking_bidirectional_symbol" if rng.random() < 0.7 else "parking_oneway_symbol"
            for xs in np.arange(-Lx / 2 + 4, Lx / 2 - 3, float(rng.uniform(10, 18))):
                fs.add(sym, place(arrow_glyph(["S"], 2.0, 0.8, rng), (c0 + rot(np.array([xs, yc]), ang)),
                                  math.radians(ang)))


def rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])
