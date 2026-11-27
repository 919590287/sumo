"""显式负样本放置（docs/DESIGN.md §5 Layer 1-2）：无标线细小支路 / 园区出入口、停车场、
并入 road_surface 的非机动车道、港湾式公交站。

全部只渲染进输入（BEV 与矢量 Token），在真值图 G* 与 M_valid* 中置空；
其区域多边形另存为 ``neg_regions`` 供合成评测统计误建率（FPR）。
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

from src.graph.lane_graph import LaneGraph


def _occupied(st, g: LaneGraph):
    polys = [st._lane_poly(l) for l in g.lanes]
    lanes = {l.id: l for l in g.lanes}
    for (i, j), d in st.dense_conn.items():
        polys.append(LineString(d).buffer(lanes[i].width / 2))
    for n in g.nodes:
        if n.polygon is not None:
            polys.append(Polygon(n.polygon).buffer(0))
    for r in st.islands + st.channels:
        polys.append(Polygon(r).buffer(0))
    for cw in g.crosswalks:
        polys.append(LineString(cw.endpoints[:, :2]).buffer(cw.width / 2, cap_style="flat"))
    return unary_union(polys)


def _carriageways(st):
    """所有臂上可挂接负样本的恒宽段：(arm, sec, inbound)。"""
    out = []
    for a in st.arms_all:
        for s in a.in_secs:
            out.append((a, s, True))
        for s in a.out_secs:
            out.append((a, s, False))
    return out


def _outer(a, s, inbound: bool, x: float, extra: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """恒宽段外侧路缘上轴向坐标 x 处的点、行驶切向与向外法向。"""
    W = sum(s.widths)
    lat = (s.left_off + W + extra) if inbound else -(s.left_off + W + extra)
    p = a.point(x, lat)
    t = a.tangent(x) * (-1 if inbound else 1)
    nrm_out = a.n * (1 if inbound else -1)
    t_axis = a.tangent(x)
    nrm_out = np.array([-t_axis[1], t_axis[0]]) * (1 if inbound else -1)
    return p, t, nrm_out


def place_negatives(st, g: LaneGraph) -> None:
    rng = st.rng
    cfg = st.cfg["negatives"]
    occ = _occupied(st, g)
    cws = _carriageways(st)
    if not cws:
        return
    kinds = []
    for k, p in (("side_road", cfg["side_road_prob"]), ("parking", cfg["parking_prob"]),
                 ("nonmotor", cfg["nonmotor_prob"]), ("bus_bay", cfg["bus_bay_prob"])):
        if rng.random() < p:
            kinds.append(k)
            if k == "side_road" and rng.random() < 0.4:
                kinds.append(k)
    rng.shuffle(kinds)
    placed = 0
    for kind in kinds[: cfg["max_per_scene"]]:
        for _ in range(6):
            a, s, inbound = cws[int(rng.integers(len(cws)))]
            lo, hi = min(s.s_a, s.s_b), max(s.s_a, s.s_b)
            if hi - lo < 20:
                continue
            fn = {"side_road": _side_road, "parking": _parking, "nonmotor": _nonmotor, "bus_bay": _bus_bay}[kind]
            res = fn(st, rng, cfg, a, s, inbound, lo, hi)
            if res is None:
                continue
            region, mark = res
            if region.buffer(-0.25).intersects(occ):
                continue
            st.negs.append(region)
            st.marks.append(mark)
            occ = occ.union(region)
            placed += 1
            break


def _side_road(st, rng, cfg, a, s, inbound, lo, hi):
    x = float(rng.uniform(lo + 6, hi - 6))
    p, t, nrm = _outer(a, s, inbound, x)
    w = float(rng.uniform(*cfg["side_road_width"]))
    L = float(rng.uniform(*cfg["side_road_len"]))
    ang = math.radians(float(rng.uniform(-30, 30)))
    d = np.array([nrm[0] * math.cos(ang) - nrm[1] * math.sin(ang), nrm[0] * math.sin(ang) + nrm[1] * math.cos(ang)])
    q = np.array([-d[1], d[0]])
    ring = np.array([p - q * w / 2, p + q * w / 2, p + q * w / 2 + d * L, p - q * w / 2 + d * L])
    poly = Polygon(ring).buffer(0)
    gate = bool(rng.random() < 0.3)
    return poly, {"t": "neg_road", "poly": np.asarray(poly.exterior.coords)[:-1], "gate": gate,
                  "gate_line": np.array([p - q * w / 2 + d * 3.0, p + q * w / 2 + d * 3.0])}


def _parking(st, rng, cfg, a, s, inbound, lo, hi):
    x = float(rng.uniform(lo + 8, hi - 8))
    gap = float(rng.uniform(1.0, 6.0))
    p, t, nrm = _outer(a, s, inbound, x)
    Lx = float(rng.uniform(20, min(45, hi - lo)))
    Ly = float(rng.uniform(15, 35))
    c0 = p + nrm * (gap + Ly / 2)
    ang = math.degrees(math.atan2(t[1], t[0]))
    lot = rotate(box(-Lx / 2, -Ly / 2, Lx / 2, Ly / 2), ang, origin=(0, 0), use_radians=False)
    lot = translate(lot, *c0)
    return lot, {"t": "neg_parking", "poly": np.asarray(lot.exterior.coords)[:-1], "angle": ang,
                 "center": c0, "size": (Lx, Ly)}


def _nonmotor(st, rng, cfg, a, s, inbound, lo, hi):
    if any(m.get("t") == "slip_edge" for m in st.marks) and rng.random() < 0.5:
        return None
    w = float(rng.uniform(*cfg["nonmotor_width"]))
    x0 = lo + float(rng.uniform(0.5, 4.0))
    x1 = hi - float(rng.uniform(0.5, 4.0))
    xs = np.linspace(x0, x1, max(8, int((x1 - x0) / 1.0)))
    sep = float(rng.uniform(0.0, 0.5))
    inner = np.array([_outer(a, s, inbound, x, sep)[0] for x in xs])
    outer = np.array([_outer(a, s, inbound, x, sep + w)[0] for x in xs])
    poly = Polygon(np.vstack([inner, outer[::-1]])).buffer(0)
    if poly.geom_type != "Polygon":
        return None
    center = (inner + outer) / 2
    sep_line = np.array([_outer(a, s, inbound, x, sep / 2)[0] for x in xs])
    return poly, {"t": "neg_nonmotor", "poly": np.asarray(poly.exterior.coords)[:-1], "center": center,
                  "outer": outer, "sep_line": sep_line, "sep": "separation" if rng.random() < 0.45 else "white_solid",
                  "greenway": bool(rng.random() < 0.5), "width": w, "sep_gap": sep}


def _bus_bay(st, rng, cfg, a, s, inbound, lo, hi):
    L = float(rng.uniform(25, 40))
    if hi - lo < L + 6:
        return None
    x0 = float(rng.uniform(lo + 3, hi - L - 3))
    depth = float(rng.uniform(2.5, 3.5))
    taper = float(rng.uniform(6, 10))
    xs = np.array([x0, x0 + taper, x0 + L - taper, x0 + L])
    ds = np.array([0.0, depth, depth, 0.0])
    if inbound:
        xs = xs[::-1]
    pts = [_outer(a, s, inbound, x, d)[0] for x, d in zip(xs, ds)]
    base = [_outer(a, s, inbound, x, 0.0)[0] for x in xs[::-1]]
    poly = Polygon(np.vstack([pts, base])).buffer(0)
    if poly.geom_type != "Polygon" or poly.area < 20:
        return None
    stop = _outer(a, s, inbound, (xs[1] + xs[2]) / 2, depth * 0.5)[0]
    return poly, {"t": "neg_bus_bay", "poly": np.asarray(poly.exterior.coords)[:-1], "stop": stop,
                  "mouth": np.array([pts[0], pts[-1]])}
