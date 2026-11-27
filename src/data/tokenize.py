"""关键实体稀疏矢量 Token 化（docs/DESIGN.md §3.2 分支 B）。

每个 Token：类别 id、P=8 个局部米制点 (x, y) 与切向 (dx, dy)、宽/长、箭头朝向 (cos θ, sin θ)、
6 维转向多标签 [L, S, R, U, FL, FR]。长折线按 ≤ chunk_max_len 分段。单窗上限 M_vec ≤ 512。
纯确定性几何编码（中心线抽取为细长漆面多边形的双侧链等弧长平均），无任何裁决规则。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString

from src.common.config import load_config
from src.common.geom import cumulative_arclength, resample_polyline, tangents
from src.data.rasterize import TURN_ORDER, polygons_of
from src.data.roi import RoiFeatures

TOKEN_CLASSES = (
    "arrow", "white_dashed", "white_solid", "white_double_solid", "white_ldash_rsolid", "white_lsolid_rdash",
    "yellow_solid", "yellow_lsolid_rdash", "yellow_double_dashed", "bus_lane_marking", "variable_lane_marking",
    "reversible_lane_marking", "road_boundary", "separation_path", "crosswalk_area", "traffic_island",
    "channelization_area", "bicycle_symbol", "greenway_boundary", "bus_stop", "parking_symbol",
)
# 超出上限时：箭头 / 斑马线 / 岛屿 / 导流区恒保留，其余按距 ROI 质心由近及远保留
_PRIORITY = {c: 1 for c in TOKEN_CLASSES}
_PRIORITY.update({"arrow": 0, "crosswalk_area": 0, "traffic_island": 0, "channelization_area": 0})

MARKING_CODES = {
    "white_dashed_stroke": "white_dashed", "white_dashed": "white_dashed", "white_solid_stroke": "white_solid",
    "white_double_solid": "white_double_solid", "white_left_dashed_right_solid": "white_ldash_rsolid",
    "white_left_solid_right_dashed": "white_lsolid_rdash", "yellow_solid_stroke": "yellow_solid",
    "yellow_left_solid_right_dashed": "yellow_lsolid_rdash", "yellow_double_dashed": "yellow_double_dashed",
    "bus_lane_marking": "bus_lane_marking", "variable_lane_marking": "variable_lane_marking",
    "reversible_lane_marking": "reversible_lane_marking", "greenway_boundary": "greenway_boundary",
}


@dataclass
class VectorTokens:
    cls: np.ndarray        # [M] int64
    pts: np.ndarray        # [M, P, 2] 局部米制坐标
    tan: np.ndarray        # [M, P, 2]
    size: np.ndarray       # [M, 2] (宽, 长)
    heading: np.ndarray    # [M, 2] (cos θ, sin θ)，非箭头为 0
    turns: np.ndarray      # [M, 6]
    mask: np.ndarray       # [M] bool

    @property
    def n(self) -> int:
        return int(self.mask.sum())


def thin_centerline(poly) -> tuple[np.ndarray, float]:
    """细长漆面多边形 → (稠密中心线, 平均漆宽)：外环在近似直径两端点处切成两条侧链，
    各自按等弧长重采样后取逐点均值。"""
    ring = np.asarray(poly.exterior.coords)[:-1]
    c = ring.mean(0)
    a = int(np.argmax(np.linalg.norm(ring - c, axis=1)))
    b = int(np.argmax(np.linalg.norm(ring - ring[a], axis=1)))
    i, j = sorted((a, b))
    side1 = ring[i:j + 1]
    side2 = np.vstack([ring[j:], ring[:i + 1]])[::-1]
    if len(side1) < 2 or len(side2) < 2:
        return np.stack([ring[a], ring[b]]), 0.0
    L = max(np.linalg.norm(ring[a] - ring[b]), 1e-3)
    k = max(2, int(np.ceil(L / 0.5)) + 1)
    s1 = resample_polyline(side1, k) if cumulative_arclength(side1)[-1] > 0 else np.repeat(side1[:1], k, 0)
    s2 = resample_polyline(side2, k) if cumulative_arclength(side2)[-1] > 0 else np.repeat(side2[:1], k, 0)
    width = float(np.mean(np.linalg.norm(s1 - s2, axis=1)))
    return (s1 + s2) / 2, width


def envelope_axis(geom) -> tuple[np.ndarray, float]:
    """要素定向最小外接矩形的长轴（两端短边中点连线）及短边长度。"""
    with np.errstate(all="ignore"):
        r = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)[:4]
    e0, e1 = np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[2] - r[1])
    if e0 >= e1:
        a, b, w = (r[1] + r[2]) / 2, (r[3] + r[0]) / 2, e1
    else:
        a, b, w = (r[0] + r[1]) / 2, (r[2] + r[3]) / 2, e0
    return np.stack([b, a]), float(w)


def _chunks(line: np.ndarray, max_len: float, P: int) -> list[np.ndarray]:
    s = cumulative_arclength(line)
    if s[-1] <= 1e-6:
        return []
    n = max(1, int(np.ceil(s[-1] / max_len)))
    out = []
    for k in range(n):
        a, b = s[-1] * k / n, s[-1] * (k + 1) / n
        t = np.linspace(a, b, 4 * P)
        seg = np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], 1)
        out.append(resample_polyline(seg, P))
    return out


def tokenize(feat: RoiFeatures, cfg: dict | None = None) -> VectorTokens:
    cfg = cfg or load_config("data")
    tc = cfg["tokens"]
    P, Mx, max_len = tc["points_per_token"], tc["max_tokens"], tc["chunk_max_len_m"]
    win = feat.window
    ox, oy = win.origin
    xmin, ymin, xmax, ymax = win.bounds
    rows = []   # (priority, dist, cls, pts, size, heading, turns)
    ctr = np.array(win.roi.centroid.coords[0])

    def add(cls, pts_global, size=(0.0, 0.0), heading=(0.0, 0.0), turns=None):
        d = float(np.linalg.norm(pts_global.mean(0) - ctr))
        rows.append((_PRIORITY[cls], d, TOKEN_CLASSES.index(cls), pts_global - (ox, oy), size, heading,
                     np.zeros(6) if turns is None else turns))

    def clip(g):
        return shapely.clip_by_rect(g, xmin, ymin, xmax, ymax)

    road = feat.road
    turn_map = cfg["arrow_turns"]
    for code, geom, hd, w, l in zip(road["分类代码"].values, road.geometry.values, road["heading"].values,
                                    road["显示宽度米"].values, road["显示长度米"].values):
        if code in turn_map:
            c = np.asarray(geom.centroid.coords[0])
            t = np.array([1.0 if k in turn_map[code] else 0.0 for k in TURN_ORDER])
            h = (np.cos(hd), np.sin(hd)) if np.isfinite(hd) else (0.0, 0.0)
            add("arrow", np.repeat(c[None], P, 0), (float(w or 0), float(l or 0)), h, t)
        elif code in MARKING_CODES:
            g = clip(geom)
            parts = [p for p in polygons_of(g) if p.area > 0]
            if not parts:
                continue
            if len(parts) == 1:          # 单段漆面：双侧链中心线
                line, width = thin_centerline(parts[0])
            else:                        # 多部件要素（文字 / 混合实虚线）：整要素定向包络长轴
                line, width = envelope_axis(shapely.MultiPolygon(parts))
            for ch in _chunks(line, max_len, P):
                add(MARKING_CODES[code], ch, (width, float(cumulative_arclength(ch)[-1])))
        elif code in ("crosswalk_area", "traffic_island", "channelization_area"):
            for p in polygons_of(clip(geom)):
                ring = resample_polyline(np.asarray(p.exterior.coords), P)
                with np.errstate(all="ignore"):
                    mrr = p.minimum_rotated_rectangle
                e = np.diff(np.asarray(mrr.exterior.coords)[:3], axis=0) if mrr.geom_type == "Polygon" else np.zeros((2, 2))
                dims = sorted(np.linalg.norm(e, axis=1))
                add(code, ring, (float(dims[0]), float(dims[1])))
        elif code == "bicycle_lane_symbol":
            c = np.asarray(geom.centroid.coords[0])
            add("bicycle_symbol", np.repeat(c[None], P, 0))
        elif code in ("parking_bidirectional_symbol", "parking_oneway_symbol"):
            c = np.asarray(geom.centroid.coords[0])
            add("parking_symbol", np.repeat(c[None], P, 0))
    rs = road.geometry.values[road["分类代码"].values == "road_surface"]
    if len(rs):
        bnd = clip(shapely.union_all(shapely.make_valid(shapely.clip_by_rect(shapely.make_valid(rs), xmin - 2, ymin - 2,
                                                                              xmax + 2, ymax + 2))).boundary)
        for ln in _lines(bnd):
            for ch in _chunks(ln, max_len, P):
                add("road_boundary", ch)
    for g in feat.separation.geometry.values:
        for ln in _lines(clip(g)):
            for ch in _chunks(ln, max_len, P):
                add("separation_path", ch)
    for g in feat.bus_stop.geometry.values:
        c = np.asarray(g.coords[0])[:2]
        add("bus_stop", np.repeat(c[None], P, 0))

    rows.sort(key=lambda r: (r[0], r[1]))
    rows = rows[:Mx]
    M = Mx
    out = VectorTokens(np.zeros(M, np.int64), np.zeros((M, P, 2), np.float32), np.zeros((M, P, 2), np.float32),
                       np.zeros((M, 2), np.float32), np.zeros((M, 2), np.float32), np.zeros((M, 6), np.float32),
                       np.zeros(M, bool))
    for k, (_, _, c, pts, size, hd, turns) in enumerate(rows):
        out.cls[k] = c
        out.pts[k] = pts
        if np.ptp(pts, axis=0).max() > 1e-6:
            out.tan[k] = tangents(pts)
        out.size[k] = size
        out.heading[k] = hd
        out.turns[k] = turns
        out.mask[k] = True
    return out


def _lines(g) -> list[np.ndarray]:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [np.asarray(g.coords)[:, :2]]
    if g.geom_type == "LinearRing":
        return [np.asarray(g.coords)[:, :2]]
    if hasattr(g, "geoms"):
        out = []
        for x in g.geoms:
            out += _lines(x)
        return out
    return []
