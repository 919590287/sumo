"""训练真值：任意 ROI 裁切、定长张量化与稠密辅助真值（docs/DESIGN.md §3.1-3、§4.1、§4.2 阶段 A）。

- ``crop_graph_to_roi``：把场景真值图裁切到任意（凸）ROI 多边形内：被 ROI 边界截断的 edge 在同一
  横断面统一切断（按参考车道弧长分数映射到同 edge 全部并行车道），切口生成
  ``roi_entry_port`` / ``roi_exit_port``，并记录 ``port_key``（全局 edge id）供跨块端口对比学习；
- ``graph_targets``：裁切图 → 定长真值数组（节点 / 车道 / 人行横道槽位、拓扑索引与成对连接轨迹）；
- ``dense_targets``：0.40 m/px（F_2 尺度）稠密真值：有效车道掩码、中心线热图、有符号横向距离、
  行驶方向、宽度场与 7 类锚点热图。

纯确定性的真值构造与张量化，不参与推理期裁决。
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from src.common.config import load_config
from src.common.geom import cumulative_arclength, resample_polyline, tangents
from src.graph.lane_graph import (CONN_MAX, K_CONN, K_LANE, LANE_ROLES, M_MAX, N_MAX, NODE_TYPES, P_MAX, PERMISSIONS,
                                  POLY_K, Connection, Crosswalk, Edge, Lane, LaneGraph, Node)

DENSE_CHANNELS = ("valid", "center_heat", "center_sdf", "dir_cos", "dir_sin", "width",
                  "a_junction", "a_split", "a_midblock", "a_port", "a_attr", "a_stop", "a_crosswalk")
ANCHOR_OF_TYPE = {"junction_core": "a_junction", "split_merge": "a_split", "midblock_node": "a_midblock",
                  "roi_entry_port": "a_port", "roi_exit_port": "a_port", "road_continuation": "a_attr"}


class CropError(RuntimeError):
    pass


# ====================================================================== ROI 裁切
def _sub_polyline(pts: np.ndarray, f0: float, f1: float) -> np.ndarray:
    s = cumulative_arclength(pts)
    a, b = f0 * s[-1], f1 * s[-1]
    eps = min(0.05, 0.25 * (b - a))                       # 去掉紧贴切点的原顶点，避免近重复点导致端部切向失真
    inner = (s > a + eps) & (s < b - eps)
    pa = np.array([np.interp(a, s, pts[:, 0]), np.interp(a, s, pts[:, 1])])
    pb = np.array([np.interp(b, s, pts[:, 0]), np.interp(b, s, pts[:, 1])])
    return np.vstack([pa, pts[inner, :2], pb])


def node_areas(g: LaneGraph, pad: float = 2.0) -> dict[int, Polygon]:
    """节点区域：接入车道端点（及 junction_core 轮廓）的凸包外扩 pad 米。"""
    pts: dict[int, list] = {}
    lanes = {l.id: l for l in g.lanes}
    for e in g.edges:
        for lid in e.lanes:
            pts.setdefault(e.from_node, []).append(lanes[lid].points[0, :2])
            pts.setdefault(e.to_node, []).append(lanes[lid].points[-1, :2])
    out = {}
    for n in g.nodes:
        p = list(pts.get(n.id, []))
        if n.polygon is not None:
            p += list(n.polygon[:, :2])
        if p:
            out[n.id] = shapely.convex_hull(shapely.multipoints(np.asarray(p))).buffer(pad)
    return out


def gt_region(g: LaneGraph, roi: Polygon, window_box: Polygon | None = None) -> Polygon:
    """真值裁切区域 = ROI ∪ 与 ROI 相交且完整落在窗口内的节点区域（不把路口 / 分合流 / 开口区切成半个）；
    与 ROI 相交但超出窗口（不可完整观测）的节点区域整体排除，再限于窗口内。

    输入 roi_mask 仍为原始选区；该约定使被选区边界穿过的节点在真值中保持完整拓扑
    （进口车道不因去向车道被裁掉而成为伪死胡同），与约束解码器的非零度公理一致。
    """
    region = roi
    drop = []
    for nid, a in node_areas(g).items():
        if a.intersects(roi) and not roi.contains(a):
            if window_box is None or window_box.contains(a):
                region = region.union(a)
            else:
                drop.append(a)                  # 节点区域超出窗口（不可完整观测）：整体排除，其接入 edge 在区外以端口截断
    for a in drop:
        region = region.difference(a.buffer(0.5))
    if window_box is not None:
        region = region.intersection(window_box)
    if region.geom_type == "MultiPolygon":
        region = max(region.geoms, key=lambda q: q.area)
    return region


def crop_graph_to_roi(g: LaneGraph, roi: Polygon, shift=(0.0, 0.0), min_piece: float = 0.5) -> LaneGraph:
    """g（任意局部坐标）→ 裁切到 roi（同一坐标系）后平移 ``-shift`` 的窗口局部坐标子图。"""
    sx, sy = shift
    lanes = {l.id: l for l in g.lanes}
    nodes = {n.id: n for n in g.nodes}
    proi = prep(roi)
    eps = 1e-6
    new_nodes: dict[int, Node] = {}
    new_lanes, new_edges = [], []
    next_id = max(n.id for n in g.nodes) + 1
    cut_start, cut_end = set(), set()

    def keep(nid):
        if nid not in new_nodes:
            n = nodes[nid]
            pos = n.pos.copy()
            pos[:2] -= (sx, sy)
            poly = None if n.polygon is None else n.polygon - (sx, sy)
            new_nodes[nid] = Node(nid, pos, n.node_type, poly)
        return nid

    for e in g.edges:
        ref = lanes[e.lanes[len(e.lanes) // 2]].points[:, :2]
        line = LineString(ref)
        if proi.contains(line):
            f0, f1 = 0.0, 1.0
        else:
            inter = line.intersection(roi)
            if inter.is_empty:
                continue
            parts = [p for p in getattr(inter, "geoms", [inter]) if p.geom_type == "LineString" and p.length > 0]
            if not parts:
                continue
            p = max(parts, key=lambda q: q.length)
            L = line.length
            d0 = line.project(Point(p.coords[0])) / L
            d1 = line.project(Point(p.coords[-1])) / L
            f0, f1 = min(d0, d1), max(d0, d1)
        if (f1 - f0) * line.length < min_piece:
            continue
        sc, ec = f0 > eps, f1 < 1 - eps
        pieces = {lid: _sub_polyline(lanes[lid].points[:, :2], f0, f1) for lid in e.lanes}
        if sc:
            p = np.mean([v[0] for v in pieces.values()], 0) - (sx, sy)
            fn = next_id
            next_id += 1
            new_nodes[fn] = Node(fn, np.r_[p, 0.0], "roi_entry_port")
            new_nodes[fn].port_key = e.id
        else:
            fn = keep(e.from_node)
        if ec:
            p = np.mean([v[-1] for v in pieces.values()], 0) - (sx, sy)
            tn = next_id
            next_id += 1
            new_nodes[tn] = Node(tn, np.r_[p, 0.0], "roi_exit_port")
            new_nodes[tn].port_key = e.id
        else:
            tn = keep(e.to_node)
        for lid, pts in pieces.items():
            l = lanes[lid]
            loc = resample_polyline(pts - (sx, sy), K_LANE)
            new_lanes.append(Lane(lid, loc, l.width, l.role, l.permission, l.allow_change_left,
                                  l.allow_change_right, l.is_stop_line_end and not ec))
            if sc:
                cut_start.add(lid)
            if ec:
                cut_end.add(lid)
        new_edges.append(Edge(e.id, list(e.lanes), fn, tn))
    kept = {l.id for l in new_lanes}
    conns = []
    for c in g.connections:
        if c.from_lane in kept and c.to_lane in kept and c.from_lane not in cut_end and c.to_lane not in cut_start:
            pts = c.points.copy()
            pts[:, :2] -= (sx, sy)
            conns.append(Connection(c.from_lane, c.to_lane, pts))
    eids = {e.id for e in new_edges}
    ne = {e.id: e for e in new_edges}
    cws = []
    for cw in g.crosswalks:
        inc = all(x in eids and cw.node in (ne[x].from_node, ne[x].to_node) for x in cw.edges)
        if cw.node in new_nodes and inc and all(proi.contains(Point(p)) for p in cw.endpoints[:, :2]):
            ep = cw.endpoints.copy()
            ep[:, :2] -= (sx, sy)
            cws.append(Crosswalk(cw.id, ep, cw.width, cw.node, list(cw.edges)))
    used = {e.from_node for e in new_edges} | {e.to_node for e in new_edges}
    out = LaneGraph([n for n in new_nodes.values() if n.id in used], new_lanes, new_edges, conns, cws,
                    origin=g.origin, meta=dict(g.meta))
    errs = out.validate()
    if errs:
        raise CropError("裁切子图不合法：" + "; ".join(errs[:3]))
    return out


# ====================================================================== 定长真值
def graph_targets(g: LaneGraph) -> dict[str, np.ndarray]:
    P, N, M, K = P_MAX, N_MAX, M_MAX, CONN_MAX
    if len(g.nodes) > P or len(g.lanes) > N or len(g.crosswalks) > M or len(g.connections) > K:
        raise CropError("裁切子图实体数超过单窗上限")
    nidx = {n.id: k for k, n in enumerate(g.nodes)}
    lidx = {l.id: k for k, l in enumerate(g.lanes)}
    t = {
        "node_pos": np.zeros((P, 2), np.float32), "node_type": np.full(P, -1, np.int64),
        "node_poly": np.zeros((P, POLY_K, 2), np.float32), "node_valid": np.zeros(P, bool),
        "port_key": np.full(P, -1, np.int64),
        "lane_pts": np.zeros((N, K_LANE, 2), np.float32), "lane_w": np.zeros(N, np.float32),
        "lane_role": np.full(N, -1, np.int64), "lane_perm": np.full(N, -1, np.int64),
        "lane_lc": np.zeros((N, 2), np.float32), "lane_stop": np.zeros(N, np.float32),
        "lane_valid": np.zeros(N, bool),
        "lane_from": np.full(N, -1, np.int64), "lane_to": np.full(N, -1, np.int64),
        "E_adj": np.zeros((N, N), np.uint8), "E_succ": np.zeros((N, N), np.uint8),
        "cw_end": np.zeros((M, 2, 2), np.float32), "cw_w": np.zeros(M, np.float32), "cw_valid": np.zeros(M, bool),
        "E_cross": np.zeros((M, N), np.uint8), "cw_node": np.full(M, -1, np.int64),
        "conn_ij": np.zeros((K, 2), np.int64), "conn_pts": np.zeros((K, K_CONN, 2), np.float32),
        "conn_valid": np.zeros(K, bool),
    }
    for k, n in enumerate(g.nodes):
        t["node_pos"][k] = n.pos[:2]
        t["node_type"][k] = NODE_TYPES.index(n.node_type)
        t["node_valid"][k] = True
        if n.polygon is not None:
            t["node_poly"][k] = canonical_ring(n.polygon)
        t["port_key"][k] = getattr(n, "port_key", -1)
    for k, l in enumerate(g.lanes):
        t["lane_pts"][k] = l.points[:, :2]
        t["lane_w"][k] = l.width
        t["lane_role"][k] = LANE_ROLES.index(l.role)
        t["lane_perm"][k] = PERMISSIONS.index(l.permission)
        t["lane_lc"][k] = (float(l.allow_change_left), float(l.allow_change_right))
        t["lane_stop"][k] = float(l.is_stop_line_end)
        t["lane_valid"][k] = True
    for e in g.edges:
        for a, b in zip(e.lanes[:-1], e.lanes[1:]):
            t["E_adj"][lidx[a], lidx[b]] = 1
        for lid in e.lanes:
            t["lane_from"][lidx[lid]] = nidx[e.from_node]
            t["lane_to"][lidx[lid]] = nidx[e.to_node]
    for k, c in enumerate(g.connections):
        i, j = lidx[c.from_lane], lidx[c.to_lane]
        t["E_succ"][i, j] = 1
        t["conn_ij"][k] = (i, j)
        t["conn_pts"][k] = c.points[:, :2]
        t["conn_valid"][k] = True
    e_of = {e.id: e for e in g.edges}
    for k, cw in enumerate(g.crosswalks):
        t["cw_end"][k] = cw.endpoints[:, :2]
        t["cw_w"][k] = cw.width
        t["cw_valid"][k] = True
        t["cw_node"][k] = nidx[cw.node]
        for eid in cw.edges:
            for lid in e_of[eid].lanes:
                t["E_cross"][k, lidx[lid]] = 1
    return t


def canonical_ring(poly: np.ndarray) -> np.ndarray:
    """轮廓规范化：逆时针，起点为最小方位角（相对质心）的顶点——仅用于真值 / 匹配的确定性约定。"""
    p = np.asarray(poly, np.float64)[:, :2]
    x, y = p[:, 0], p[:, 1]
    if 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        p = p[::-1]
    c = p.mean(0)
    k = int(np.argmin(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])))
    return np.roll(p, -k, axis=0)


# ====================================================================== 稠密真值（F_2 尺度）
def dense_targets(g: LaneGraph, H: int, W: int, res: float, stride: int = 2, sigma_center_m: float = 0.4,
                  sigma_anchor_m: float = 1.0, sdf_trunc_m: float = 2.0) -> np.ndarray:
    """返回 [13, H/stride, W/stride] float32。像素 (r, c) ↔ 窗口局部米制 (x, y)：x = (c+0.5)·r2，y = (h−r−0.5)·r2。"""
    h, w = H // stride, W // stride
    r2 = res * stride
    out = np.zeros((len(DENSE_CHANNELS), h, w), np.float32)
    ch = {n: k for k, n in enumerate(DENSE_CHANNELS)}

    def px(pts):
        p = np.asarray(pts, np.float64)[..., :2]
        return np.stack([p[..., 0] / r2, h - p[..., 1] / r2], -1)

    lanes = {l.id: l for l in g.lanes}
    valid = np.zeros((h, w), np.uint8)
    for l in g.lanes:
        poly = LineString(l.points[:, :2]).buffer(l.width / 2, cap_style="flat")
        _fill(valid, poly, px)
    for c in g.connections:
        poly = LineString(c.points[:, :2]).buffer(lanes[c.from_lane].width / 2)
        _fill(valid, poly, px)
    out[ch["valid"]] = valid

    # 中心线（L：热图）与 L∪K（方向 / 宽度 / 有符号距离）
    zero_l = np.ones((h, w), np.uint8)
    zero_all = np.ones((h, w), np.uint8)
    tcos = np.zeros((h, w), np.float32)
    tsin = np.zeros((h, w), np.float32)
    wmap = np.zeros((h, w), np.float32)
    lines = [(l.points[:, :2], l.width, True) for l in g.lanes] + \
            [(c.points[:, :2], lanes[c.from_lane].width, False) for c in g.connections]
    for pts, wd, is_lane in lines:
        dense = resample_polyline(pts, max(2, int(cumulative_arclength(pts)[-1] / (r2 * 0.5)) + 2))
        tg = tangents(dense)
        q = px(dense) - 0.5                                 # 像素中心约定（cv2 以整数坐标为像素中心）
        for k in range(len(q) - 1):
            a = tuple(np.round(q[k] * 16).astype(int))
            b = tuple(np.round(q[k + 1] * 16).astype(int))
            cv2.line(zero_all, a, b, 0, 1, cv2.LINE_8, 4)
            cv2.line(tcos, a, b, float(tg[k, 0]), 1, cv2.LINE_8, 4)
            cv2.line(tsin, a, b, float(tg[k, 1]), 1, cv2.LINE_8, 4)
            cv2.line(wmap, a, b, float(wd), 1, cv2.LINE_8, 4)
            if is_lane:
                cv2.line(zero_l, a, b, 0, 1, cv2.LINE_8, 4)
    if (zero_l == 0).any():
        d_l = cv2.distanceTransform(zero_l, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out[ch["center_heat"]] = np.exp(-(d_l * r2) ** 2 / (2 * sigma_center_m ** 2))
    if (zero_all == 0).any():
        d, lab = cv2.distanceTransformWithLabels(zero_all, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(zero_all == 0)
        lut_y = np.zeros(lab.max() + 1, np.int64)
        lut_x = np.zeros(lab.max() + 1, np.int64)
        lut_y[lab[ys, xs]] = ys
        lut_x[lab[ys, xs]] = xs
        ny, nx = lut_y[lab], lut_x[lab]
        tc, ts = tcos[ny, nx], tsin[ny, nx]
        rr, cc = np.mgrid[0:h, 0:w]
        vx = (cc - nx).astype(np.float32)
        vy = -(rr - ny).astype(np.float32)                  # 行向下 → y 向上
        sign = np.sign(tc * vy - ts * vx)
        sdf = np.clip(sign * d * r2, -sdf_trunc_m, sdf_trunc_m) / sdf_trunc_m
        m = valid > 0
        out[ch["center_sdf"]] = np.where(m, sdf, 0)
        out[ch["dir_cos"]] = np.where(m, tc, 0)
        out[ch["dir_sin"]] = np.where(m, ts, 0)
        out[ch["width"]] = np.where(m, wmap[ny, nx] / 4.0, 0)

    # 锚点热图
    anchors: dict[str, list] = {k: [] for k in DENSE_CHANNELS if k.startswith("a_")}
    for n in g.nodes:
        anchors[ANCHOR_OF_TYPE[n.node_type]].append(n.pos[:2])
    e_of = g.edge_of_lane()
    nt = {n.id: n.node_type for n in g.nodes}
    for l in g.lanes:
        if l.is_stop_line_end and nt[e_of[l.id].to_node] in ("junction_core", "midblock_node"):
            anchors["a_stop"].append(l.points[-1, :2])
    for cw in g.crosswalks:
        anchors["a_crosswalk"].append(cw.endpoints[:, :2].mean(0))
    sp = sigma_anchor_m / r2
    rad = int(math.ceil(3 * sp))
    for k, pts in anchors.items():
        img = out[ch[k]]
        for p in pts:
            c, r = px(np.asarray(p))
            c0, r0 = int(round(c - 0.5)), int(round(r - 0.5))
            ys = slice(max(r0 - rad, 0), min(r0 + rad + 1, h))
            xs = slice(max(c0 - rad, 0), min(c0 + rad + 1, w))
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            yy, xx = np.mgrid[ys, xs]
            gk = np.exp(-(((xx + 0.5) - c) ** 2 + ((yy + 0.5) - r) ** 2) / (2 * sp ** 2))
            np.maximum(img[ys, xs], gk, out=img[ys, xs])
    return out


def _fill(img: np.ndarray, poly, px) -> None:
    for p in getattr(poly, "geoms", [poly]):
        if p.is_empty or p.geom_type != "Polygon":
            continue
        ring = np.round((px(np.asarray(p.exterior.coords)) - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(img, [ring], 1, cv2.LINE_8, 4)
