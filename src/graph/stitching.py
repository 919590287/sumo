"""超大选区确定性分块与端口拼接（docs/DESIGN.md §3.1-4）。

- ``plan_tiles``：把任意 ROI 外包框按固定 32 m 重叠（两侧各 16 m 上下文缓冲）均匀划分为
  净核心区（core）互不重叠的窗口，每窗含缓冲 ≤ 204.8 m；
- ``crop_graph``：把全局图裁切为某窗口核心区内的子图，并在核心边界生成
  roi_entry_port / roi_exit_port 端口（训练期跨块端口对比样本 + 本模块单测真值）；
- ``stitch_graphs``：按端口全局 UTM 坐标 + port_embed 余弦相似度构造代价矩阵，
  Hungarian 求解一一匹配，把匹配端口两侧的 edge 逐车道首尾拼接为一条 edge。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, box

from src.common.config import load_config
from src.common.geom import resample_polyline
from src.graph.lane_graph import (K_LANE, PORT_DIM, Connection, Crosswalk, Edge, Lane, LaneGraph, Node)


@dataclass
class Tile:
    ix: int
    iy: int
    core: tuple[float, float, float, float]     # 全局 UTM (xmin, ymin, xmax, ymax)
    window: tuple[float, float, float, float]   # core ± 缓冲
    origin: tuple[float, float]                 # 窗口左下角取整全局偏移 (X_0, Y_0)
    shape_px: tuple[int, int]                   # (H, W)，已 padding 至 32 倍数


def plan_tiles(roi_bounds, cfg: dict | None = None) -> list[Tile]:
    cfg = cfg or load_config("tiling")
    res, buf = cfg["resolution_m"], cfg["context_buffer_m"]
    max_core = cfg["max_window_px"] * res - 2 * buf
    xmin, ymin, xmax, ymax = map(float, roi_bounds)
    nx = max(1, math.ceil((xmax - xmin) / max_core - 1e-9))
    ny = max(1, math.ceil((ymax - ymin) / max_core - 1e-9))
    sx, sy = (xmax - xmin) / nx, (ymax - ymin) / ny
    tiles = []
    mult = cfg["pad_multiple"]
    for iy in range(ny):
        for ix in range(nx):
            core = (xmin + ix * sx, ymin + iy * sy, xmin + (ix + 1) * sx, ymin + (iy + 1) * sy)
            win = (core[0] - buf, core[1] - buf, core[2] + buf, core[3] + buf)
            origin = (math.floor(win[0]), math.floor(win[1]))
            w_px = math.ceil((win[2] - origin[0]) / res)
            h_px = math.ceil((win[3] - origin[1]) / res)
            shape = (math.ceil(h_px / mult) * mult, math.ceil(w_px / mult) * mult)
            tiles.append(Tile(ix, iy, core, win, origin, shape))
    return tiles


def plan_tiles_avoiding(roi_bounds, avoid: list, cfg: dict | None = None, search_m: float | None = None,
                        step_m: float | None = None, min_core_m: float | None = None) -> list[Tile]:
    """在 ``plan_tiles`` 的等分切线基础上，于 ±search_m 内平移各切线，使切线不穿过 ``avoid`` 中的区域
    （由模型第一遍推理输出的节点区域给出），且每块净核心仍 ≤ 单窗上限。纯确定性分块几何（I/O），
    不涉及任何道路几何裁决；找不到可行位置时保留等分切线。"""
    cfg = cfg or load_config("tiling")
    av = cfg.get("avoid_nodes", {})
    search_m = float(av.get("search_m", 40.0) if search_m is None else search_m)
    step_m = float(av.get("step_m", 1.0) if step_m is None else step_m)
    min_core_m = float(av.get("min_core_m", 20.0) if min_core_m is None else min_core_m)
    res, buf = cfg["resolution_m"], cfg["context_buffer_m"]
    max_core = cfg["max_window_px"] * res - 2 * buf
    xmin, ymin, xmax, ymax = map(float, roi_bounds)
    base = plan_tiles(roi_bounds, cfg)
    nx = len({t.ix for t in base})
    ny = len({t.iy for t in base})

    def cuts(lo, hi, n, axis):
        c = [lo + (hi - lo) * k / n for k in range(1, n)]
        out = []
        prev = lo
        for k, x in enumerate(c):
            nxt_limit = hi if k == len(c) - 1 else None
            cands = sorted(np.arange(-search_m, search_m + 1e-9, step_m), key=abs)
            chosen = x
            for d in cands:
                xx = x + d
                if xx - prev > max_core or xx - prev < min_core_m:
                    continue
                if nxt_limit is not None and (nxt_limit - xx > max_core or nxt_limit - xx < min_core_m):
                    continue
                if k < len(c) - 1 and c[k + 1] - xx > max_core:
                    continue
                line = LineString([(xx, ymin), (xx, ymax)] if axis == 0 else [(xmin, xx), (xmax, xx)])
                if not any(a.intersects(line) for a in avoid):
                    chosen = xx
                    break
            out.append(chosen)
            prev = chosen
        return [lo] + out + [hi]

    xs = cuts(xmin, xmax, nx, 0)
    ys = cuts(ymin, ymax, ny, 1)
    tiles = []
    mult = cfg["pad_multiple"]
    for iy in range(ny):
        for ix in range(nx):
            core = (xs[ix], ys[iy], xs[ix + 1], ys[iy + 1])
            win = (core[0] - buf, core[1] - buf, core[2] + buf, core[3] + buf)
            origin = (math.floor(win[0]), math.floor(win[1]))
            w_px = math.ceil((win[2] - origin[0]) / res)
            h_px = math.ceil((win[3] - origin[1]) / res)
            tiles.append(Tile(ix, iy, core, win, origin,
                              (math.ceil(h_px / mult) * mult, math.ceil(w_px / mult) * mult)))
    return tiles


# ====================================================================== 裁切（真值/单测）
def _port_embed(key: tuple, rng_noise: np.random.Generator | None, noise: float) -> np.ndarray:
    """确定性端口嵌入（以全局 edge 与切口序号为种子；仅用于构造真值与单测）。"""
    seed = abs(hash(key)) % (2**32)
    v = np.random.default_rng(seed).standard_normal(PORT_DIM)
    if rng_noise is not None and noise > 0:
        v = v + noise * rng_noise.standard_normal(PORT_DIM)
    return v / np.linalg.norm(v)


def crop_graph(g: LaneGraph, tile: Tile, embed_noise: float = 0.0, seed: int = 0) -> LaneGraph:
    """全局坐标图 g（origin 视为全局偏移）→ 裁切到 tile.core 的局部坐标子图。"""
    gx, gy = g.origin
    ox, oy = tile.origin
    core = box(*tile.core)
    rng = np.random.default_rng(seed)
    lanes_by_id = {l.id: l for l in g.lanes}
    node_map = {n.id: n for n in g.nodes}
    new_nodes: dict[int, Node] = {}
    new_lanes: list[Lane] = []
    new_edges: list[Edge] = []
    next_node = max(n.id for n in g.nodes) + 1
    cut_start: set[int] = set()
    cut_end: set[int] = set()

    def keep_node(nid):
        n = node_map[nid]
        if nid not in new_nodes:
            pos = n.pos.copy()
            pos[:2] += (gx - ox, gy - oy)
            poly = None if n.polygon is None else n.polygon + (gx - ox, gy - oy)
            new_nodes[nid] = Node(nid, pos, n.node_type, poly, n.port_embed, n.score)
        return nid

    for e in g.edges:
        pieces = {}
        for lid in e.lanes:
            l = lanes_by_id[lid]
            glob = l.points[:, :2] + (gx, gy)
            inter = LineString(glob).intersection(core)
            if inter.is_empty:
                continue
            if inter.geom_type != "LineString":
                raise ValueError(f"车道 {lid} 与核心区相交为多段，测试夹具须保证单次穿越")
            pts = np.asarray(inter.coords)
            if np.linalg.norm(pts[0] - glob[0]) > np.linalg.norm(pts[-1] - glob[0]):
                pts = pts[::-1]
            pieces[lid] = (pts, np.linalg.norm(pts[0] - glob[0]) > 1e-6, np.linalg.norm(pts[-1] - glob[-1]) > 1e-6)
        if not pieces:
            continue
        if len(pieces) != len(e.lanes):
            raise ValueError(f"edge {e.id} 的车道被核心区边界不一致地裁切")
        start_cut = any(v[1] for v in pieces.values())
        end_cut = any(v[2] for v in pieces.values())
        if start_cut:
            p = np.mean([v[0][0] for v in pieces.values()], axis=0)
            fn = next_node
            next_node += 1
            new_nodes[fn] = Node(fn, np.r_[p - (ox, oy), 0.0], "roi_entry_port",
                                 port_embed=_port_embed((e.id, int(round(p[0])), int(round(p[1]))), rng, embed_noise))
        else:
            fn = keep_node(e.from_node)
        if end_cut:
            p = np.mean([v[0][-1] for v in pieces.values()], axis=0)
            tn = next_node
            next_node += 1
            new_nodes[tn] = Node(tn, np.r_[p - (ox, oy), 0.0], "roi_exit_port",
                                 port_embed=_port_embed((e.id, int(round(p[0])), int(round(p[1]))), rng, embed_noise))
        else:
            tn = keep_node(e.to_node)
        for lid, (pts, sc, ec) in pieces.items():
            l = lanes_by_id[lid]
            loc = resample_polyline(pts - (ox, oy), K_LANE)
            new_lanes.append(Lane(lid, np.c_[loc, np.zeros(K_LANE)], l.width, l.role, l.permission,
                                  l.allow_change_left, l.allow_change_right, l.is_stop_line_end and not ec, l.score))
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
            pts[:, :2] += (gx - ox, gy - oy)
            conns.append(Connection(c.from_lane, c.to_lane, pts))
    cws = []
    edge_ids = {e.id for e in new_edges}
    for cw in g.crosswalks:
        if cw.node in new_nodes and all(eid in edge_ids for eid in cw.edges):
            ep = cw.endpoints.copy()
            ep[:, :2] += (gx - ox, gy - oy)
            cws.append(Crosswalk(cw.id, ep, cw.width, cw.node, list(cw.edges), cw.score))
    out = LaneGraph(list(new_nodes.values()), new_lanes, new_edges, conns, cws, origin=tile.origin,
                    meta={"tile": (tile.ix, tile.iy)})
    errs = out.validate()
    if errs:
        raise ValueError("裁切子图不合法：" + "; ".join(errs[:5]))
    return out


# ====================================================================== 拼接
@dataclass
class StitchReport:
    matched: list[tuple[int, int]]
    unmatched_exit: list[int]
    unmatched_entry: list[int]
    lane_count_mismatch: list[tuple[int, int]]


def _to_global(g: LaneGraph, k: int, base: tuple[float, float], id_off: dict):
    """把分块子图搬到统一全局偏移 base，并为节点/车道/edge 加分块前缀 id。"""
    dx, dy = g.origin[0] - base[0], g.origin[1] - base[1]
    nmap = {n.id: id_off["node"] + i for i, n in enumerate(g.nodes)}
    lmap = {l.id: id_off["lane"] + i for i, l in enumerate(g.lanes)}
    emap = {e.id: id_off["edge"] + i for i, e in enumerate(g.edges)}
    nodes = [Node(nmap[n.id], n.pos + (dx, dy, 0), n.node_type,
                  None if n.polygon is None else n.polygon + (dx, dy), n.port_embed, n.score) for n in g.nodes]
    lanes = [Lane(lmap[l.id], l.points + (dx, dy, 0), l.width, l.role, l.permission, l.allow_change_left,
                  l.allow_change_right, l.is_stop_line_end, l.score) for l in g.lanes]
    edges = [Edge(emap[e.id], [lmap[x] for x in e.lanes], nmap[e.from_node], nmap[e.to_node]) for e in g.edges]
    conns = [Connection(lmap[c.from_lane], lmap[c.to_lane], c.points + (dx, dy, 0)) for c in g.connections]
    cws = [Crosswalk(id_off["cw"] + i, c.endpoints + (dx, dy, 0), c.width, nmap[c.node], [emap[x] for x in c.edges],
                     c.score) for i, c in enumerate(g.crosswalks)]
    id_off["node"] += len(nodes)
    id_off["lane"] += len(lanes)
    id_off["edge"] += len(edges)
    id_off["cw"] += len(cws)
    return nodes, lanes, edges, conns, cws


def stitch_graphs(tiles: list[LaneGraph], base_origin: tuple[float, float] | None = None,
                  cfg: dict | None = None) -> tuple[LaneGraph, StitchReport]:
    cfg = (cfg or load_config("tiling"))["port_match"]
    base = base_origin or (min(t.origin[0] for t in tiles), min(t.origin[1] for t in tiles))
    off = {"node": 0, "lane": 0, "edge": 0, "cw": 0}
    nodes, lanes, edges, conns, cws = [], [], [], [], []
    for k, t in enumerate(tiles):
        n, l, e, c, w = _to_global(t, k, base, off)
        nodes += n
        lanes += l
        edges += e
        conns += c
        cws += w
    node = {n.id: n for n in nodes}
    lane = {l.id: l for l in lanes}
    exits = [n.id for n in nodes if n.node_type == "roi_exit_port"]
    entries = [n.id for n in nodes if n.node_type == "roi_entry_port"]
    report = StitchReport([], [], [], [])
    if exits and entries:
        C = np.full((len(exits), len(entries)), 1e6)
        for a, na in enumerate(exits):
            for b, nb in enumerate(entries):
                d = np.linalg.norm(node[na].pos[:2] - node[nb].pos[:2])
                if d > cfg["gate_m"]:
                    continue
                ea, eb = node[na].port_embed, node[nb].port_embed
                cos = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb))) if ea is not None and eb is not None else 0.0
                C[a, b] = cfg["w_dist"] * d / cfg["gate_m"] + cfg["w_embed"] * (1 - cos)
        ra, rb = linear_sum_assignment(C)
        for a, b in zip(ra, rb):
            if C[a, b] <= cfg["max_cost"]:
                report.matched.append((exits[a], entries[b]))
    matched_a = {a for a, _ in report.matched}
    matched_b = {b for _, b in report.matched}
    report.unmatched_exit = [a for a in exits if a not in matched_a]
    report.unmatched_entry = [b for b in entries if b not in matched_b]

    edge = {e.id: e for e in edges}
    for na, nb in report.matched:
        ea = next(e for e in edge.values() if e.to_node == na)
        eb = next(e for e in edge.values() if e.from_node == nb)
        if len(ea.lanes) != len(eb.lanes):
            report.lane_count_mismatch.append((na, nb))
            continue
        merged = []
        for la, lb in zip(ea.lanes, eb.lanes):
            A, B = lane[la], lane[lb]
            pts = np.concatenate([A.points[:, :2], B.points[:, :2]])
            keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
            new_pts = resample_polyline(pts[keep], K_LANE)
            lane[la] = Lane(la, np.c_[new_pts, np.zeros(K_LANE)], (A.width + B.width) / 2, A.role, A.permission,
                            A.allow_change_left and B.allow_change_left, A.allow_change_right and B.allow_change_right,
                            B.is_stop_line_end, min(A.score, B.score))
            merged.append(la)
            for c in conns:
                if c.from_lane == lb:
                    c.from_lane = la
            del lane[lb]
        edge[ea.id] = Edge(ea.id, merged, ea.from_node, eb.to_node)
        for cw in cws:
            cw.edges = [ea.id if x == eb.id else x for x in cw.edges]
        del edge[eb.id]
        del node[na]
        del node[nb]
    out = LaneGraph(list(node.values()), list(lane.values()), list(edge.values()), conns, cws, origin=base)
    errs = out.validate()
    if errs:
        raise RuntimeError("拼接结果不合法：" + "; ".join(errs[:5]))
    return out, report
