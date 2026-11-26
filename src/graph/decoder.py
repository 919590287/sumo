"""结构化约束图解码器（docs/DESIGN.md §4.2 阶段 E）。

把神经网络输出的连续概率矩阵经**纯图论组合优化**解码为 100% 拓扑合法的离散图 G：
1. ``E_adj`` 编组：无环最大权有向路径覆盖（出度 ≤ 1、入度 ≤ 1、无环；
   scipy HiGHS MILP + 懒惰破环约束，N ≤ 128 毫秒级精确求解）；
2. 联合 ``(from, to)`` 节点归属：同一 edge 内全部车道共享唯一节点对（u ≠ v）；
3. ``E_succ`` 合法化：节点衔接 + 内部节点非零度 + 同 (edge_in, edge_out) 单调保序无交叉 的
   最大权连接子图（MILP 精确求解）；
4. ``E_cross`` / ``E_cross_node``：绑定 junction_core / midblock_node 并按 edge 聚合解码跨越列表。

防遮蔽纪律：``DecodeResult.raw`` 同步给出“仅阈值化”的原始拓扑统计，供评测双轨记录。
本模块不包含任何几何阈值或场景特判；全部门限为 configs/decoder.yaml 中的图论解码参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from src.common.config import load_config
from src.graph.lane_graph import (LANE_ROLES, NODE_TYPES, PERMISSIONS, Connection, Crosswalk, Edge,
                                  GraphTensors, Lane, LaneGraph, Node)

CROSS_NODE_TYPES = ("junction_core", "midblock_node")


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


@dataclass
class TopologyStats:
    dead_end: int = 0
    unfed: int = 0
    same_approach_crossings: int = 0
    n_succ: int = 0


@dataclass
class DecodeResult:
    graph: LaneGraph
    raw: TopologyStats
    decoded: TopologyStats
    info: dict = field(default_factory=dict)


# ====================================================================== MILP 子问题
def solve_path_cover(w: np.ndarray, max_cuts: int = 200) -> list[tuple[int, int]]:
    """无环最大权有向路径覆盖：仅正权边参与（负权边不会改善目标）。

    w[i, j] 为 j 作为 i 左邻的边权；返回选中的 (i, j)。
    """
    n = w.shape[0]
    cand = [(i, j) for i in range(n) for j in range(n) if i != j and w[i, j] > 0]
    if not cand:
        return []
    m = len(cand)
    c = -np.array([w[i, j] for i, j in cand])
    rows, cols = [], []
    for k, (i, j) in enumerate(cand):
        rows += [i, n + j]
        cols += [k, k]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, m)).tocsr()
    cons = [LinearConstraint(A, -np.inf, np.ones(2 * n))]
    for _ in range(max_cuts):
        res = milp(c, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1))
        if res.x is None:
            raise RuntimeError(f"E_adj 路径覆盖求解失败：{res.message}")
        sel = [cand[k] for k in range(m) if res.x[k] > 0.5]
        cycles = _find_cycles(sel)
        if not cycles:
            return sel
        for cyc in cycles:
            row = np.zeros(m)
            for k, e in enumerate(cand):
                if e in cyc:
                    row[k] = 1
            cons.append(LinearConstraint(row[None], -np.inf, len(cyc) - 1))
    raise RuntimeError("E_adj 破环迭代超过上限")


def _find_cycles(edges: list[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    nxt = dict(edges)
    seen, cycles = set(), []
    for s in nxt:
        if s in seen:
            continue
        path, u = [], s
        while u in nxt and u not in seen:
            seen.add(u)
            path.append(u)
            u = nxt[u]
        if u in path:
            k = path.index(u)
            cyc = path[k:]
            cycles.append({(a, nxt[a]) for a in cyc})
    return cycles


def solve_succ(cand: list[tuple[int, int]], weight: np.ndarray, need_out: list[int], need_in: list[int],
               crossings: list[tuple[int, int]]) -> tuple[np.ndarray, bool]:
    """E_succ 合法化 MILP。返回 (选择向量, 是否在硬约束下可行)。"""
    m = len(cand)
    if m == 0:
        return np.zeros(0), True
    idx_out: dict[int, list[int]] = {}
    idx_in: dict[int, list[int]] = {}
    for k, (i, j) in enumerate(cand):
        idx_out.setdefault(i, []).append(k)
        idx_in.setdefault(j, []).append(k)
    rows, cols, lb = [], [], []
    r = 0
    for i in need_out:
        for k in idx_out.get(i, []):
            rows.append(r)
            cols.append(k)
        lb.append(1.0)
        r += 1
    for j in need_in:
        for k in idx_in.get(j, []):
            rows.append(r)
            cols.append(k)
        lb.append(1.0)
        r += 1
    n_deg = r
    for a, b in crossings:
        rows += [r, r]
        cols += [a, b]
        lb.append(-np.inf)
        r += 1
    ub = [np.inf] * n_deg + [1.0] * (r - n_deg)

    def run(soft: bool):
        n_s = n_deg if soft else 0
        c = np.concatenate([-weight, np.full(n_s, 1e3)])
        if r:
            data = np.ones(len(rows))
            A = coo_matrix((data, (rows, cols)), shape=(r, m + n_s)).tolil()
            for s in range(n_s):
                A[s, m + s] = 1.0
            cons = [LinearConstraint(A.tocsr(), np.array(lb), np.array(ub))]
        else:
            cons = []
        return milp(c, constraints=cons, integrality=np.ones(m + n_s),
                    bounds=Bounds(0, np.r_[np.ones(m), np.full(n_s, np.inf)]))

    res = run(False)
    if res.x is not None:
        return res.x[:m] > 0.5, True
    res = run(True)
    if res.x is None:
        raise RuntimeError(f"E_succ 合法化求解失败：{res.message}")
    return res.x[:m] > 0.5, False


# ====================================================================== 拓扑统计
def topology_stats(succ: set[tuple[int, int]], lane_from: dict[int, int], lane_to: dict[int, int],
                   lane_edge: dict[int, int], lane_idx: dict[int, int], node_type: dict[int, str]) -> TopologyStats:
    s = TopologyStats(n_succ=len(succ))
    outs = {i for i, _ in succ}
    ins = {j for _, j in succ}
    for l in lane_from:
        if node_type.get(lane_to[l]) != "roi_exit_port" and l not in outs:
            s.dead_end += 1
        if node_type.get(lane_from[l]) != "roi_entry_port" and l not in ins:
            s.unfed += 1
    lst = sorted(succ)
    for a in range(len(lst)):
        i1, j1 = lst[a]
        for b in range(a + 1, len(lst)):
            i2, j2 = lst[b]
            if lane_edge[i1] == lane_edge[i2] and lane_edge[j1] == lane_edge[j2]:
                if (lane_idx[i1] - lane_idx[i2]) * (lane_idx[j1] - lane_idx[j2]) < 0:
                    s.same_approach_crossings += 1
    return s


# ====================================================================== 主解码
def decode_graph(t: GraphTensors, conn_pts: dict | Callable | None = None, cfg: dict | None = None) -> DecodeResult:
    cfg = cfg or load_config("decoder")
    eps = cfg["eps"]
    nodes_act = [p for p in range(len(t.node_score)) if t.node_score[p] >= cfg["tau_node"]]
    lanes_act = [i for i in range(len(t.lane_score)) if t.lane_score[i] >= cfg["tau_lane"]]
    ntype = {p: NODE_TYPES[int(np.argmax(t.node_type[p]))] for p in nodes_act}
    info: dict = {"n_nodes_active": len(nodes_act), "n_lanes_active": len(lanes_act)}

    # ---------------------------------------------------------------- 1. E_adj 路径覆盖
    L = lanes_act
    W = _logit(t.E_adj[np.ix_(L, L)], eps) - cfg["tau_adj"]
    np.fill_diagonal(W, -np.inf)
    sel = solve_path_cover(np.where(np.isfinite(W), W, -1.0), cfg["max_cycle_cuts"])
    left_of = {L[i]: L[j] for i, j in sel}
    has_right = {L[j] for _, j in sel}
    chains = []
    for s in L:
        if s in has_right:
            continue
        ch = [s]
        while ch[-1] in left_of:
            ch.append(left_of[ch[-1]])
        chains.append(ch)

    # ---------------------------------------------------------------- 2. 联合 (from, to)
    if len(nodes_act) < 2:
        raise RuntimeError("有效节点少于 2 个，无法解码")
    lf = np.log(np.clip(t.E_from[:, nodes_act], eps, 1.0))
    lt = np.log(np.clip(t.E_to[:, nodes_act], eps, 1.0))
    edges: list[Edge] = []
    for k, ch in enumerate(chains):
        A = lf[ch].sum(0)
        B = lt[ch].sum(0)
        S = A[:, None] + B[None, :]
        np.fill_diagonal(S, -np.inf)
        u, v = np.unravel_index(int(np.argmax(S)), S.shape)
        edges.append(Edge(k, list(ch), nodes_act[u], nodes_act[v]))
    lane_edge = {l: e.id for e in edges for l in e.lanes}
    lane_idx = {l: i for e in edges for i, l in enumerate(e.lanes)}
    lane_from = {l: e.from_node for e in edges for l in e.lanes}
    lane_to = {l: e.to_node for e in edges for l in e.lanes}

    # ---------------------------------------------------------------- 原始阈值化拓扑（双轨）
    raw_succ = {(i, j) for i in L for j in L if i != j and t.E_succ[i, j] >= 0.5}
    raw_ok = {(i, j) for i, j in raw_succ if lane_to[i] == lane_from[j]}
    raw = topology_stats(raw_ok, lane_from, lane_to, lane_edge, lane_idx, ntype)
    info["raw_succ_node_inconsistent"] = len(raw_succ) - len(raw_ok)

    # ---------------------------------------------------------------- 3. E_succ 合法化
    cand = [(i, j) for i in L for j in L
            if lane_edge[i] != lane_edge[j] and lane_to[i] == lane_from[j]]
    wt = np.array([_logit(np.array(t.E_succ[i, j]), eps) - cfg["tau_succ"] for i, j in cand], dtype=float)
    has_out_cand = {i for i, _ in cand}
    has_in_cand = {j for _, j in cand}
    need_out = [i for i in L if ntype[lane_to[i]] != "roi_exit_port" and i in has_out_cand]
    need_in = [j for j in L if ntype[lane_from[j]] != "roi_entry_port" and j in has_in_cand]
    groups: dict[tuple[int, int], list[int]] = {}
    for k, (i, j) in enumerate(cand):
        groups.setdefault((lane_edge[i], lane_edge[j]), []).append(k)
    crossings = []
    for ks in groups.values():
        for a in range(len(ks)):
            i1, j1 = cand[ks[a]]
            for b in range(a + 1, len(ks)):
                i2, j2 = cand[ks[b]]
                if (lane_idx[i1] - lane_idx[i2]) * (lane_idx[j1] - lane_idx[j2]) < 0:
                    crossings.append((ks[a], ks[b]))
    x, feasible = solve_succ(cand, wt, need_out, need_in, crossings)
    succ = [cand[k] for k in range(len(cand)) if x[k]]
    info["succ_hard_feasible"] = feasible
    decoded = topology_stats(set(succ), lane_from, lane_to, lane_edge, lane_idx, ntype)

    # ---------------------------------------------------------------- 4. 人行横道
    cws: list[Crosswalk] = []
    cw_nodes = [p for p in nodes_act if ntype[p] in CROSS_NODE_TYPES]
    for m in range(len(t.cw_score)):
        if t.cw_score[m] < cfg["tau_cw"] or not cw_nodes:
            continue
        p = cw_nodes[int(np.argmax(t.E_cross_node[m, cw_nodes]))]
        inc = [e for e in edges if p in (e.from_node, e.to_node)]
        if not inc:
            continue
        sc = np.array([max(t.E_cross[m, l] for l in e.lanes) for e in inc])
        chosen = [e.id for e, s in zip(inc, sc) if s >= cfg["tau_cross"]] or [inc[int(np.argmax(sc))].id]
        cws.append(Crosswalk(len(cws), t.cw_endpoints[m], float(t.cw_width[m]), p, chosen,
                             float(t.cw_score[m])))

    # ---------------------------------------------------------------- 5. 实体装配
    lanes = []
    for l in L:
        lanes.append(Lane(l, t.lane_pts[l], float(t.lane_width[l]),
                          role=LANE_ROLES[int(np.argmax(t.lane_role[l]))],
                          permission=PERMISSIONS[int(np.argmax(t.lane_perm[l]))],
                          allow_change_left=bool(t.lane_lc[l, 0] >= 0.5),
                          allow_change_right=bool(t.lane_lc[l, 1] >= 0.5),
                          is_stop_line_end=bool(t.lane_stop[l] >= 0.5), score=float(t.lane_score[l])))
    lane_map = {l.id: l for l in lanes}
    conns = []
    for i, j in succ:
        if conn_pts is None:
            raise ValueError("缺少成对连接轨迹 K_conn（conn_pts）")
        pts = conn_pts(i, j) if callable(conn_pts) else conn_pts[(i, j)]
        pts = np.array(pts, dtype=np.float64)
        if pts.shape[-1] == 2:
            pts = np.c_[pts, np.zeros(len(pts))]
        pts[0] = lane_map[i].points[-1]
        pts[-1] = lane_map[j].points[0]
        conns.append(Connection(i, j, pts))
    used = {e.from_node for e in edges} | {e.to_node for e in edges} | {c.node for c in cws}
    nodes = []
    for p in nodes_act:
        if p not in used:
            continue
        poly = t.node_poly[p] if ntype[p] == "junction_core" else None
        pe = t.port_embed[p] if ntype[p] in ("roi_entry_port", "roi_exit_port") else None
        nodes.append(Node(p, t.node_pos[p], ntype[p], poly, pe, float(t.node_score[p])))
    g = LaneGraph(nodes, lanes, edges, conns, cws, origin=tuple(t.origin))
    errs = g.validate()
    if errs:
        raise RuntimeError("约束解码输出不合法：" + "; ".join(errs[:5]))
    return DecodeResult(g, raw, decoded, info)
