import numpy as np
import pytest

from src.graph.decoder import decode_graph, solve_path_cover, solve_succ
from src.graph.lane_graph import GraphTensors


def sig(g):
    return ({(tuple(e.lanes), e.from_node, e.to_node) for e in g.edges}, g.succ_pairs(),
            {(c.node, tuple(sorted(tuple(g.edge(e).lanes) for e in c.edges))) for c in g.crosswalks})


def fallback(t):
    return lambda i, j: t.conn_pts.get((i, j), np.linspace(t.lane_pts[i, -1], t.lane_pts[j, 0], 8))


def soften(t: GraphTensors, rng) -> GraphTensors:
    for k in ("E_adj", "E_succ", "E_from", "E_to", "E_cross", "E_cross_node"):
        a = getattr(t, k)
        setattr(t, k, np.where(a > 0.5, 0.65 + 0.35 * rng.random(a.shape), 0.35 * rng.random(a.shape)))
    return t


def test_exact_and_noisy_recovery(standard_graphs):
    rng = np.random.default_rng(0)
    for name, g in standard_graphs.items():
        if name == "zero_out_edge":
            continue
        t = g.to_tensors()
        r = decode_graph(t, fallback(t))
        assert sig(r.graph) == sig(g), name
        assert r.decoded.dead_end == 0 and r.decoded.unfed == 0 and r.decoded.same_approach_crossings == 0
        tn = soften(g.to_tensors(), rng)
        rn = decode_graph(tn, fallback(tn))
        assert sig(rn.graph) == sig(g), f"{name} (noisy)"


def test_nonzero_degree_enforced(standard_graphs):
    g = standard_graphs["zero_out_edge"]
    t = g.to_tensors()
    r = decode_graph(t, fallback(t))
    assert r.raw.dead_end > 0                  # 防遮蔽：原始阈值化输出如实记录死胡同
    assert r.decoded.dead_end == 0             # 约束解码强制内部节点出度 ≥ 1
    assert g.succ_pairs() < r.graph.succ_pairs()


def test_path_cover_breaks_cycles():
    w = np.full((4, 4), -1.0)
    w[0, 1] = w[1, 2] = w[2, 0] = 5.0        # 三元环
    w[2, 3] = 1.0
    sel = solve_path_cover(w)
    nxt = dict(sel)
    assert len(nxt) == len(sel)                # 出度 ≤ 1
    assert len({j for _, j in sel}) == len(sel)  # 入度 ≤ 1
    seen, u = set(), 0
    for start in range(4):
        u, seen = start, set()
        while u in nxt:
            assert u not in seen               # 无环
            seen.add(u)
            u = nxt[u]


def test_succ_noncrossing_and_degree():
    cand = [(0, 2), (0, 3), (1, 2), (1, 3)]
    w = np.array([1.0, 3.0, 3.0, 1.0])        # 最大权会选 0→3 与 1→2（交叉）
    x, feas = solve_succ(cand, w, need_out=[0, 1], need_in=[2, 3], crossings=[(1, 2)])
    chosen = {cand[k] for k in range(4) if x[k]}
    assert feas
    assert not ({(0, 3), (1, 2)} <= chosen)
    assert {i for i, _ in chosen} == {0, 1} and {j for _, j in chosen} == {2, 3}


def test_crosswalk_binding_only_to_core_or_midblock(standard_graphs):
    g = standard_graphs["midblock_two_stage"]
    t = g.to_tensors()
    r = decode_graph(t, fallback(t))
    for cw in r.graph.crosswalks:
        assert r.graph.node(cw.node).node_type in ("junction_core", "midblock_node")


def test_low_score_entities_dropped(standard_graphs):
    g = standard_graphs["cross4_basic"]
    t = g.to_tensors()
    t.cw_score[0] = 0.1
    r = decode_graph(t, fallback(t))
    assert len(r.graph.crosswalks) == len(g.crosswalks) - 1
