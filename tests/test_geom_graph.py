import numpy as np
import pytest

from src.common.geom import hausdorff, offset_polyline, polyline_length, resample_polyline
from src.graph.lane_graph import K_CONN, K_LANE, NODE_TYPES, POLY_K, LaneGraph


def test_resample_equal_arclength():
    t = np.linspace(0, np.pi, 200)
    arc = np.c_[10 * np.cos(t), 10 * np.sin(t)]
    r = resample_polyline(arc, K_LANE)
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    assert r.shape == (K_LANE, 2)
    assert seg.std() / seg.mean() < 1e-3
    assert np.allclose(r[0], arc[0]) and np.allclose(r[-1], arc[-1])


def test_offset_and_hausdorff():
    line = np.c_[np.linspace(0, 50, 20), np.zeros(20), np.zeros(20)]
    left = offset_polyline(line, 1.75)
    assert np.allclose(left[:, 1], 1.75)
    assert hausdorff(line, left) == pytest.approx(1.75, abs=1e-9)
    assert polyline_length(line) == pytest.approx(50)


def test_standard_graph_library(standard_graphs):
    assert len(standard_graphs) >= 20
    types = set()
    for name, g in standard_graphs.items():
        assert g.validate() == [], name
        types |= {n.node_type for n in g.nodes}
        for l in g.lanes:
            assert l.points.shape == (K_LANE, 3)
        for c in g.connections:
            assert c.points.shape == (K_CONN, 3)
        for n in g.nodes:
            if n.node_type == "junction_core":
                assert n.polygon.shape == (POLY_K, 2)
    # 覆盖 M0 要求的场景要素
    assert {"junction_core", "split_merge", "midblock_node", "roi_entry_port", "roi_exit_port",
            "road_continuation"} <= types == set(NODE_TYPES) | types
    perms = {l.permission for g in standard_graphs.values() for l in g.lanes}
    assert {"general", "bus_only", "variable", "reversible"} <= perms
    assert any(not l.allow_change_left for g in standard_graphs.values() for l in g.lanes)
    assert any(c.node for g in standard_graphs.values() for c in g.crosswalks)


def test_json_and_tensor_roundtrip(standard_graphs):
    g = standard_graphs["slip_right_turn"]
    g2 = LaneGraph.from_dict(g.to_dict())
    assert g2.validate() == []
    assert g2.succ_pairs() == g.succ_pairs()
    t = g.to_tensors()
    assert t.E_succ.sum() == len(g.connections)
    assert t.E_adj.sum() == sum(len(e.lanes) - 1 for e in g.edges)
    assert np.all(t.E_from.sum(1)[: len(g.lanes)] == 1)


def test_validate_catches_errors(standard_graphs):
    g = LaneGraph.from_dict(standard_graphs["cross4_basic"].to_dict())
    g.connections[0].points[0, 0] += 1.0
    assert any("锁定" in e for e in g.validate())
    g = LaneGraph.from_dict(standard_graphs["cross4_basic"].to_dict())
    g.edges[0].to_node = g.edges[0].from_node
    assert any("自环" in e for e in g.validate())
