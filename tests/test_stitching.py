"""超大选区确定性分块端口拼接单测（M0 验收第 1 条明确要求）。"""
import numpy as np
import pytest

from src.common.geom import hausdorff
from src.data.standard_graphs import g_corridor_two_junctions
from src.graph.stitching import Tile, crop_graph, plan_tiles, stitch_graphs

ORIGIN = (500000.0, 2560000.0)


def _setup(noise=0.0):
    g = g_corridor_two_junctions()
    g.origin = ORIGIN
    X0, Y0 = ORIGIN
    roi = (X0 - 60, Y0 - 60, X0 + 360, Y0 + 60)
    tiles = plan_tiles(roi)
    subs = [crop_graph(g, t, embed_noise=noise, seed=k) for k, t in enumerate(tiles)]
    full = crop_graph(g, Tile(0, 0, roi, roi, tiles[0].origin, (0, 0)))
    return g, tiles, subs, full


def test_plan_tiles_limits():
    tiles = plan_tiles((0, 0, 1000, 300))
    for t in tiles:
        assert t.shape_px[0] <= 1024 + 32 and t.shape_px[1] <= 1024 + 32
        assert t.shape_px[0] % 32 == 0 and t.shape_px[1] % 32 == 0
        assert t.core[2] - t.core[0] <= 172.8 + 1e-9
        assert t.window[0] == pytest.approx(t.core[0] - 16)
    # 核心区互不重叠且完全覆盖选区
    area = sum((t.core[2] - t.core[0]) * (t.core[3] - t.core[1]) for t in tiles)
    assert area == pytest.approx(1000 * 300)
    assert len(plan_tiles((0, 0, 150, 150))) == 1


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_stitch_reproduces_single_window(noise):
    g, tiles, subs, full = _setup(noise)
    assert len(tiles) == 3
    assert sum(len(s.lanes) for s in subs) > len(full.lanes)        # 跨块边界的车道被切分
    st, rep = stitch_graphs(subs)
    assert not rep.lane_count_mismatch
    assert len(rep.matched) == 4                                      # 两条双向走廊各跨 2 条块边界
    assert (len(st.lanes), len(st.edges), len(st.connections), len(st.crosswalks)) == \
        (len(full.lanes), len(full.edges), len(full.connections), len(full.crosswalks))
    A = {l.id: l.points[:, :2] + st.origin for l in st.lanes}
    B = {l.id: l.points[:, :2] + full.origin for l in full.lanes}
    m = {}
    for i, pa in A.items():
        j, pb = min(B.items(), key=lambda kv: hausdorff(pa, kv[1], 0.5))
        assert hausdorff(pa, pb) < 0.05
        m[i] = j
    assert len(set(m.values())) == len(m)
    assert {(m[c.from_lane], m[c.to_lane]) for c in st.connections} == full.succ_pairs()
    types = sorted(n.node_type for n in st.nodes)
    assert types == sorted(n.node_type for n in full.nodes)


def test_unrelated_ports_not_matched():
    g, tiles, subs, full = _setup()
    for s in subs:                     # 出口端口嵌入取反（cos = −1）：仅位置相近不足以拼接（代价超限）
        for n in s.nodes:
            if n.node_type == "roi_exit_port":
                n.port_embed = -n.port_embed
    st, rep = stitch_graphs(subs)
    assert len(rep.matched) == 0


@pytest.mark.sumo
def test_stitched_graph_compiles(tmp_path):
    from src.sumo_io.pipeline import compile_graph
    _, _, subs, _ = _setup(0.3)
    st, _ = stitch_graphs(subs)
    r = compile_graph(st, tmp_path / "stitched")
    assert r.ok and r.zero_warning, (r.site.warnings, r.roundtrip.failures if r.roundtrip else None)
