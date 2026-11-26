import re

import pytest

from src.sumo_io.netconvert import run_netconvert
from src.sumo_io.pipeline import compile_graph
from src.sumo_io.roundtrip import classify_warnings, parse_net
from src.sumo_io.serializer import SumoSerializer
from src.sumo_io.simulate import run_sumo, write_demand

pytestmark = pytest.mark.sumo


def test_all_standard_graphs_roundtrip(standard_graphs, tmp_path):
    for name, g in standard_graphs.items():
        r = compile_graph(g, tmp_path / name)
        assert r.site.ok and r.eval_sim.ok, (name, r.site.errors)
        rt = r.roundtrip
        assert rt.ok, (name, rt.failures)
        assert not rt.conn_missing and not rt.conn_extra
        assert rt.lane_hausdorff_max <= 0.15 and rt.conn_hausdorff_max <= 0.15
        _, unexpected = classify_warnings(g, r.plain.ids, r.site.warnings + r.eval_sim.warnings)
        assert not unexpected, (name, unexpected)


def test_connection_from_blocks_auto_connections(standard_graphs, tmp_path):
    g = standard_graphs["zero_out_edge"]
    r = compile_graph(g, tmp_path / "a", with_eval_sim=False)
    con = r.paths["con"].read_text(encoding="utf-8")
    blocked = re.findall(r'<connection from="(e\d+)"/>', con)
    assert blocked
    stripped = tmp_path / "a" / "nob.con.xml"
    stripped.write_text(re.sub(r'\s*<connection from="e\d+"/>', "", con), encoding="utf-8")
    p = dict(r.paths)
    p["con"] = stripped
    rb = run_netconvert(p, tmp_path / "a" / "nob.net.xml")
    auto = [c for c in parse_net(rb.net_path).connections if c["from"] == blocked[0] and not c["to"].startswith(":")]
    kept = [c for c in parse_net(r.site.net_path).connections if c["from"] == blocked[0] and not c["to"].startswith(":")]
    assert auto and not kept          # 缺少禁补连语句时 netconvert 擅自补连；写出后 0 补连


def test_serializer_contracts(standard_graphs):
    g = standard_graphs["bus_lane"]
    x = SumoSerializer().serialize(g)
    assert 'allow="bus"' in x.edg and 'disallow="pedestrian" changeLeft' not in x.edg.split('allow="bus"')[0][-50:]
    assert re.search(r'<lane index="0" width="1.5" allow="pedestrian"', x.edg)          # 辅助人行道
    assert 'aux_crossing_anchor' in x.edg
    assert "netOffset" in x.nod and "+proj=utm +zone=49" in x.nod
    g2 = standard_graphs["solid_no_lane_change"]
    x2 = SumoSerializer().serialize(g2)
    assert 'changeLeft="authority"' in x2.edg and 'changeRight="authority"' in x2.edg
    assert 'changeLeft="none"' not in x2.edg and 'ignoring' not in x2.edg
    g3 = standard_graphs["slip_right_turn"]
    x3 = SumoSerializer().serialize(g3)
    assert x3.ids.aux_edges                                                           # 岛侧锚点边
    for n in g3.nodes:                                                                # 全节点显式 shape
        assert re.search(rf'<node id="n{n.id}" [^>]*shape="', x3.nod)


def test_eval_sim_tls_only_on_junction_core(standard_graphs, tmp_path):
    g = standard_graphs["midblock_two_stage"]          # 无 junction_core：不得建灯
    r = compile_graph(g, tmp_path / "m")
    assert "traffic_light" not in set(parse_net(r.eval_sim.net_path).junction_types.values())
    g = standard_graphs["slip_right_turn"]
    r = compile_graph(g, tmp_path / "s")
    jt = parse_net(r.eval_sim.net_path).junction_types
    core = [f"n{n.id}" for n in g.nodes if n.node_type == "junction_core"]
    others = [f"n{n.id}" for n in g.nodes if n.node_type in ("midblock_node", "split_merge")]
    assert all(jt[c] == "traffic_light" for c in core)
    assert all(jt[o] != "traffic_light" for o in others)


@pytest.mark.parametrize("name", ["cross4_basic", "two_stage_crossing", "slip_right_turn", "midblock_uturn"])
def test_probe_simulation_collision_free(standard_graphs, tmp_path, name):
    g = standard_graphs[name]
    r = compile_graph(g, tmp_path / name)
    for mode, net in (("sparse_single", r.site.net_path), ("probe", r.eval_sim.net_path)):
        rou = write_demand(g, r.plain.ids, tmp_path / name / f"{mode}.rou.xml", mode, 42)
        s = run_sumo(net, rou, tmp_path / name, 42, mode)
        assert s.loaded > 0 and s.arrived == s.loaded and s.collisions == 0 and s.teleports == 0, (name, mode, s)
