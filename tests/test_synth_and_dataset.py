"""M1 合成引擎、紧凑分片库与在线数据管线单元测试（设备无关；docs/DESIGN.md §5 Layer 1）。"""
from __future__ import annotations

import json

import numpy as np
import pytest
from shapely.geometry import box

from src.common.config import load_config
from src.data.dataset import SynthWindowDataset, collate
from src.data.synth.codec import decode_sample, encode_sample
from src.data.synth.kinematic import KinematicFail, Obstacles, TrajectoryGenerator, min_radius
from src.data.synth.library import SynthLibrary, build_split, generate_one, sha256_file
from src.data.synth.render import render_scene
from src.data.synth.scene import SceneGenerator
from src.data.targets import DENSE_CHANNELS, crop_graph_to_roi, dense_targets, graph_targets, gt_region
from src.graph.lane_graph import K_CONN, K_LANE, NODE_TYPES, POLY_K


def test_kinematic_trajectory_constraints():
    cfg = load_config("synth")["kinematic"]
    gen = TrajectoryGenerator(cfg)
    # 1. 直线续接
    pts, dense = gen.generate([0.0, 0.0], [1.0, 0.0], [12.0, 0.0], [1.0, 0.0], K_CONN)
    assert pts.shape == (K_CONN, 2)
    assert np.allclose(pts[0], [0.0, 0.0]) and np.allclose(pts[-1], [12.0, 0.0])
    assert np.allclose(pts[:, 1], 0.0, atol=1e-6)
    # 2. 90° 左转 + 安全岛避障 + 并列转向间隔
    # 安全岛位于外侧车行道之外（y ≥ 6.5，距外侧出口车道中心 y=3.2 净距 3.3 m > 车体半宽+净距 1.2 m）
    obs = Obstacles([np.array([[-12.0, 6.5], [-5.0, 6.5], [-5.0, 13.5], [-12.0, 13.5]])])
    p_in, d_in = gen.generate([0.0, -14.0], [0.0, 1.0], [-14.0, 0.0], [-1.0, 0.0], K_CONN,
                              obstacles=obs, lane_width=3.2)
    assert p_in.shape == (K_CONN, 2)
    assert np.allclose(p_in[0], [0.0, -14.0]) and np.allclose(p_in[-1], [-14.0, 0.0])
    assert min_radius(d_in) >= cfg["r_min_turn"] * 0.95
    need = cfg["vehicle_half_width_m"] + cfg["obstacle_clearance_m"]
    assert obs.signed_distance(d_in).min() >= need - 0.1
    p_out, d_out = gen.generate([3.2, -14.0], [0.0, 1.0], [-14.0, 3.2], [-1.0, 0.0], K_CONN,
                                obstacles=obs, siblings=[d_in], lane_width=3.2)
    assert p_out.shape == (K_CONN, 2)
    assert min_radius(d_out) >= cfg["r_min_turn"] * 0.95
    assert obs.signed_distance(d_out).min() >= need - 0.1
    from scipy.spatial import cKDTree
    mid = slice(len(d_out) // 4, 3 * len(d_out) // 4)
    assert cKDTree(d_in).query(d_out[mid])[0].min() >= cfg["parallel_gap_frac"] * 3.2 - 0.15
    # 不可行障碍物（直接封堵端点前通路）须触发 KinematicFail
    bad_obs = Obstacles([np.array([[-13.0, -2.0], [-11.0, -2.0], [-11.0, 2.0], [-13.0, 2.0]])])
    with pytest.raises(KinematicFail):
        gen.generate([0.0, -14.0], [0.0, 1.0], [-14.0, 0.0], [-1.0, 0.0], K_CONN, obstacles=bad_obs)
    # 3. 掉头最小转弯半径
    pu, du = gen.generate([0.0, 0.0], [1.0, 0.0], [0.0, 10.0], [-1.0, 0.0], K_CONN, uturn=True)
    assert pu.shape == (K_CONN, 2)
    assert min_radius(du) >= cfg["r_min_uturn"] * 0.95
    # 4. 首尾重合拒绝
    with pytest.raises(KinematicFail):
        gen.generate([1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], K_CONN)


@pytest.mark.parametrize("kind", ["junction", "corridor", "segment"])
def test_scene_generator_and_renderer_all_kinds(kind):
    cfg = dict(load_config("synth"), scene_mix={kind: 1.0})
    gen = SceneGenerator(cfg)
    sc = gen.generate(seed=100 + ["junction", "corridor", "segment"].index(kind))
    g = sc.graph
    assert sc.kind == kind
    assert g.validate() == []
    assert all(n.node_type in NODE_TYPES for n in g.nodes)
    assert all(l.points.shape == (K_LANE, 3) for l in g.lanes)
    assert all(c.points.shape == (K_CONN, 3) for c in g.connections)
    for n in g.nodes:
        if n.node_type == "junction_core":
            assert n.polygon is not None and n.polygon.shape == (POLY_K, 2)
    fs, meta = render_scene(sc, gen, np.random.default_rng(7))
    codes = {r[0] for r in fs.road}
    assert "road_surface" in codes
    assert isinstance(meta["arrows"], list) and isinstance(meta["neg_regions"], list)
    # 编解码往返保真
    blob = encode_sample(g, fs, meta)
    rec = decode_sample(blob)
    assert rec.graph.validate() == []
    assert len(rec.graph.lanes) == len(g.lanes)
    assert len(rec.graph.connections) == len(g.connections)
    assert len(rec.features.road) == len(fs.road)


@pytest.mark.sumo
def test_generate_one_sumo_roundtrip(tmp_path):
    blob, info = generate_one(seed=42, work=tmp_path, compile_check=True, force_kind="junction")
    assert blob is not None, info
    assert info["hausdorff"] <= 0.15
    rec = decode_sample(blob)
    assert rec.graph.validate() == []


def test_build_split_dataset_targets_and_pair_cap(tmp_path):
    stats = build_split("train", count=4, root=tmp_path, workers=2, compile_check=False,
                        force_kind="junction", seed_base=500, progress=lambda *_: None)
    assert stats["count"] == 4
    man = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["splits"]["train"]["count"] == 4
    for fname, digest in man["splits"]["train"]["shards"].items():
        assert sha256_file(tmp_path / "train" / fname) == digest
    lib = SynthLibrary("train", root=tmp_path)
    assert len(lib) == 4
    rec = lib[0]
    # ROI 裁切与真值张量化
    core = next(n for n in rec.graph.nodes if n.node_type == "junction_core")
    cx, cy = core.pos[:2]
    roi = box(cx - 45.0, cy - 45.0, cx + 45.0, cy + 45.0)
    wbox = box(cx - 60.0, cy - 60.0, cx + 60.0, cy + 60.0)
    reg = gt_region(rec.graph, roi, wbox)
    sub = crop_graph_to_roi(rec.graph, reg, shift=(cx - 60.0, cy - 60.0))
    assert sub.validate() == []
    ports = [n for n in sub.nodes if n.node_type in ("roi_entry_port", "roi_exit_port")]
    assert len(ports) >= 2 and all(getattr(n, "port_key", -1) >= 0 for n in ports)
    gt = graph_targets(sub)
    assert int(gt["lane_valid"].sum()) == len(sub.lanes)
    assert int(gt["conn_valid"].sum()) == len(sub.connections)
    dt = dense_targets(sub, 512, 512, 0.20)
    assert dt.shape == (len(DENSE_CHANNELS), 256, 256)
    assert dt[0].max() <= 1.0 and dt[0].min() >= 0.0
    assert dt[2].max() <= 1.0 and dt[2].min() >= -1.0
    # 在线 Dataset + collate max_pairs 上限（防多对同时触发导致显存膨胀）
    ds = SynthWindowDataset("train", root=tmp_path, train=True, seed=11, randomize=True, pair=False, fixed_px=512)
    s0, s1 = ds[0][0], ds[1][0]
    assert s0["bev"].shape == (36, 512, 512)
    assert s0["dense"].shape == (len(DENSE_CHANNELS), 256, 256)
    # 人工构造两组 pair，验证 max_pairs=1 严格截断至 2+1=3 窗且仅保留第一组 pair_id
    unbounded = collate([[s0, s0], [s1, s1]], max_pairs=None)
    bounded = collate([[s0, s0], [s1, s1]], max_pairs=1)
    assert unbounded["bev"].shape[0] == 4 and (unbounded["pair_id"] >= 0).sum().item() == 4
    assert bounded["bev"].shape[0] == 3
    assert bounded["pair_id"].tolist() == [0, 0, -1]
