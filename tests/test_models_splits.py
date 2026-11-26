import json
import math

import numpy as np
import pytest

from src.data.splits import SplitManifest, assign_blocks


@pytest.mark.mps
def test_mps_bilinear_matches_grid_sample():
    import torch
    from src.models.sampling import bilinear_sample, grid_sample_reference
    torch.manual_seed(0)
    f = torch.randn(2, 16, 40, 52)
    g = torch.rand(2, 300, 2) * 2.4 - 1.2
    fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
    ref = grid_sample_reference(fr, gr)
    w = torch.randn_like(ref)
    (ref * w).sum().backward()
    fm, gm = f.to("mps").requires_grad_(), g.to("mps").requires_grad_()
    out = bilinear_sample(fm, gm)
    (out * w.to("mps")).sum().backward()          # PYTORCH_ENABLE_MPS_FALLBACK=0：任何回退即报错
    assert (out.cpu() - ref).abs().max().item() < 1e-5
    assert (fm.grad.cpu() - fr.grad).abs().max().item() < 1e-5
    assert ((gm.grad.cpu() - gr.grad).abs().max() / gr.grad.abs().max()).item() < 1e-5


@pytest.mark.mps
def test_deform_attn_and_encoder_on_mps():
    import torch
    from src.models.bev_encoder import BevEncoder
    from src.models.sampling import MSDeformAttn
    enc = BevEncoder().to("mps")
    x = torch.randn(1, 36, 256, 320, device="mps")
    f = enc(x)
    assert f["F2"].shape == (1, 128, 128, 160) and f["F4"].shape == (1, 256, 64, 80)
    assert f["F8"].shape == (1, 256, 32, 40) and f["F16"].shape == (1, 256, 16, 20)
    attn = MSDeformAttn().to("mps")
    q = torch.randn(1, 64, 256, device="mps", requires_grad=True)
    out = attn(q.expand(1, -1, -1), torch.rand(1, 64, 2, device="mps"), [f["F4"], f["F8"], f["F16"]])
    (out.pow(2).mean() + f["F2"].pow(2).mean()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def _toy_manifest(tmp_path, splits):
    blocks = [{"ix": i, "iy": 0, "split": s, "district": "d", "n_signal": 1} for i, s in enumerate(splits)]
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"version": "t", "block_size_m": 5000, "origin": [0, 0], "buffer_margin_m": 368.4,
                             "blocks": blocks}), encoding="utf-8")
    return SplitManifest.load(p)


def test_split_buffer_semantics(tmp_path):
    m = _toy_manifest(tmp_path, ["train", "test", "test"])
    assert m.assign(2500, 2500) == "train"
    assert m.assign(4800, 2500) == "buffer"          # 距 test 区块 200 m
    assert m.assign(5200, 2500) == "buffer"
    assert m.assign(9990, 2500) == "test"            # 两侧同为 test：无隔离带
    assert m.assign(-10, 10) == "unassigned"
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [15000, 5000], size=(3000, 2))
    lab = [m.assign(*p) for p in pts]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if lab[i] in ("train", "test") and lab[j] in ("train", "test") and lab[i] != lab[j]:
                assert np.linalg.norm(pts[i] - pts[j]) >= 500 + 2 * 118.4


def test_assign_blocks_deterministic():
    cfg = {"seed": 1, "fractions": {"train": 0.7, "val": 0.15, "test": 0.15}}
    bd = {(i, 0): "a" for i in range(20)} | {(i, 1): "b" for i in range(10)}
    w = {(i, 0): i for i in range(20)}
    a1 = assign_blocks(bd, w, {(0, 1)}, cfg)
    a2 = assign_blocks(bd, w, {(0, 1)}, cfg)
    assert a1 == a2 and a1[(0, 1)] == "dev"
    assert {"train", "val", "test"} <= set(a1.values())


def test_repo_manifest_valid():
    m = SplitManifest.load()
    splits = {b["split"] for b in m.blocks.values()}
    assert {"train", "val", "test", "dev"} <= splits
    assert m.margin >= 250 + 118.4
    d = json.loads((__import__("src.common.config", fromlist=["REPO_ROOT"]).REPO_ROOT
                    / "configs/splits/split_v1.json").read_text(encoding="utf-8"))
    assert d["stats"]["cross_split_pairs_within_gap"] == 0
    devs = [b for b in d["blocks"] if b["split"] == "dev"]
    assert devs and all(b["split"] not in ("train", "test") for b in devs)
