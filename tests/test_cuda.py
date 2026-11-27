"""CUDA 专属单元测试（docs/DESIGN.md §6.1-5、§6.2-3 第 3 步）。

本机 Mac（无 CUDA）自动跳过；Linux 服务器训练包自检（B2S_DEVICE=cuda）强制执行：
1. CUDA 上 gather 双线性采样与 F.grid_sample 前向及梯度等价性（误差 < 1e-5）；
2. CUDA 上 AMP（bf16 / fp16）相对 fp32 梯度余弦相似度与 decide_amp 校准逻辑；
3. CUDA 模型权重以 CPU state_dict 保存、跨设备（CUDA ↔ CPU）前向数值一致性。
"""
from __future__ import annotations

import copy

import pytest
import torch

from scripts.calibrate import decide_amp, grads
from src.common.config import load_config
from src.common.device import amp_dtype, batch_to
from src.data.targets import DENSE_CHANNELS
from src.graph.lane_graph import CONN_MAX, K_CONN, K_LANE, M_MAX, N_MAX, P_MAX, POLY_K
from src.models.b2s_net import build_model
from src.models.losses import SimCriterion
from src.models.sampling import bilinear_sample, grid_sample_reference
from src.train.trainer import load_checkpoint_model, seed_everything, train_step


def _mini_model_cfg() -> dict:
    cfg = copy.deepcopy(load_config("model"))
    cfg.update({"d_model": 64, "d2": 32, "n_heads": 4, "ffn": 128})
    cfg["vector_encoder"].update({"point_hidden": 32, "layers": 1, "rpe_hidden": 16})
    cfg["fusion"].update({"layers": 1})
    cfg["decoder"].update({"layers": 2})
    cfg["topology"].update({"layers": 1, "pair_hidden": 32})
    cfg["conn_head"].update({"layers": 1})
    cfg["dense_head"].update({"hidden": 32})
    return cfg


def _dummy_batch(B: int = 1, H: int = 128, W: int = 128) -> dict:
    torch.manual_seed(0)
    b = {
        "bev": torch.rand(B, 36, H, W, dtype=torch.float16),
        "hw": torch.tensor([[H, W]] * B, dtype=torch.long),
        "tok_cls": torch.zeros(B, 32, dtype=torch.long),
        "tok_pts": torch.rand(B, 32, 8, 2) * 20.0,
        "tok_tan": torch.ones(B, 32, 8, 2) * 0.707,
        "tok_size": torch.ones(B, 32, 2),
        "tok_heading": torch.zeros(B, 32, 2),
        "tok_turns": torch.zeros(B, 32, 6),
        "tok_mask": torch.ones(B, 32, dtype=torch.bool),
        "dense": torch.zeros(B, len(DENSE_CHANNELS), H // 2, W // 2, dtype=torch.float16),
        "node_pos": torch.zeros(B, P_MAX, 2), "node_type": torch.full((B, P_MAX), -1, dtype=torch.long),
        "node_poly": torch.zeros(B, P_MAX, POLY_K, 2), "node_valid": torch.zeros(B, P_MAX, dtype=torch.bool),
        "port_key": torch.full((B, P_MAX), -1, dtype=torch.long),
        "lane_pts": torch.zeros(B, N_MAX, K_LANE, 2), "lane_w": torch.zeros(B, N_MAX),
        "lane_role": torch.full((B, N_MAX), -1, dtype=torch.long),
        "lane_perm": torch.full((B, N_MAX), -1, dtype=torch.long),
        "lane_lc": torch.zeros(B, N_MAX, 2), "lane_stop": torch.zeros(B, N_MAX),
        "lane_valid": torch.zeros(B, N_MAX, dtype=torch.bool),
        "lane_from": torch.full((B, N_MAX), -1, dtype=torch.long),
        "lane_to": torch.full((B, N_MAX), -1, dtype=torch.long),
        "E_adj": torch.zeros(B, N_MAX, N_MAX, dtype=torch.uint8),
        "E_succ": torch.zeros(B, N_MAX, N_MAX, dtype=torch.uint8),
        "cw_end": torch.zeros(B, M_MAX, 2, 2), "cw_w": torch.zeros(B, M_MAX),
        "cw_valid": torch.zeros(B, M_MAX, dtype=torch.bool),
        "E_cross": torch.zeros(B, M_MAX, N_MAX, dtype=torch.uint8),
        "cw_node": torch.full((B, M_MAX), -1, dtype=torch.long),
        "conn_ij": torch.zeros(B, CONN_MAX, 2, dtype=torch.long),
        "conn_pts": torch.zeros(B, CONN_MAX, K_CONN, 2),
        "conn_valid": torch.zeros(B, CONN_MAX, dtype=torch.bool),
        "pair_id": torch.full((B,), -1, dtype=torch.long),
    }
    for k in range(B):
        b["node_valid"][k, :2] = True
        b["node_type"][k, 0], b["node_type"][k, 1] = 3, 4
        b["node_pos"][k, 0], b["node_pos"][k, 1] = torch.tensor([5.0, 12.0]), torch.tensor([20.0, 12.0])
        b["lane_valid"][k, 0] = True
        xs = torch.linspace(5.0, 20.0, K_LANE)
        b["lane_pts"][k, 0] = torch.stack([xs, torch.full_like(xs, 12.0)], -1)
        b["lane_w"][k, 0] = 3.2
        b["lane_role"][k, 0] = 0
        b["lane_perm"][k, 0] = 0
        b["lane_from"][k, 0], b["lane_to"][k, 0] = 0, 1
        b["dense"][k, 0, 20:40, 20:40] = 1.0
    return b


@pytest.mark.cuda
def test_cuda_bilinear_matches_grid_sample():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    f = torch.randn(2, 16, 40, 52, device="cuda")
    g = torch.rand(2, 300, 2, device="cuda") * 2.4 - 1.2
    fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
    ref = grid_sample_reference(fr, gr)
    w = torch.randn_like(ref)
    (ref * w).sum().backward()
    fm, gm = f.clone().requires_grad_(), g.clone().requires_grad_()
    out = bilinear_sample(fm, gm)
    (out * w).sum().backward()
    torch.cuda.synchronize()
    assert (out - ref).abs().max().item() < 1e-5
    assert (fm.grad - fr.grad).abs().max().item() < 1e-5
    assert ((gm.grad - gr.grad).abs().max() / gr.grad.abs().max()).item() < 1e-5


@pytest.mark.cuda
def test_cuda_amp_calibration_logic():
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    seed_everything(0)
    model = build_model(_mini_model_cfg()).to(dev)
    crit, lc = SimCriterion(), load_config("losses")
    b = batch_to(_dummy_batch(1, 128, 128), dev)
    model.zero_grad(set_to_none=True)
    t32, _, _, _ = train_step(model, b, crit, lc, dev, None)
    t32.backward()
    g32 = grads(model)
    assert torch.isfinite(g32).all()
    amp_res = {"fp32": {"step_s": 0.10, "loss_finite": True}}
    mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    model.zero_grad(set_to_none=True)
    ta, _, _, _ = train_step(model, b, crit, lc, dev, amp_dtype(mode))
    ta.backward()
    ga = grads(model)
    fin = bool(len(ga) == len(g32) and torch.isfinite(ga).all().item())
    cos = float(torch.nn.functional.cosine_similarity(g32, ga, dim=0)) if fin else float("nan")
    amp_res[mode] = {"step_s": 0.05, "loss_finite": bool(torch.isfinite(ta).item()), "grad_finite": fin,
                     "grad_cos_vs_fp32": cos}
    decision = decide_amp(amp_res, load_config("train")["calibration"]["amp_grad_cos_min"])
    assert decision in ("fp32", "bf16", "fp16")


@pytest.mark.cuda
def test_cuda_checkpoint_cross_device_roundtrip(tmp_path):
    assert torch.cuda.is_available()
    cfg = _mini_model_cfg()
    seed_everything(7)
    m_cuda = build_model(cfg).to("cuda").eval()
    ckpt = tmp_path / "cross.pt"
    state = {"model": {k: v.detach().cpu() for k, v in m_cuda.state_dict().items()},
             "config_snapshot": {"configs": {"model": cfg}}, "step": 1}
    torch.save(state, ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in raw["model"].values())
    m_cpu, _ = load_checkpoint_model(ckpt, device="cpu")
    m_gpu, _ = load_checkpoint_model(ckpt, device="cuda")
    b = _dummy_batch(1, 128, 128)
    with torch.no_grad():
        o_cpu = m_cpu(b)
        o_gpu = m_gpu(batch_to(b, torch.device("cuda")))
    for k in ("dense_raw", "lane_pts", "lane_score", "node_pos", "E_succ"):
        a = o_cpu[k].float().cpu()
        c = o_gpu[k].float().cpu()
        rel = (a - c).abs().max().item() / max(a.abs().max().item(), 1.0)
        assert rel < 1e-3, f"{k} 跨设备偏差过大：{rel}"
