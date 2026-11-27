"""M1 完整模型、损失、设备抽象、训练器、校准、推理评测与成果校验单元测试（docs/DESIGN.md §4–§7）。"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from shapely.geometry import box

from scripts.calibrate import compute_recommendation, decide_amp
from scripts.eval_synth import oracle_pred
from scripts.export_reference import KEYS as REF_KEYS, compare as compare_ref
from scripts.import_results import sha256 as file_sha256, verify as verify_sha256
from src.common.config import load_config
from src.common.device import amp_dtype, batch_to, select_device
from src.data.dataset import SynthWindowDataset, collate
from src.data.synth.library import build_split
from src.eval.quick import quick_metrics
from src.eval.synth_eval import aggregate, eval_geometry
from src.graph.stitching import plan_tiles_avoiding
from src.infer.predict import predict_batch
from src.models.b2s_net import B2SNet, build_model, count_params, to_graph_tensors
from src.models.losses import SimCriterion
from src.models.matching import conn_pairs, match_all
from src.train.trainer import Trainer, load_checkpoint_model, seed_everything, train_step


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


@pytest.fixture(scope="module")
def mini_synth_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_unit")
    build_split("train", count=4, root=root, workers=2, compile_check=False,
                force_kind="junction", seed_base=900, progress=lambda *_: None)
    build_split("val", count=2, root=root, workers=2, compile_check=False,
                force_kind="junction", seed_base=950, progress=lambda *_: None)
    return root


def _make_paired_batch(root, fixed_px: int = 256):
    """构造含真实合成样本 + 共享 port_key 跨块对的批次（验证全部损失项含 L_port_match）。"""
    ds = SynthWindowDataset("train", root=root, train=False, seed=3, randomize=False, pair=False, fixed_px=fixed_px)
    s0 = ds[0][0]
    # 复制样本作为成对窗（相同 port_key 端口触发跨块对比损失）
    s0_b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s0.items()}
    return collate([[s0, s0_b]])


def test_device_selection_and_utils(monkeypatch):
    monkeypatch.setenv("B2S_DEVICE", "cpu")
    assert select_device().type == "cpu"
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in ("cuda", "mps", "cpu")
    with pytest.raises(ValueError):
        select_device("tpu")
    assert amp_dtype("fp32") is None
    assert amp_dtype("bf16") == torch.bfloat16
    assert amp_dtype("fp16") == torch.float16
    out = batch_to({"a": torch.ones(2), "m": ["meta"]}, torch.device("cpu"))
    assert out["a"].device.type == "cpu" and out["m"] == ["meta"]


def test_default_model_param_budget():
    n = count_params(build_model())
    assert 25_000_000 <= n <= 40_000_000, f"默认模型参数量 {n / 1e6:.2f}M 超出 25M–40M 预算"


def test_b2s_net_forward_backward_and_all_losses(mini_synth_dir):
    seed_everything(0)
    dev = torch.device("cpu")
    model = B2SNet(_mini_model_cfg()).to(dev)
    crit, lc = SimCriterion(), load_config("losses")
    batch = _make_paired_batch(mini_synth_dir, fixed_px=256)
    total, parts, fo, matches = train_step(model, batch, crit, lc, dev, None)
    assert torch.isfinite(total), parts
    for key in ("d_valid", "d_center", "l_pts", "l_score", "n_pos", "n_poly_end", "t_adj", "t_succ",
                "t_from", "t_to", "k_pts", "port"):
        assert key in parts and np.isfinite(parts[key]), f"缺失或非有限损失项：{key}"
    total.backward()
    for mod_name in ("bev", "vec", "fusion", "dense", "decoder", "topo", "conn"):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(mod_name + ".") and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), f"模块 {mod_name} 缺少有限梯度"
    # 快速指标接口
    qm = quick_metrics(fo, batch, matches[-1])
    assert {"lane_f1", "node_f1", "cw_f1", "succ_f1", "lat_p50", "lat_p90"} <= set(qm)


@pytest.mark.mps
def test_b2s_net_mps_forward_backward_no_fallback(mini_synth_dir):
    seed_everything(0)
    dev = torch.device("mps")
    model = B2SNet(_mini_model_cfg()).to(dev)
    crit, lc = SimCriterion(), load_config("losses")
    batch = batch_to(_make_paired_batch(mini_synth_dir, fixed_px=256), dev)
    total, parts, _, _ = train_step(model, batch, crit, lc, dev, None)
    assert torch.isfinite(total), parts
    total.backward()                    # PYTORCH_ENABLE_MPS_FALLBACK=0：全网零 CPU 回退
    torch.mps.synchronize()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_predict_batch_oracle_and_synth_eval(mini_synth_dir):
    ds = SynthWindowDataset("val", root=mini_synth_dir, train=False, seed=5, randomize=False, pair=False, fixed_px=512)
    batch = collate([ds[0]])
    meta = batch["meta"][0]
    gt = meta["graph"]
    # 1. Oracle 评测自检：真值张量经约束解码 → 几何与拓扑指标应近满分
    pred_o = oracle_pred(gt)
    assert pred_o.ok, pred_o.error
    rec = eval_geometry(pred_o.result.graph, gt, meta, batch["hw"][0].tolist(), 0.20)
    rec.update({"decode_ok": True, "raw_dead_end": pred_o.result.raw.dead_end,
                "raw_unfed": pred_o.result.raw.unfed, "raw_crossings": pred_o.result.raw.same_approach_crossings,
                "decoded_dead_end": pred_o.result.decoded.dead_end, "decoded_unfed": pred_o.result.decoded.unfed,
                "decoded_crossings": pred_o.result.decoded.same_approach_crossings})
    summ = aggregate([rec])
    assert summ["lane_f1"] == pytest.approx(1.0)
    assert summ["lat_p50_m"] < 1e-4
    assert summ["neg_fpr"] < 0.01
    # 2. 真实模型推理管线（predict_batch + to_graph_tensors）
    model = B2SNet(_mini_model_cfg()).eval()
    preds, out = predict_batch(model, batch, torch.device("cpu"))
    assert len(preds) == 1
    t = to_graph_tensors(out, 0)
    assert t.lane_pts.shape == (128, 32, 3) and t.E_succ.shape == (128, 128)


def test_trainer_checkpoint_resume_and_cross_device_load(mini_synth_dir, tmp_path, monkeypatch):
    mini_cfg = _mini_model_cfg()
    monkeypatch.setenv("B2S_SYNTH_DIR", str(mini_synth_dir))
    monkeypatch.setattr("src.train.trainer.build_model", lambda cfg=None: B2SNet(mini_cfg))
    run_dir = tmp_path / "run1"
    ov = {"max_steps": 2, "batch_size": 1, "grad_accum": 1, "num_workers": 0, "fixed_px": 256,
          "warmup_steps": 1, "val_every_steps": 2, "val_samples": 1, "ckpt_every_steps": 2,
          "log_every_steps": 1, "amp": "fp32", "figures": 0, "max_pairs_per_batch": 1}
    tr = Trainer(run_dir, "layer1", ov, resume=False, device="cpu")
    s1 = tr.fit()
    tr.close()
    assert s1["steps"] == 2
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    assert ckpt_path.exists()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in raw["model"].values())
    assert "config_snapshot" in raw and raw["step"] == 2
    # 断点续训至 step 4，校验 ETA 按续训步数计算（非负且合理）
    tr2 = Trainer(run_dir, "layer1", {**ov, "max_steps": 4, "val_every_steps": 4, "ckpt_every_steps": 4},
                  resume=True, device="cpu")
    s2 = tr2.fit()
    tr2.close()
    assert s2["steps"] == 4
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if '"loss"' in x]
    assert [r["step"] for r in rows][-2:] == [3, 4]
    assert rows[-2]["eta_s"] > 0 and rows[-1]["eta_s"] == pytest.approx(0.0)
    # 校验 load_checkpoint_model（写入与权重一致的 mini_cfg 快照）
    raw["config_snapshot"]["configs"]["model"] = mini_cfg
    ckpt_mini = tmp_path / "mini.pt"
    torch.save(raw, ckpt_mini)
    m_loaded, st = load_checkpoint_model(ckpt_mini, device="cpu")
    assert st["step"] == 2
    b = _make_paired_batch(mini_synth_dir, fixed_px=256)
    with torch.no_grad():
        o1 = m_loaded(b)
        m2, _ = load_checkpoint_model(ckpt_mini, device="cpu")
        o2 = m2(b)
    assert torch.allclose(o1["lane_pts"], o2["lane_pts"])


def test_calibrate_amp_and_recommendation_logic():
    # 1. AMP 判定：满足余弦 ≥ 0.99、有限且更快才选，否则回退 fp32
    assert decide_amp({"fp32": {"step_s": 0.4},
                       "bf16": {"step_s": 0.25, "grad_cos_vs_fp32": 0.995, "loss_finite": True, "grad_finite": True},
                       "fp16": {"step_s": 0.22, "grad_cos_vs_fp32": 0.97, "loss_finite": True, "grad_finite": True}},
                      0.99) == "bf16"
    assert decide_amp({"fp32": {"step_s": 0.4},
                       "bf16": {"step_s": 0.25, "grad_cos_vs_fp32": float("nan"), "loss_finite": True, "grad_finite": False}},
                      0.99) == "fp32"
    # 2. 推荐 batch 与显存门槛：末次尝试 OOM 时不能返回 None，且预留跨块成对窗口冗余
    mb_oom = {
        "max_windows": 6,
        "tried": [
            {"batch": 2, "peak_gib": 7.0, "ok": True},
            {"batch": 4, "peak_gib": 13.0, "ok": True},
            {"batch": 6, "peak_gib": 18.0, "ok": True},
            {"batch": 8, "oom": True},
        ],
    }
    rec = compute_recommendation(mb_oom, {"bf16": {"peak_gib": 4.0}}, "bf16", 8,
                                 {"batch_size": 4, "max_pairs_per_batch": 1}, {"window": {"pair_prob": 0.15}})
    assert rec["batch_size"] == 5
    assert rec["worst_case_windows"] == 6 <= rec["max_windows_fit"]
    assert rec["min_free_vram_gib"] == pytest.approx(round(18.0 * 1.15, 1))
    # 3. 末次尝试超 mem_headroom（ok=False）时不得错取失败项峰值
    mb_over = {
        "max_windows": 4,
        "tried": [
            {"batch": 2, "peak_gib": 8.0, "ok": True},
            {"batch": 4, "peak_gib": 15.0, "ok": True},
            {"batch": 6, "peak_gib": 22.0, "ok": False},
        ],
    }
    rec2 = compute_recommendation(mb_over, {"fp32": {"peak_gib": 5.0}}, "fp32", 6,
                                  {"batch_size": 4, "max_pairs_per_batch": 1}, {"window": {"pair_prob": 0.15}})
    assert rec2["batch_size"] == 3 and rec2["worst_case_windows"] == 4
    assert rec2["min_free_vram_gib"] == pytest.approx(round(15.0 * 1.15, 1))
    # 4. 未设 max_pairs_per_batch 时按最坏 2x 折半预留
    rec3 = compute_recommendation(mb_over, {"fp32": {"peak_gib": 5.0}}, "fp32", 6,
                                  {"batch_size": 4}, {"window": {"pair_prob": 0.15}})
    assert rec3["batch_size"] == 2 and rec3["worst_case_windows"] == 4


def test_plan_tiles_avoiding_reads_config():
    cfg = load_config("tiling")
    assert {"search_m", "step_m", "min_core_m"} <= set(cfg["avoid_nodes"])
    # 300 m 走廊需切 2 块，默认切线在 x=150，在 x=150 放节点避让区 → 切线应偏移避开且保持每块 ≤ max_core
    avoid = [box(140.0, 20.0, 160.0, 80.0)]
    tiles = plan_tiles_avoiding((0.0, 0.0, 300.0, 100.0), avoid)
    assert len(tiles) == 2
    cut_x = tiles[0].core[2]
    assert cut_x <= 140.0 or cut_x >= 160.0
    max_core = cfg["max_window_px"] * cfg["resolution_m"] - 2 * cfg["context_buffer_m"]
    min_core = cfg["avoid_nodes"]["min_core_m"]
    for t in tiles:
        w = t.core[2] - t.core[0]
        assert min_core <= w <= max_core + 1e-6


def test_import_verify_and_reference_compare(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("hello", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(f"{file_sha256(f1)}  a.txt\n", encoding="utf-8")
    assert verify_sha256(tmp_path) == []
    f1.write_text("tampered", encoding="utf-8")
    assert verify_sha256(tmp_path) == ["a.txt"]
    arrs = {k: np.ones((2, 4), np.float32) for k in REF_KEYS}
    arrs2 = {k: v + 1e-5 for k, v in arrs.items()}
    rep = compare_ref(arrs, arrs2)
    assert all(rep[k]["rel"] < 1e-4 for k in REF_KEYS)
