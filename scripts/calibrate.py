"""训练前校准（docs/DESIGN.md §6.1-3、§6.2-3 第 4 步）：

- 前反向基准（固定 896 px 窗口，fp32 与可用 AMP 精度）；
- AMP 判定：同一批数据比较 AMP 与 fp32 梯度余弦相似度（≥ configs/train.yaml amp_grad_cos_min）且损失有限才启用；
  MPS 已实测定型 fp32（§8.3），只记录；
- 显存内最大 batch（CUDA：峰值显存 ≤ 总显存 × mem_headroom）；
- DataLoader 进程数（依据 CPU 核数与在线光栅化实测吞吐）。

结果写入 <run_dir>/calibration.json；训练配置据此确定，命令行可显式覆盖。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True

from src.common.config import load_config  # noqa: E402
from src.common.device import amp_dtype, batch_to, device_info, select_device, synchronize  # noqa: E402
from src.data.dataset import SynthWindowDataset, collate  # noqa: E402
from src.models.b2s_net import build_model  # noqa: E402
from src.models.losses import SimCriterion  # noqa: E402
from src.train.trainer import seed_everything, train_step  # noqa: E402


def grads(model) -> torch.Tensor:
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])


def bench(model, batch, crit, lc, dev, mode, steps) -> dict:
    ad = amp_dtype(mode)
    ts = []
    for k in range(steps + 2):
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(dev)
        t0 = time.time()
        model.zero_grad(set_to_none=True)
        total, _, _, _ = train_step(model, batch, crit, lc, dev, ad)
        total.backward()
        synchronize(dev)
        if k >= 2:
            ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
    return {"mode": mode, "step_s": float(np.median(ts)), "peak_gib": peak, "loss_finite": bool(torch.isfinite(total))}


def decide_amp(amp_results: dict, grad_cos_min: float) -> str:
    """根据实测梯度余弦相似度、有限性与耗时决定 AMP 模式（docs/DESIGN.md §6.1-3）。"""
    fp32_s = amp_results["fp32"]["step_s"]
    cands = []
    for mode in ("bf16", "fp16"):
        r = amp_results.get(mode)
        if not r:
            continue
        cos = r.get("grad_cos_vs_fp32", float("nan"))
        elig = bool(np.isfinite(cos) and cos >= grad_cos_min and r.get("loss_finite") and r.get("grad_finite", True))
        r["eligible"] = elig
        if elig and r["step_s"] < fp32_s:
            cands.append(mode)
    return min(cands, key=lambda m: amp_results[m]["step_s"]) if cands else "fp32"


def compute_recommendation(mb: dict, amp: dict, decision: str, workers: int, train_cfg: dict, synth_cfg: dict) -> dict:
    """根据显存实测最大窗口数与跨块成对窗口冗余计算推荐 batch_size 与显存门槛。

    训练期 DataLoader 启用 ``pair_prob > 0`` 时，单步张量窗口数最坏为 ``batch_size + max_pairs_per_batch``
    （未设上限时为 ``2 * batch_size``）；推荐 ``batch_size`` 预留该冗余以避免正式训练 OOM。
    ``min_free_vram_gib`` 取实际目标窗口数对应的成功试探峰值（而非末次失败/OOM 项）× 1.15。
    """
    best_win = int(mb.get("max_windows", mb.get("max_batch", 1)))
    pair_prob = float(synth_cfg.get("window", {}).get("pair_prob", 0.0))
    max_pairs = train_cfg.get("max_pairs_per_batch")
    if pair_prob > 0:
        if max_pairs is not None:
            pair_extra = min(int(max_pairs), max(best_win - 1, 0))
            safe_bs = max(1, best_win - pair_extra)
        else:
            safe_bs = max(1, best_win // 2)
            pair_extra = safe_bs
    else:
        pair_extra = 0
        safe_bs = max(1, best_win)
    rec_bs = min(safe_bs, int(train_cfg["batch_size"]) * 4)
    worst_win = min(best_win, rec_bs + pair_extra)
    ok_tried = [x for x in mb.get("tried", []) if x.get("ok") and np.isfinite(x.get("peak_gib", float("nan")))]
    chosen = next((x for x in reversed(ok_tried) if x["batch"] <= worst_win), None)
    if chosen is None and ok_tried:
        chosen = ok_tried[-1]
    peak = chosen["peak_gib"] if chosen is not None else amp.get(decision, amp.get("fp32", {})).get("peak_gib")
    min_vram = round(float(peak) * 1.15, 1) if peak is not None and np.isfinite(peak) else None
    return {"batch_size": int(rec_bs), "max_windows_fit": best_win, "worst_case_windows": int(rec_bs + pair_extra),
            "num_workers": int(workers), "amp": decision, "min_free_vram_gib": min_vram}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="仅 1 个批量规格（本机 / 单测）")
    args = ap.parse_args()
    cc = load_config("train")["calibration"]
    dev = select_device(args.device)
    seed_everything(0)
    crit, lc = SimCriterion(), load_config("losses")
    ds = SynthWindowDataset("val", train=False, seed=0, pair=False, fixed_px=cc["px"] if not args.quick else 512)
    rep = {"device": device_info(dev), "px": cc["px"], "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    model = build_model().to(dev)
    b1 = batch_to(collate([ds[0]]), dev)
    # ---------------- AMP 判定
    model.zero_grad(set_to_none=True)
    seed_everything(1)
    t, _, _, _ = train_step(model, b1, crit, lc, dev, None)
    t.backward()
    g32 = grads(model)
    amp = {"fp32": bench(model, b1, crit, lc, dev, "fp32", cc["bench_steps"] if not args.quick else 1)}
    decision = "fp32"
    if dev.type == "cuda":
        for mode in ("bf16", "fp16"):
            if mode == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            model.zero_grad(set_to_none=True)
            seed_everything(1)
            t2, _, _, _ = train_step(model, b1, crit, lc, dev, amp_dtype(mode))
            t2.backward()
            ga = grads(model)
            grad_fin = bool(len(ga) == len(g32) and torch.isfinite(ga).all().item())
            cos = float(torch.nn.functional.cosine_similarity(g32, ga, dim=0)) if grad_fin else float("nan")
            r = bench(model, b1, crit, lc, dev, mode, cc["bench_steps"])
            r["grad_cos_vs_fp32"] = cos
            r["grad_finite"] = grad_fin
            r["loss_finite"] = bool(r["loss_finite"] and torch.isfinite(t2).item())
            amp[mode] = r
        decision = decide_amp(amp, cc["amp_grad_cos_min"])
    else:
        amp["note"] = "MPS / CPU：按 §8.3 实测定型 fp32"
    rep["amp"] = {**amp, "decision": decision}
    # ---------------- 显存内最大窗口数（pair=False，每样本 1 窗）
    mb = {"tried": []}
    best = 1
    if dev.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        for bs in cc["max_batch_try"]:
            try:
                bb = batch_to(collate([ds[i] for i in range(bs)]), dev)
                r = bench(model, bb, crit, lc, dev, decision, 2)
                ok = r["peak_gib"] <= total_mem * cc["mem_headroom"]
                mb["tried"].append({"batch": bs, **r, "ok": ok})
                if not ok:
                    break
                best = bs
                del bb
            except torch.OutOfMemoryError:
                mb["tried"].append({"batch": bs, "oom": True})
                torch.cuda.empty_cache()
                break
        mb["total_gib"] = total_mem
    mb["max_windows"] = best
    mb["max_batch"] = best
    rep["max_batch"] = mb
    # ---------------- DataLoader 进程数
    ncpu = os.cpu_count() or 4
    t0 = time.time()
    n = 4 if args.quick else 12
    for i in range(n):
        ds[i]
    per = (time.time() - t0) / n
    step_s = amp[decision]["step_s"] if decision in amp else amp["fp32"]["step_s"]
    need = per * max(best, 1) / max(step_s, 1e-3) * 1.5
    workers = int(min(max(2, np.ceil(need)), max(ncpu - 2, 2)))
    rep["dataloader"] = {"cpu_count": ncpu, "sample_s": per, "workers": workers}
    rep["recommended"] = compute_recommendation(mb, amp, decision, workers, load_config("train")["layer1"],
                                                load_config("synth"))
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
