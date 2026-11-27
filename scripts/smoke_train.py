"""本机冒烟训练（docs/DESIGN.md §6.1-1、§6.2-1 构建前置条件）：

1. 少量样本过拟合：固定 ``overfit_samples`` 个窗口反复训练 ``overfit_steps`` 步，末段损失须 ≤ 初始 × pass_ratio；
2. 短程收敛检查：在合成库上训练 ``steps`` 步（真实 DataLoader 多进程 + 在线光栅化），损失须有限且下降。

结果写入 runs/smoke/smoke_<git短哈希>.json（构建训练包时校验该文件与当前提交一致）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True

from src.common.config import REPO_ROOT, cache_dir, load_config  # noqa: E402
from src.common.device import batch_to, device_info, select_device  # noqa: E402
from src.data.dataset import SynthWindowDataset, collate  # noqa: E402
from src.models.b2s_net import build_model  # noqa: E402
from src.models.losses import SimCriterion  # noqa: E402
from src.train.trainer import Trainer, seed_everything, train_step  # noqa: E402


def git_head() -> tuple[str, bool]:
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, cwd=REPO_ROOT).stdout.strip())
    return h, dirty


def overfit(cfg: dict, dev) -> dict:
    seed_everything(0)
    ds = SynthWindowDataset("train", train=False, seed=0, pair=False, fixed_px=cfg["fixed_px"])
    batch = batch_to(collate([ds[i] for i in range(cfg["overfit_samples"])]), dev)
    model = build_model().to(dev)
    crit, lc = SimCriterion(), load_config("losses")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    hist = []
    t0 = time.time()
    for it in range(cfg["overfit_steps"]):
        total, parts, _, _ = train_step(model, batch, crit, lc, dev, None)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        hist.append(float(total))
        if it % 25 == 0:
            print(f"[overfit] {it}/{cfg['overfit_steps']} loss {hist[-1]:.3f}  {time.time() - t0:.0f}s", flush=True)
    first = float(np.mean(hist[:5]))
    last = float(np.mean(hist[-10:]))
    return {"first": first, "last": last, "ratio": last / first, "pass": last <= first * cfg["pass_ratio"]
            and np.isfinite(last), "seconds": time.time() - t0, "curve": hist[::5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-short", action="store_true")
    args = ap.parse_args()
    cfg = load_config("train")["smoke"]
    dev = select_device(args.device)
    head, dirty = git_head()
    rep = {"git": head, "dirty": dirty, "device": device_info(dev), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    rep["overfit"] = overfit(cfg, dev)
    print(f"过拟合：{rep['overfit']['first']:.3f} → {rep['overfit']['last']:.3f}  通过={rep['overfit']['pass']}")
    if not args.skip_short:
        run = cache_dir("runs") / f"smoke_{head}"
        tr = Trainer(run, "layer1", {"max_steps": cfg["steps"], "batch_size": cfg["batch_size"], "grad_accum": 1,
                                     "num_workers": cfg["num_workers"], "lr": cfg["lr"], "warmup_steps": 20,
                                     "val_every_steps": cfg["steps"], "val_samples": 8, "ckpt_every_steps": cfg["steps"],
                                     "log_every_steps": 10, "amp": "fp32"}, device=args.device)
        summ = tr.fit()
        lines = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if '"loss"' in x]
        losses = [x["loss"] for x in lines]
        rep["short"] = {"run_dir": str(run), "steps": summ["steps"], "first": losses[0], "last": float(np.mean(losses[-3:])),
                        "finite": bool(np.all(np.isfinite(losses))), "samples_per_s": lines[-1]["samples_per_s"],
                        "val": summ["final_val"]}
        rep["short"]["pass"] = rep["short"]["finite"] and rep["short"]["last"] < rep["short"]["first"]
    rep["pass"] = rep["overfit"]["pass"] and rep.get("short", {"pass": True})["pass"]
    out = REPO_ROOT / "runs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"smoke_{head}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "overfit"} | {"overfit_pass": rep["overfit"]["pass"]},
                     ensure_ascii=False, indent=1, default=str))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
