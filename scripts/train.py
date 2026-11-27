"""训练入口（设备无关；docs/DESIGN.md §6.1–6.2）。

用法：
    B2S_DEVICE=auto python -B -m scripts.train --stage layer1 --run-id <id> [--resume] [--epochs N]
        [--batch-size B] [--num-workers W] [--max-steps S] [--amp auto|fp32|bf16|fp16] [--runs-root DIR]

训练输出写入 <runs-root>/<run_id>/（默认 APFS 缓存 runs/；训练包内为包内 runs/）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

from src.common.config import cache_dir  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="layer1")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runs-root", default=os.environ.get("B2S_RUNS_ROOT"))
    ap.add_argument("--device", default=None)
    for k, t in (("epochs", int), ("batch_size", int), ("grad_accum", int), ("num_workers", int),
                 ("max_steps", int), ("lr", float), ("val_every_steps", int), ("val_samples", int),
                 ("ckpt_every_steps", int), ("log_every_steps", int), ("fixed_px", int), ("train_length", int)):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=t, default=None)
    ap.add_argument("--amp", default=None, choices=[None, "auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--train-split", default=None)
    ap.add_argument("--val-split", default=None)
    args = ap.parse_args()
    run_id = args.run_id or f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.runs_root) if args.runs_root else cache_dir("runs")
    overrides = {k: getattr(args, k) for k in ("epochs", "batch_size", "grad_accum", "num_workers", "max_steps", "lr",
                                               "val_every_steps", "val_samples", "ckpt_every_steps", "log_every_steps",
                                               "fixed_px", "train_length", "amp", "train_split", "val_split")}
    tr = Trainer(root / run_id, args.stage, overrides, resume=args.resume, device=args.device)
    tr.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
