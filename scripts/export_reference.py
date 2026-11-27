"""跨设备一致性参考前向输出（docs/DESIGN.md §6.1-4）：固定合成验证样本（确定性窗口、无域随机化）+ 固定权重，
导出关键输出张量；另一设备上以同一输入重算并比较（scripts/import_results.py）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True

from src.common.device import batch_to, select_device  # noqa: E402
from src.data.dataset import SynthWindowDataset, collate  # noqa: E402
from src.train.trainer import load_checkpoint_model  # noqa: E402

KEYS = ("dense_raw", "lane_pts", "lane_score", "node_pos", "node_score", "E_succ", "E_adj", "cw_end")


def reference_batch():
    ds = SynthWindowDataset("val", train=False, seed=4242, pair=False, randomize=False, fixed_px=512)
    return collate([ds[0], ds[1]])


@torch.no_grad()
def forward_outputs(model, dev) -> dict[str, np.ndarray]:
    torch.manual_seed(0)
    out = model(batch_to(reference_batch(), dev))
    return {k: out[k].float().cpu().numpy() for k in KEYS}


def compare(a: dict, b: dict) -> dict:
    rep = {}
    for k in KEYS:
        d = np.abs(a[k] - b[k])
        scale = max(float(np.abs(a[k]).max()), 1.0)
        rep[k] = {"max_abs": float(d.max()), "rel": float(d.max() / scale)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = select_device(args.device)
    model, _ = load_checkpoint_model(Path(args.ckpt), args.device)
    np.savez_compressed(args.out, device=str(dev), **forward_outputs(model, dev))
    print(f"参考前向输出（{dev}）已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
