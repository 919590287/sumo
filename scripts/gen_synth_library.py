"""生成 Layer 1 合成轻量矢量库（docs/DESIGN.md §5 Layer 1-3）。

用法（仓库根目录）：
    /opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.gen_synth_library --split train [--count N] [--workers 10]
    ... --split tile_test --force-kind corridor      # 超大选区分块拼接验证集（>172.8 m 走廊场景）

每个样本均通过 netconvert 往返一致性编译校验后才入库；输出位于 APFS 缓存（B2S_SYNTH_DIR 可覆盖）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

from src.common.config import load_config  # noqa: E402
from src.data.synth.library import build_split, library_root  # noqa: E402


def main() -> int:
    cfg = load_config("synth")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--force-kind", default=None, choices=[None, "junction", "corridor", "segment"])
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--no-compile-check", action="store_true")
    args = ap.parse_args()
    count = args.count or cfg["counts"].get(args.split)
    if not count:
        ap.error(f"未给出 --count，且 configs/synth.yaml counts 中无 {args.split}")
    seed_base = args.seed_base
    if seed_base is None and args.split not in cfg["seeds"]:
        seed_base = cfg["seeds"]["test"] + 50_000_000
    print(f"合成库：{library_root()}  split={args.split} count={count} workers={args.workers}", flush=True)
    stats = build_split(args.split, count, workers=args.workers, compile_check=not args.no_compile_check,
                        force_kind=args.force_kind, seed_base=seed_base,
                        progress=lambda m: print(m, flush=True))
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
