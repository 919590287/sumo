"""行政区 + 5 km × 5 km 区块的防泄漏空间切分与隔离带判定（docs/DESIGN.md §7.1-1）。

- 区块按行政区分层、以信号灯点位密度为权重确定性随机分配 train / val / test；
- 开创大道 dev 样本所在区块整体划为 dev（val-dev），不进入 train 与 test；
- 隔离带：样本中心距任一“相邻且集合不同”的区块矩形 < gap/2 + 最大单窗半跨度 时判为 buffer，
  从而任意两个不同集合样本的窗口之间间距 ≥ gap（500 m）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.config import REPO_ROOT, load_config


@dataclass
class SplitManifest:
    version: str
    block_size: float
    origin: tuple[float, float]
    blocks: dict[tuple[int, int], dict]      # (ix, iy) → {split, district, n_signal, ...}
    margin: float                            # gap/2 + window_half_extent

    @classmethod
    def load(cls, path: Path | None = None) -> "SplitManifest":
        cfg = load_config("splits")
        path = Path(path or REPO_ROOT / cfg["manifest_path"])
        d = json.loads(path.read_text(encoding="utf-8"))
        blocks = {(b["ix"], b["iy"]): b for b in d["blocks"]}
        return cls(d["version"], d["block_size_m"], tuple(d["origin"]), blocks, d["buffer_margin_m"])

    def block_of(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor((x - self.origin[0]) / self.block_size)),
                int(math.floor((y - self.origin[1]) / self.block_size)))

    def assign(self, x: float, y: float) -> str:
        """返回 'train' | 'val' | 'test' | 'dev' | 'buffer' | 'unassigned'（UTM 全局坐标）。"""
        ix, iy = self.block_of(x, y)
        b = self.blocks.get((ix, iy))
        if b is None:
            return "unassigned"
        s = b["split"]
        bs = self.block_size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nb = self.blocks.get((ix + dx, iy + dy))
                if nb is None or nb["split"] == s:
                    continue
                x0 = self.origin[0] + (ix + dx) * bs
                y0 = self.origin[1] + (iy + dy) * bs
                ddx = max(x0 - x, 0.0, x - (x0 + bs))
                ddy = max(y0 - y, 0.0, y - (y0 + bs))
                if math.hypot(ddx, ddy) < self.margin:
                    return "buffer"
        return s


def assign_blocks(block_district: dict, block_weight: dict, dev_blocks: set, cfg: dict) -> dict:
    """按行政区分层：区块按权重降序（同权重以种子随机打散）依次分给当前缺口最大的集合。"""
    rng = np.random.default_rng(cfg["seed"])
    frac = cfg["fractions"]
    out = {}
    by_d: dict[str, list] = {}
    for b, d in sorted(block_district.items()):
        if b in dev_blocks:
            out[b] = "dev"
        else:
            by_d.setdefault(d, []).append(b)
    for d in sorted(by_d):
        blocks = by_d[d]
        tie = rng.permutation(len(blocks))
        order = sorted(range(len(blocks)), key=lambda i: (-(block_weight.get(blocks[i], 0)), tie[i]))
        total = sum(block_weight.get(b, 0) + 1 for b in blocks)
        acc = {k: 0.0 for k in frac}
        for i in order:
            b = blocks[i]
            w = block_weight.get(b, 0) + 1
            deficit = {k: frac[k] * total - acc[k] for k in frac}
            k = max(deficit, key=lambda s: (deficit[s], s == "train"))
            out[b] = k
            acc[k] += w
    return out
