"""高保真 BEV 编码器（docs/DESIGN.md §3.2 双编码器融合-1）。

轻卷积高保真 Stem（3×3 s1 + 3×3 s2 残差块 → C_2，0.40 m/px）→ ResNet34 对应阶段
（layer1/2/3 → C_4 / C_8 / C_16）→ 4 层 FPN 输出 {F_2(D_2=128), F_4, F_8, F_16(D=256)}。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _stage(cin, cout, blocks, stride):
    down = None
    if stride != 1 or cin != cout:
        down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    layers = [BasicBlock(cin, cout, stride, down)]
    layers += [BasicBlock(cout, cout) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class BevEncoder(nn.Module):
    def __init__(self, in_ch: int = 36, d: int = 256, d2: int = 128, stem_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(in_ch, stem_ch // 2, 3, 1), _stage(stem_ch // 2, stem_ch, 1, 2))  # C_2
        self.layer1 = _stage(stem_ch, 64, 3, 2)      # C_4
        self.layer2 = _stage(64, 128, 4, 2)          # C_8
        self.layer3 = _stage(128, 256, 6, 2)         # C_16
        self.lat16 = nn.Conv2d(256, d, 1)
        self.lat8 = nn.Conv2d(128, d, 1)
        self.lat4 = nn.Conv2d(64, d, 1)
        self.lat2 = nn.Conv2d(stem_ch, d2, 1)
        self.top2 = nn.Conv2d(d, d2, 1)
        self.out16 = nn.Conv2d(d, d, 3, padding=1)
        self.out8 = nn.Conv2d(d, d, 3, padding=1)
        self.out4 = nn.Conv2d(d, d, 3, padding=1)
        self.out2 = nn.Conv2d(d2, d2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c2 = self.stem(x)
        c4 = self.layer1(c2)
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)
        p16 = self.lat16(c16)
        p8 = self.lat8(c8) + F.interpolate(p16, size=c8.shape[-2:], mode="nearest")
        p4 = self.lat4(c4) + F.interpolate(p8, size=c4.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(self.top2(p4), size=c2.shape[-2:], mode="nearest")
        return {"F2": self.out2(p2), "F4": self.out4(p4), "F8": self.out8(p8), "F16": self.out16(p16)}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
