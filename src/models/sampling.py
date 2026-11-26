"""纯 MPS 原生可微双线性采样与多尺度 Deformable Attention（docs/DESIGN.md §3.2 融合 / §6）。

PyTorch 2.8.0 MPS 未实现 ``aten::grid_sampler_2d_backward``（M0 实测直接 NotImplementedError），
因此以 ``floor + gather + 显式双线性权重`` 实现与 ``F.grid_sample(mode="bilinear",
padding_mode="zeros", align_corners=False)`` 数值等价的采样；前向与反向全部为 MPS 原生算子
（gather / scatter_add / 逐元素运算），可在 ``PYTORCH_ENABLE_MPS_FALLBACK=0`` 下运行。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_sample(feat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """feat [B, C, H, W]，grid [B, Q, 2]（归一化 [-1, 1]，(x, y)）→ [B, Q, C]。"""
    B, C, H, W = feat.shape
    Q = grid.shape[1]
    x = ((grid[..., 0] + 1) * W - 1) * 0.5
    y = ((grid[..., 1] + 1) * H - 1) * 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1 - wx1
    wy0 = 1 - wy1
    x0i = x0.long()
    y0i = y0.long()
    flat = feat.reshape(B, C, H * W)
    out = feat.new_zeros(B, C, Q)
    for dx, dy, w in ((0, 0, wx0 * wy0), (1, 0, wx1 * wy0), (0, 1, wx0 * wy1), (1, 1, wx1 * wy1)):
        xi = x0i + dx
        yi = y0i + dy
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1))
        v = torch.gather(flat, 2, idx[:, None, :].expand(B, C, Q))
        out = out + v * (w * valid.to(w.dtype))[:, None, :]
    return out.transpose(1, 2)


class MSDeformAttn(nn.Module):
    """多尺度可变形注意力（纯张量实现，Deformable-DETR 公式；采样走 bilinear_sample）。"""

    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d, self.L, self.H, self.P = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset()

    def _reset(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.H, dtype=torch.float32) * (2.0 * math.pi / self.H)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.H, 1, 1, 2).repeat(1, self.L, self.P, 1)
        for i in range(self.P):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, ref: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        """query [B, Q, D]；ref [B, Q, 2] 归一化 [0, 1] (x, y)；values: L 个 [B, D, H_l, W_l]。"""
        query = query.contiguous()          # MPS：stride-0（expand）输入经 Linear 会触发 MPSNDArray 断言
        ref = ref.contiguous()
        B, Q, D = query.shape
        Hh, L, P = self.H, self.L, self.P
        dh = D // Hh
        off = self.sampling_offsets(query).view(B, Q, Hh, L, P, 2)
        attn = self.attention_weights(query).view(B, Q, Hh, L * P).softmax(-1).view(B, Q, Hh, L, P)
        out = query.new_zeros(B, Hh, Q, dh)
        for l, v in enumerate(values):
            _, _, Hl, Wl = v.shape
            val = self.value_proj(v.flatten(2).transpose(1, 2))               # [B, HW, D]
            val = val.view(B, Hl * Wl, Hh, dh).permute(0, 2, 3, 1).reshape(B * Hh, dh, Hl, Wl)
            norm = torch.tensor([Wl, Hl], dtype=query.dtype, device=query.device)
            loc = ref[:, :, None, None, :] + off[:, :, :, l] / norm             # [B, Q, Hh, P, 2]
            grid = (loc * 2 - 1).permute(0, 2, 1, 3, 4).reshape(B * Hh, Q * P, 2)
            s = bilinear_sample(val, grid).view(B, Hh, Q, P, dh)
            w = attn[:, :, :, l].permute(0, 2, 1, 3)[..., None]                # [B, Hh, Q, P, 1]
            out = out + (s * w).sum(3)
        out = out.permute(0, 2, 1, 3).reshape(B, Q, D)
        return self.output_proj(out)


def grid_sample_reference(feat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """CPU 参考实现：F.grid_sample 结果整理为 [B, Q, C]。"""
    return F.grid_sample(feat, grid[:, :, None, :], mode="bilinear", padding_mode="zeros",
                         align_corners=False)[..., 0].transpose(1, 2)
