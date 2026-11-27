"""模型通用组件：米制坐标 ↔ 归一化特征图坐标、正弦位置编码、MLP、注意力块（设备无关）。

坐标约定：模型内部全部几何量为**窗口局部米制坐标**（x 向东、y 向北，原点为窗口左下角）；
在特征图上采样时按逐样本真实高度换算到归一化 [0, 1] 图像坐标（u 向右、v 向下）：
    u = x / (W_canvas · res)，v = (H_true · res − y) / (H_canvas · res)
批内不同尺寸窗口在右 / 下方补零，因此已有像素的归一化坐标与补零无关。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Frame:
    """一个批次的米制 ↔ 归一化坐标换算器。"""

    def __init__(self, hw_true: torch.Tensor, canvas_hw: tuple[int, int], res: float):
        self.h_true = hw_true[:, 0].to(torch.float32) * res          # [B] 米
        self.w_canvas = canvas_hw[1] * res
        self.h_canvas = canvas_hw[0] * res
        self.res = res

    def to_norm(self, xy: torch.Tensor) -> torch.Tensor:
        """xy [B, ..., 2] 米 → [B, ..., 2] 归一化 (u, v)。"""
        shape = (xy.shape[0],) + (1,) * (xy.dim() - 2)
        h = self.h_true.to(xy.device).view(shape)
        u = xy[..., 0] / self.w_canvas
        v = (h - xy[..., 1]) / self.h_canvas
        return torch.stack([u, v], -1)

    def from_pixel(self, col: torch.Tensor, row: torch.Tensor, stride: int, b_idx: torch.Tensor) -> torch.Tensor:
        """特征图（步长 stride，像素中心）→ 米。"""
        x = (col.to(torch.float32) + 0.5) * stride * self.res
        y = self.h_true.to(col.device)[b_idx] - (row.to(torch.float32) + 0.5) * stride * self.res
        return torch.stack([x, y], -1)


def sine_embed(xy: torch.Tensor, d: int, scale: float, temperature: float = 10000.0) -> torch.Tensor:
    """二维正弦位置编码：xy [..., 2]（米）→ [..., d]。"""
    half = d // 2
    dim_t = torch.arange(half // 2, dtype=torch.float32, device=xy.device)
    dim_t = temperature ** (2 * dim_t / half)
    p = xy[..., :, None] / scale * 2 * math.pi / dim_t                     # [..., 2, half/2]
    emb = torch.cat([p.sin(), p.cos()], -1)                                # [..., 2, half]
    return emb.flatten(-2)


class MLP(nn.Module):
    def __init__(self, din: int, dh: int, dout: int, n: int = 2, zero_last: bool = False):
        super().__init__()
        dims = [din] + [dh] * (n - 1) + [dout]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        if zero_last:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for k, l in enumerate(self.layers):
            x = l(x)
            if k < len(self.layers) - 1:
                x = F.gelu(x)
        return x


class SelfAttn(nn.Module):
    """预归一化多头自注意力（可选加性注意力偏置 [B, H, Q, Q] 与键 padding 掩码）。"""

    def __init__(self, d: int, h: int, dropout: float = 0.0):
        super().__init__()
        self.h = h
        self.norm = nn.LayerNorm(d)
        self.qk = nn.Linear(d, 2 * d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, pos=None, bias=None, key_padding_mask=None):
        B, Q, D = x.shape
        y = self.norm(x)
        q, k = self.qk(y if pos is None else y + pos).chunk(2, -1)
        v = self.v(y)
        q, k, v = (t.reshape(B, Q, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        mask = bias
        if key_padding_mask is not None:
            kp = torch.zeros(B, 1, 1, Q, device=x.device, dtype=x.dtype).masked_fill(
                ~key_padding_mask[:, None, None, :], float("-inf"))
            mask = kp if mask is None else mask + kp
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return x + self.out(o.transpose(1, 2).reshape(B, Q, D))


class CrossAttn(nn.Module):
    """预归一化多头交叉注意力（键可带 padding 掩码；长查询按块计算以控制显存）。"""

    def __init__(self, d: int, h: int, chunk: int = 8192):
        super().__init__()
        self.h = h
        self.norm_q = nn.LayerNorm(d)
        self.norm_k = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.chunk = chunk

    def forward(self, x, mem, qpos=None, kpos=None, key_mask=None):
        B, Q, D = x.shape
        K = mem.shape[1]
        y = self.norm_q(x)
        m = self.norm_k(mem)
        q = self.q(y if qpos is None else y + qpos).reshape(B, Q, self.h, D // self.h).transpose(1, 2)
        k = self.k(m if kpos is None else m + kpos).reshape(B, K, self.h, D // self.h).transpose(1, 2)
        v = self.v(m).reshape(B, K, self.h, D // self.h).transpose(1, 2)
        mask = None
        if key_mask is not None:
            km = key_mask.clone()
            empty = ~km.any(1)
            km[empty, 0] = True                                # 全空键集：放行 1 个键，输出再置零
            mask = torch.zeros(B, 1, 1, K, device=x.device, dtype=x.dtype).masked_fill(
                ~km[:, None, None, :], float("-inf"))
        outs = []
        for s0 in range(0, Q, self.chunk):
            outs.append(F.scaled_dot_product_attention(q[:, :, s0:s0 + self.chunk], k, v, attn_mask=mask))
        o = self.out(torch.cat(outs, 2).transpose(1, 2).reshape(B, Q, D))
        if key_mask is not None:
            o = o * key_mask.any(1).to(o.dtype)[:, None, None]
        return x + o


class FFN(nn.Module):
    def __init__(self, d: int, dff: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, dff)
        self.l2 = nn.Linear(dff, d)

    def forward(self, x):
        return x + self.l2(F.gelu(self.l1(self.norm(x))))


def gather_points(feat: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """feat [B, C, H, W]，uv [B, Q, 2] 归一化 [0, 1] → [B, Q, C]（gather 双线性，MPS 原生）。"""
    from src.models.sampling import bilinear_sample
    return bilinear_sample(feat, uv * 2 - 1)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))
