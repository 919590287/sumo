"""双向跨模态融合（docs/DESIGN.md §3.2 双编码器融合-3）。

每层：
- BEV-to-Vector：矢量 Token 以其质心为参考点，对 {F_4, F_8, F_16} 做多尺度可变形交叉注意力
  （gather 双线性采样，MPS / CUDA 同一实现）；
- Vector-to-BEV：F_8 / F_16 像素对矢量 Token 做交叉注意力（键为至多 512 个 Token），F_4' 由 F_8' 的增量
  上采样更新，F_2 融合 F_4' 的上采样语义（专供稠密细标线感知头）。
实现说明：Vector-to-BEV 方向的参考集合是稀疏 Token（无规则网格），因此以全局交叉注意力代替网格可变形
采样；F_4 层级（896² 窗口下 5 万像素）的稠密交叉注意力显存过高，改为经 F_8' 增量上采样（见 §8 M1 记录）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import FFN, CrossAttn, Frame, sine_embed
from src.models.sampling import MSDeformAttn


class FusionLayer(nn.Module):
    def __init__(self, d: int, h: int, dff: int, n_points: int, v2b_levels: list[str], pos_scale: float):
        super().__init__()
        self.b2v_norm = nn.LayerNorm(d)
        self.b2v = MSDeformAttn(d, 3, h, n_points)
        self.b2v_ffn = FFN(d, dff)
        self.v2b_levels = v2b_levels
        self.v2b = nn.ModuleDict({k: CrossAttn(d, h) for k in v2b_levels})
        self.v2b_ffn = nn.ModuleDict({k: FFN(d, dff) for k in v2b_levels})
        self.pos_scale = pos_scale

    def forward(self, feats: dict, vec, vec_ctr, vec_mask, frame: Frame, bev_pos: dict):
        # ---------------- BEV → Vector
        ref = frame.to_norm(vec_ctr)
        upd = self.b2v(self.b2v_norm(vec), ref, [feats["F4"], feats["F8"], feats["F16"]])
        vec = self.b2v_ffn(vec + upd * vec_mask[..., None].to(vec.dtype))
        vpos = sine_embed(vec_ctr, vec.shape[-1], self.pos_scale)
        # ---------------- Vector → BEV
        out = dict(feats)
        delta8 = None
        for k in self.v2b_levels:
            f = feats[k]
            B, C, H, W = f.shape
            q = f.flatten(2).transpose(1, 2)
            q2 = self.v2b_ffn[k](self.v2b[k](q, vec, qpos=bev_pos[k], kpos=vpos, key_mask=vec_mask))
            out[k] = q2.transpose(1, 2).reshape(B, C, H, W)
            if k == "F8":
                delta8 = out[k] - f
        if delta8 is not None:
            out["F4"] = feats["F4"] + F.interpolate(delta8, size=feats["F4"].shape[-2:], mode="bilinear",
                                                    align_corners=False)
        return out, vec


def bev_positions(feats: dict, frame: Frame, strides: dict, d: int, pos_scale: float) -> dict:
    """各层级像素中心的米制正弦位置编码 [B, H·W, D]。"""
    out = {}
    for k, s in strides.items():
        f = feats[k]
        B, _, H, W = f.shape
        rows = torch.arange(H, device=f.device)
        cols = torch.arange(W, device=f.device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        b_idx = torch.arange(B, device=f.device)[:, None].expand(B, H * W)
        xy = frame.from_pixel(cc.flatten()[None].expand(B, -1), rr.flatten()[None].expand(B, -1), s, b_idx)
        out[k] = sine_embed(xy, d, pos_scale)
    return out
