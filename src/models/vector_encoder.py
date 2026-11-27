"""矢量编码器（docs/DESIGN.md §3.2 双编码器融合-2）：Polyline-MLP 聚合点序列为实体 Token（D=256），
再经带相对位置偏置（RPE）的自注意力建模长距离矢量上下文。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import FFN, MLP, SelfAttn, sine_embed


class VectorEncoder(nn.Module):
    def __init__(self, d: int, h: int, dff: int, cfg: dict, pos_scale: float):
        super().__init__()
        self.h = h
        self.pos_scale = pos_scale
        ph = cfg["point_hidden"]
        self.point_mlp = MLP(2 + 2 + 2, ph, ph, 2)                 # 相对质心坐标 + 切向 + 绝对位置/尺度
        self.point_mlp2 = MLP(2 * ph, ph, d, 2)
        self.cls_emb = nn.Embedding(cfg["n_classes"], d)
        self.attr = MLP(2 + 2 + 6, ph, d, 2)                      # 尺寸、箭头朝向、6 维转向
        self.pos = MLP(d, d, d, 2)
        self.layers = nn.ModuleList()
        for _ in range(cfg["layers"]):
            self.layers.append(nn.ModuleList([SelfAttn(d, h), FFN(d, dff)]))
        self.rpe = MLP(4, cfg["rpe_hidden"], h * cfg["layers"], 2)
        self.rpe_scale = cfg["rpe_scale_m"]

    def forward(self, tok: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (Token 特征 [B, M, D], 质心米制坐标 [B, M, 2], 掩码 [B, M])。"""
        pts, tan, mask = tok["tok_pts"], tok["tok_tan"], tok["tok_mask"]
        B, M, P, _ = pts.shape
        ctr = pts.mean(2)
        rel = (pts - ctr[:, :, None]) / 10.0
        scale = torch.stack([tok["tok_size"][..., 1] / 20.0, tok["tok_size"][..., 0]], -1)[:, :, None].expand(-1, -1, P, -1)
        f = self.point_mlp(torch.cat([rel, tan, scale], -1))                       # [B, M, P, ph]
        g = f.max(2, keepdim=True).values.expand_as(f)
        x = self.point_mlp2(torch.cat([f, g], -1)).max(2).values                  # [B, M, D]
        x = x + self.cls_emb(tok["tok_cls"]) + self.attr(torch.cat(
            [tok["tok_size"] / 10.0, tok["tok_heading"], tok["tok_turns"]], -1))
        pe = self.pos(sine_embed(ctr, x.shape[-1], self.pos_scale))
        # 相对位置偏置：两两质心差与朝向一致性
        d = (ctr[:, :, None] - ctr[:, None]) / self.rpe_scale                     # [B, M, M, 2]
        hd = tok["tok_heading"]
        tdir = tan.mean(2)
        dirv = torch.where(hd.abs().sum(-1, keepdim=True) > 0, hd, tdir)
        cos = (dirv[:, :, None] * dirv[:, None]).sum(-1, keepdim=True)
        dist = d.norm(dim=-1, keepdim=True)
        bias_all = self.rpe(torch.cat([d, cos, dist], -1))                        # [B, M, M, H·L]
        L = len(self.layers)
        bias_all = bias_all.view(B, M, M, L, self.h).permute(3, 0, 4, 1, 2)       # [L, B, H, M, M]
        safe = mask.clone()
        safe[~mask.any(1), 0] = True
        for k, (attn, ffn) in enumerate(self.layers):
            x = attn(x, pos=pe, bias=bias_all[k], key_padding_mask=safe)
            x = ffn(x)
        x = x * mask[..., None].to(x.dtype)
        return x, ctr, mask
