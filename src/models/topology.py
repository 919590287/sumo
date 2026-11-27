"""阶段 D（docs/DESIGN.md §4.2）：GNN 拓扑关系头 + 成对连接轨迹解码头（TopoNet 风格）。

- 拓扑头：最终节点特征 H_V、车道特征 H_L（拼接起终点坐标 / 切向 / 标量宽）与人行横道特征 H_C 经 2 层
  异构图注意力（全连接 + 类型嵌入 + 几何相对偏置）后，由分解式 MLP 边头并行输出
  E_adj、E_from、E_to、E_succ、E_cross、E_cross_node 的 logits；
- 成对连接轨迹头：对有向车道对 (l_i, l_j) 拼接 (h_i, h_j, end_i, tangent_end_i, start_j, tangent_start_j)，
  在首尾连线线性插值初值上执行 2 层多尺度可变形交叉注意力，回归中间 6 个控制点偏移；首尾点精确锁定。
几何特征一律停梯度（拓扑头不反向改动几何）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import FFN, MLP, Frame, SelfAttn, sine_embed
from src.models.sampling import MSDeformAttn


def lane_geom(pts: torch.Tensor, w: torch.Tensor) -> dict:
    """pts [B, N, K, 2] → 起终点 / 端部单位切向 / 中点 / 宽。"""
    t0 = pts[:, :, 1] - pts[:, :, 0]
    t1 = pts[:, :, -1] - pts[:, :, -2]
    return {"start": pts[:, :, 0], "end": pts[:, :, -1], "mid": pts[:, :, pts.shape[2] // 2],
            "t0": t0 / t0.norm(dim=-1, keepdim=True).clamp_min(1e-6),
            "t1": t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-6), "w": w}


def to_frame(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v [..., 2] 投影到以 t（单位切向）为 x 轴的局部系（纵向, 横向左正）。"""
    return torch.stack([(v * t).sum(-1), t[..., 0] * v[..., 1] - t[..., 1] * v[..., 0]], -1)


class PairHead(nn.Module):
    """分解式成对 MLP：A(h_i) + B(h_j) + C(g_ij) → GELU → Linear。"""

    def __init__(self, d: int, g: int, hidden: int, prior: float | None = None):
        super().__init__()
        self.a = nn.Linear(d, hidden)
        self.b = nn.Linear(d, hidden)
        self.c = nn.Linear(g, hidden)
        self.o = MLP(hidden, hidden, 1, 2)
        if prior is not None:                       # 稀疏正例（sigmoid 边头）：偏置初始化为先验概率
            nn.init.constant_(self.o.layers[-1].bias, -math.log((1 - prior) / prior))

    def forward(self, hi, hj, g):
        x = self.a(hi)[:, :, None] + self.b(hj)[:, None] + self.c(g)
        return self.o(F.gelu(x))[..., 0]


class TopologyHead(nn.Module):
    def __init__(self, d: int, h: int, dff: int, cfg: dict):
        super().__init__()
        self.gs = cfg["geo_scale_m"]
        hid = cfg["pair_hidden"]
        self.type_emb = nn.Embedding(3, d)
        self.lane_geo = MLP(11, d, d, 2)
        self.node_geo = MLP(2 + 6, d, d, 2)
        self.cw_geo = MLP(5, d, d, 2)
        self.layers = nn.ModuleList(nn.ModuleList([SelfAttn(d, h), FFN(d, dff)]) for _ in range(cfg["layers"]))
        self.rpe = MLP(3, 32, h * cfg["layers"], 2)
        self.h = h
        self.adj = PairHead(d, 6, hid, prior=0.01)
        self.succ = PairHead(d, 6, hid, prior=0.01)
        self.from_head = PairHead(d, 4, hid)
        self.to_head = PairHead(d, 4, hid)
        self.cross = PairHead(d, 4, hid, prior=0.01)
        self.cross_node = PairHead(d, 3, hid)

    def forward(self, dec: dict) -> dict:
        lp, lw = dec["lane_pts"].detach(), dec["lane_w"].detach()
        npos, ntype = dec["node_pos"].detach(), dec["node_type"].detach().softmax(-1)
        cend, cw = dec["cw_end"].detach(), dec["cw_w"].detach()
        G = lane_geom(lp, lw)
        s = self.gs
        hl = dec["h_lane"] + self.lane_geo(torch.cat([G["start"] / s / 5, G["end"] / s / 5, G["t0"], G["t1"],
                                                       (G["end"] - G["start"]) / s, (lw / 4)[..., None]], -1))
        hn = dec["h_node"] + self.node_geo(torch.cat([npos / s / 5, ntype], -1))
        cc = cend.mean(2)
        ax = cend[:, :, 1] - cend[:, :, 0]
        hc = dec["h_cw"] + self.cw_geo(torch.cat([cc / s / 5, ax / s, (cw / 4)[..., None]], -1))
        B, P, N, M = hn.shape[0], hn.shape[1], hl.shape[1], hc.shape[1]
        x = torch.cat([hn + self.type_emb.weight[0], hl + self.type_emb.weight[1], hc + self.type_emb.weight[2]], 1)
        pos = torch.cat([npos, G["mid"], cc], 1)
        dd = (pos[:, :, None] - pos[:, None]) / s
        bias = self.rpe(torch.cat([dd, dd.norm(dim=-1, keepdim=True)], -1))
        L = len(self.layers)
        bias = bias.view(B, P + N + M, P + N + M, L, self.h).permute(3, 0, 4, 1, 2)
        for k, (att, ffn) in enumerate(self.layers):
            x = ffn(att(x, bias=bias[k]))
        hn, hl, hc = x[:, :P], x[:, P:P + N], x[:, P + N:]
        # ---------------- 成对几何特征
        mid_rel = to_frame(G["mid"][:, None] - G["mid"][:, :, None], G["t0"][:, :, None].expand(-1, -1, N, -1)) / 5
        cos = (G["t0"][:, :, None] * G["t0"][:, None]).sum(-1, keepdim=True)
        sin = (G["t0"][:, :, None, 0] * G["t0"][:, None, :, 1] - G["t0"][:, :, None, 1] * G["t0"][:, None, :, 0])[..., None]
        dw = ((lw[:, None] - lw[:, :, None]) / 4)[..., None]
        ovl = to_frame(G["start"][:, None] - G["start"][:, :, None], G["t0"][:, :, None].expand(-1, -1, N, -1))[..., :1] / s
        g_adj = torch.cat([mid_rel, cos, sin, dw, ovl], -1)
        gap = to_frame(G["start"][:, None] - G["end"][:, :, None], G["t1"][:, :, None].expand(-1, -1, N, -1)) / s
        c2 = (G["t1"][:, :, None] * G["t0"][:, None]).sum(-1, keepdim=True)
        s2 = (G["t1"][:, :, None, 0] * G["t0"][:, None, :, 1] - G["t1"][:, :, None, 1] * G["t0"][:, None, :, 0])[..., None]
        g_succ = torch.cat([gap, c2, s2, gap.norm(dim=-1, keepdim=True), (lw[:, None] / 4).expand(-1, N, -1)[..., None]], -1)
        rel_s = to_frame(npos[:, None] - G["start"][:, :, None], G["t0"][:, :, None].expand(-1, -1, P, -1)) / s
        rel_e = to_frame(npos[:, None] - G["end"][:, :, None], G["t1"][:, :, None].expand(-1, -1, P, -1)) / s
        g_from = torch.cat([rel_s, rel_s.norm(dim=-1, keepdim=True), ntype[:, None, :, :1].expand(-1, N, -1, -1)], -1)
        g_to = torch.cat([rel_e, rel_e.norm(dim=-1, keepdim=True), ntype[:, None, :, :1].expand(-1, N, -1, -1)], -1)
        axn = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rel_c = to_frame(G["mid"][:, None] - cc[:, :, None], axn[:, :, None].expand(-1, -1, N, -1)) / s
        g_cross = torch.cat([rel_c, (G["t0"][:, None] * axn[:, :, None]).sum(-1, keepdim=True),
                             (cw / 4)[:, :, None, None].expand(-1, -1, N, -1)], -1)
        rel_cn = (npos[:, None] - cc[:, :, None]) / s
        g_cn = torch.cat([rel_cn, rel_cn.norm(dim=-1, keepdim=True)], -1)
        out = {"E_adj": self.adj(hl, hl, g_adj), "E_succ": self.succ(hl, hl, g_succ),
               "E_from": self.from_head(hl, hn, g_from), "E_to": self.to_head(hl, hn, g_to),
               "E_cross": self.cross(hc, hl, g_cross), "E_cross_node": self.cross_node(hc, hn, g_cn),
               "h_lane_topo": hl}
        eye = torch.eye(N, dtype=torch.bool, device=hl.device)
        out["E_adj"] = out["E_adj"].masked_fill(eye, -20.0)
        out["E_succ"] = out["E_succ"].masked_fill(eye, -20.0)
        return out


class ConnTrajHead(nn.Module):
    def __init__(self, d: int, h: int, dff: int, cfg: dict, pos_scale: float, step: float):
        super().__init__()
        self.k = cfg["k_conn"]
        self.q = MLP(2 * d + 8, d, d, 2)
        self.pt_emb = nn.Parameter(torch.randn(self.k, d) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(cfg["layers"]):
            self.layers.append(nn.ModuleDict({"sa": SelfAttn(d, h), "norm": nn.LayerNorm(d),
                                              "deform": MSDeformAttn(d, 3, h, cfg["n_points"]), "ffn": FFN(d, dff),
                                              "delta": MLP(d, d, 2, 2, zero_last=True)}))
        self.pos = MLP(d, d, d, 2)
        self.d = d
        self.pos_scale = pos_scale
        self.step = step

    def forward(self, h_lane, lane_pts, pairs, feats: dict, frame: Frame) -> torch.Tensor:
        """h_lane [B, N, D]；lane_pts [B, N, K, 2]（停梯度）；pairs [B, Kp, 2] 车道槽位 → [B, Kp, 8, 2]。"""
        B, Kp, _ = pairs.shape
        lp = lane_pts.detach()
        bi = torch.arange(B, device=pairs.device)[:, None].expand(B, Kp)
        i, j = pairs[..., 0], pairs[..., 1]
        end_i, start_j = lp[bi, i, -1], lp[bi, j, 0]
        ti = lp[bi, i, -1] - lp[bi, i, -2]
        tj = lp[bi, j, 1] - lp[bi, j, 0]
        ti = ti / ti.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tj = tj / tj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rel = to_frame(start_j - end_i, ti) / 30.0
        q = self.q(torch.cat([h_lane[bi, i], h_lane[bi, j], rel, ti, tj, end_i / 500, ], -1))
        t = torch.linspace(0, 1, self.k, device=pairs.device)
        pts = end_i[:, :, None] + t[None, None, :, None] * (start_j - end_i)[:, :, None]
        x = (q[:, :, None] + self.pt_emb[None, None]).reshape(B * Kp, self.k, self.d)
        values = [feats["F4"], feats["F8"], feats["F16"]]
        free = torch.zeros(self.k, 1, device=pairs.device)
        free[1:-1] = 1.0
        for lay in self.layers:
            pe = self.pos(sine_embed(pts, self.d, self.pos_scale)).reshape(B * Kp, self.k, self.d)
            x = lay["sa"](x, pos=pe)
            xq = x.reshape(B, Kp * self.k, self.d)
            ref = frame.to_norm(pts.reshape(B, Kp * self.k, 2)).clamp(-0.5, 1.5).detach()
            xq = xq + lay["deform"](lay["norm"](xq) + pe.reshape(B, Kp * self.k, self.d), ref, values)
            x = lay["ffn"](xq).reshape(B * Kp, self.k, self.d)
            pts = pts.detach() + self.step * lay["delta"](x).reshape(B, Kp, self.k, 2) * free
        return pts
