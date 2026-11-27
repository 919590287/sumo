"""阶段 A–C（docs/DESIGN.md §4.2）：稠密辅助感知头 → 可学习 Top-K 分类别 Query 提议 →
6 层解耦层级迭代矢量解码器（MapTRv2 Intra/Inter-Instance 自注意力 + 多尺度可变形交叉注意力 +
稀疏矢量交叉注意力 + FFN + 几何迭代更新）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.targets import DENSE_CHANNELS
from src.graph.lane_graph import LANE_ROLES, NODE_TYPES, PERMISSIONS, POLY_K
from src.models.common import FFN, MLP, CrossAttn, Frame, SelfAttn, gather_points, sine_embed
from src.models.sampling import MSDeformAttn

DCH = {n: k for k, n in enumerate(DENSE_CHANNELS)}
NODE_ANCHORS = ("a_junction", "a_split", "a_midblock", "a_port", "a_attr")
# 节点锚点通道 → 初始类型（端口按入 / 出不可由锚点区分，类型由解码器学习）
ANCHOR_TYPE = {"a_junction": "junction_core", "a_split": "split_merge", "a_midblock": "midblock_node",
               "a_port": "roi_entry_port", "a_attr": "road_continuation"}


class DenseHead(nn.Module):
    """挂接 F_2 与 F_4'：有效车道掩码、中心线热图 / SDF、方向场、宽度场与 7 类锚点热图（0.40 m/px）。"""

    def __init__(self, d: int, d2: int, hidden: int):
        super().__init__()
        self.up = nn.Conv2d(d, d2, 1)
        self.body = nn.Sequential(
            nn.Conv2d(2 * d2, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.out = nn.Conv2d(hidden, len(DENSE_CHANNELS), 1)
        prior = -math.log((1 - 0.01) / 0.01)
        with torch.no_grad():
            self.out.bias.zero_()
            for k in ("valid", "center_heat") + tuple(n for n in DENSE_CHANNELS if n.startswith("a_")):
                self.out.bias[DCH[k]] = prior

    def forward(self, f2: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        u = F.interpolate(self.up(f4), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.body(torch.cat([f2, u], 1)))


def dense_activations(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """原始稠密输出 → 物理量：概率 / [-1,1] / 单位向量 / 宽度（米/4）。"""
    o = {}
    for k in DENSE_CHANNELS:
        x = raw[:, DCH[k]]
        if k in ("center_sdf",):
            o[k] = torch.tanh(x)
        elif k in ("dir_cos", "dir_sin"):
            o[k] = x
        elif k == "width":
            o[k] = F.softplus(x)
        else:
            o[k] = torch.sigmoid(x)
    n = torch.sqrt(o["dir_cos"] ** 2 + o["dir_sin"] ** 2 + 1e-6)
    o["dir_cos"], o["dir_sin"] = o["dir_cos"] / n, o["dir_sin"] / n
    return o


def topk_peaks(score: torch.Tensor, valid: torch.Tensor, k: int, kernel: int):
    """score [B, H, W] → 极大值抑制后 Top-K：返回 (行, 列, 分数) 各 [B, k]。"""
    s = score.masked_fill(~valid, -1.0)
    m = F.max_pool2d(s[:, None], kernel, 1, kernel // 2)[:, 0]
    peak = torch.where(s >= m, s, torch.full_like(s, -1.0))
    B, H, W = s.shape
    v, idx = peak.flatten(1).topk(k, dim=1)
    return idx // W, idx % W, v


class QueryProposer(nn.Module):
    """可学习 Top-K 评分器（即稠密头的中心线 / 锚点热图）→ 三类异构 Query 初值。"""

    def __init__(self, d: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.p_q, self.n_q, self.m_q, self.k = cfg["p_q"], cfg["n_q"], cfg["m_q"], cfg["k_lane"]
        self.feat = nn.ModuleDict({k: MLP(d + 8, d, d, 2) for k in ("node", "lane", "cw")})
        self.slot = nn.ParameterDict({
            "node": nn.Parameter(torch.randn(self.p_q, d) * 0.02),
            "lane": nn.Parameter(torch.randn(self.n_q, d) * 0.02),
            "cw": nn.Parameter(torch.randn(self.m_q, d) * 0.02)})
        self.point_emb = nn.Parameter(torch.randn(self.k, d) * 0.02)
        self.lane_len = MLP(d, d // 2, 1, 2, zero_last=True)
        self.poly_r = MLP(d, d // 2, 1, 2, zero_last=True)
        self.cw_len = MLP(d, d // 2, 1, 2, zero_last=True)
        self.type_bias = nn.Parameter(torch.zeros(len(NODE_ANCHORS), len(NODE_TYPES)))
        with torch.no_grad():
            for k, a in enumerate(NODE_ANCHORS):
                self.type_bias[k, NODE_TYPES.index(ANCHOR_TYPE[a])] = 2.0

    def forward(self, f4: torch.Tensor, dense: dict, frame: Frame, valid_px: torch.Tensor):
        """valid_px [B, H2, W2]：真实窗口范围（排除补零区）。"""
        B = f4.shape[0]
        dev = f4.device
        stride = 2
        bidx = lambda k: torch.arange(B, device=dev)[:, None].expand(B, k)  # noqa: E731
        feats_ctx = torch.stack([dense[c] for c in ("valid", "center_heat", "center_sdf", "dir_cos", "dir_sin",
                                                    "width", "a_stop", "a_crosswalk")], 1)

        def content(kind, xy, extra):
            uv = frame.to_norm(xy)
            f = gather_points(f4, uv)
            ctx = gather_points(feats_ctx, uv)
            return self.feat[kind](torch.cat([f, ctx], -1)) + self.slot[kind][None] + extra

        # ---------------- 车道
        lane_score = dense["center_heat"] * dense["valid"]
        r, c, _ = topk_peaks(lane_score, valid_px, self.n_q, self.cfg["nms_kernel"]["lane"])
        p = frame.from_pixel(c, r, stride, bidx(self.n_q))
        d = torch.stack([dense["dir_cos"][bidx(self.n_q), r, c], dense["dir_sin"][bidx(self.n_q), r, c]], -1)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        w0 = (dense["width"][bidx(self.n_q), r, c] * 4.0).clamp(2.0, 4.5)
        lq = content("lane", p, 0.0)
        L = self.cfg["init_lane_len_m"] * torch.exp(self.lane_len(lq)[..., 0].clamp(-2, 2))
        t = torch.linspace(-0.5, 0.5, self.k, device=dev)
        lane_pts = p[:, :, None] + t[None, None, :, None] * L[..., None, None] * d[:, :, None]
        lane_q = lq[:, :, None] + self.point_emb[None, None]
        # ---------------- 节点
        node_maps = torch.stack([dense[a] for a in NODE_ANCHORS], 1)                  # [B, 5, H, W]
        nscore, ntype = node_maps.max(1)
        r, c, _ = topk_peaks(nscore, valid_px, self.p_q, self.cfg["nms_kernel"]["node"])
        npos = frame.from_pixel(c, r, stride, bidx(self.p_q))
        tsel = ntype[bidx(self.p_q), r, c]
        nq = content("node", npos, 0.0)
        type_init = self.type_bias[tsel]
        R = self.cfg["init_poly_radius_m"] * torch.exp(self.poly_r(nq)[..., 0].clamp(-2, 2))
        ang = torch.linspace(0, 2 * math.pi, POLY_K + 1, device=dev)[:-1] - math.pi
        circ = torch.stack([ang.cos(), ang.sin()], -1)
        poly = npos[:, :, None] + R[..., None, None] * circ[None, None]
        # ---------------- 人行横道
        r, c, _ = topk_peaks(dense["a_crosswalk"], valid_px, self.m_q, self.cfg["nms_kernel"]["cw"])
        cpos = frame.from_pixel(c, r, stride, bidx(self.m_q))
        cd = torch.stack([dense["dir_cos"][bidx(self.m_q), r, c], dense["dir_sin"][bidx(self.m_q), r, c]], -1)
        cd = cd / cd.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        axis = torch.stack([-cd[..., 1], cd[..., 0]], -1)
        cq = content("cw", cpos, 0.0)
        hl = self.cfg["init_cw_half_len_m"] * torch.exp(self.cw_len(cq)[..., 0].clamp(-2, 2))
        cw_end = torch.stack([cpos - axis * hl[..., None], cpos + axis * hl[..., None]], 2)
        return {"lane_q": lane_q, "lane_pts": lane_pts, "lane_w": w0, "node_q": nq, "node_pos": npos,
                "node_poly": poly, "node_type_init": type_init, "cw_q": cq, "cw_end": cw_end,
                "cw_w": torch.full_like(hl, 4.0)}


class DecoderLayer(nn.Module):
    def __init__(self, d: int, h: int, dff: int, n_points: int):
        super().__init__()
        self.intra = SelfAttn(d, h)
        self.inter = SelfAttn(d, h)
        self.deform_norm = nn.LayerNorm(d)
        self.deform = MSDeformAttn(d, 3, h, n_points)
        self.vec = CrossAttn(d, h)
        self.ffn = FFN(d, dff)

    def forward(self, lane_q, node_q, cw_q, lane_pe, node_pe, cw_pe, refs, values, vec, vec_pos, vec_mask):
        B, N, K, D = lane_q.shape
        P, M = node_q.shape[1], cw_q.shape[1]
        # 1. 实例内点级自注意力
        x = self.intra(lane_q.reshape(B * N, K, D), pos=lane_pe.reshape(B * N, K, D)).reshape(B, N, K, D)
        # 2. 跨实体实例级自注意力（节点 + 车道池化 + 人行横道）
        inst = x.mean(2)
        tok = torch.cat([node_q, inst, cw_q], 1)
        tpe = torch.cat([node_pe, lane_pe.mean(2), cw_pe], 1)
        tok2 = self.inter(tok, pos=tpe)
        node_q, inst2, cw_q = tok2[:, :P], tok2[:, P:P + N], tok2[:, P + N:]
        x = x + (inst2 - inst)[:, :, None]
        # 3. 多尺度可变形交叉注意力
        q = torch.cat([x.reshape(B, N * K, D), node_q, cw_q], 1)
        pe = torch.cat([lane_pe.reshape(B, N * K, D), node_pe, cw_pe], 1)
        q = q + self.deform(self.deform_norm(q) + pe, refs, values)
        # 4. 稀疏矢量交叉注意力（实体级：车道点特征池化为实例 Token 后与节点 / 人行横道共 216 个查询，
        #    增量广播回各点；显存较逐点查询降低约 20×）
        lane = q[:, :N * K].reshape(B, N, K, D)
        inst = lane.mean(2)
        ent = torch.cat([q[:, N * K:N * K + P], inst, q[:, N * K + P:]], 1)
        epe = torch.cat([node_pe, lane_pe.mean(2), cw_pe], 1)
        ent2 = self.vec(ent, vec, qpos=epe, kpos=vec_pos, key_mask=vec_mask)
        lane = lane + (ent2[:, P:P + N] - inst)[:, :, None]
        q = torch.cat([lane.reshape(B, N * K, D), ent2[:, :P], ent2[:, P + N:]], 1)
        # 5. FFN
        q = self.ffn(q)
        return q[:, :N * K].reshape(B, N, K, D), q[:, N * K:N * K + P], q[:, N * K + P:]


class MapDecoder(nn.Module):
    def __init__(self, d: int, h: int, dff: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.d = d
        self.layers = nn.ModuleList(DecoderLayer(d, h, dff, cfg["n_points"]) for _ in range(cfg["layers"]))
        self.pos_mlp = MLP(d, d, d, 2)
        L = cfg["layers"]
        mk = lambda dout, zero=True: nn.ModuleList(MLP(d, d, dout, 2, zero_last=zero) for _ in range(L))  # noqa: E731
        self.lane_dpts = mk(2)
        self.lane_dw = mk(1)
        self.node_dpos = mk(2)
        self.node_dpoly = mk(POLY_K * 2)
        self.cw_dend = mk(4)
        self.cw_dw = mk(1)
        self.lane_score = mk(1, False)
        self.node_score = mk(1, False)
        self.cw_score = mk(1, False)
        self.node_type = mk(len(NODE_TYPES), False)
        self.lane_role = nn.Linear(d, len(LANE_ROLES))
        self.lane_perm = nn.Linear(d, len(PERMISSIONS))
        self.lane_lc = nn.Linear(d, 2)
        self.lane_stop = nn.Linear(d, 1)
        self.port_embed = MLP(d, d, 256, 2)
        prior = -math.log((1 - 0.01) / 0.01)
        for heads in (self.lane_score, self.node_score, self.cw_score):
            for m in heads:
                nn.init.constant_(m.layers[-1].bias, prior)

    def pe(self, xy):
        return self.pos_mlp(sine_embed(xy, self.d, self.cfg["pos_scale_m"]))

    def forward(self, init: dict, feats: dict, frame: Frame, vec, vec_ctr, vec_mask, vec_pos):
        step = self.cfg["step_scale_m"]
        lane_q, node_q, cw_q = init["lane_q"], init["node_q"], init["cw_q"]
        lane_pts, lane_w = init["lane_pts"], init["lane_w"]
        node_pos, node_poly = init["node_pos"], init["node_poly"]
        cw_end, cw_w = init["cw_end"], init["cw_w"]
        values = [feats["F4"], feats["F8"], feats["F16"]]
        B, N, K, _ = lane_q.shape
        outs = []
        for li, layer in enumerate(self.layers):
            lane_pe, node_pe = self.pe(lane_pts), self.pe(node_pos)
            cw_ctr = cw_end.mean(2)
            cw_pe = self.pe(cw_ctr)
            refs = frame.to_norm(torch.cat([lane_pts.reshape(B, N * K, 2), node_pos, cw_ctr], 1)).clamp(-0.5, 1.5)
            lane_q, node_q, cw_q = layer(lane_q, node_q, cw_q, lane_pe, node_pe, cw_pe, refs.detach(), values,
                                         vec, vec_pos, vec_mask)
            inst = lane_q.mean(2)
            lane_pts = lane_pts.detach() + step * self.lane_dpts[li](lane_q)
            lane_w = (lane_w.detach() + 0.5 * self.lane_dw[li](inst)[..., 0]).clamp(0.5, 8.0)
            node_pos = node_pos.detach() + step * self.node_dpos[li](node_q)
            node_poly = node_poly.detach() + step * self.node_dpoly[li](node_q).view(B, -1, POLY_K, 2) \
                + (node_pos - node_pos.detach())[:, :, None]
            cw_end = cw_end.detach() + step * self.cw_dend[li](cw_q).view(B, -1, 2, 2)
            cw_w = (cw_w.detach() + 0.5 * self.cw_dw[li](cw_q)[..., 0]).clamp(0.5, 12.0)
            o = {"lane_pts": lane_pts, "lane_w": lane_w, "lane_score": self.lane_score[li](inst)[..., 0],
                 "node_pos": node_pos, "node_poly": node_poly, "node_score": self.node_score[li](node_q)[..., 0],
                 "node_type": self.node_type[li](node_q) + init["node_type_init"],
                 "cw_end": cw_end, "cw_w": cw_w, "cw_score": self.cw_score[li](cw_q)[..., 0]}
            outs.append(o)
        last = outs[-1]
        inst = lane_q.mean(2)
        last.update({"lane_role": self.lane_role(inst), "lane_perm": self.lane_perm(inst),
                     "lane_lc": self.lane_lc(inst), "lane_stop": self.lane_stop(inst)[..., 0],
                     "port_embed": F.normalize(self.port_embed(node_q), dim=-1),
                     "h_lane": inst, "h_lane_pts": lane_q, "h_node": node_q, "h_cw": cw_q})
        return outs
