"""Baidu-to-SUMO 完整网络（docs/DESIGN.md §2–§4）：

BEV 编码器（高保真 Stem + ResNet34 阶段 + 4 层 FPN）‖ 矢量编码器 → 2 层双向跨模态融合 →
稠密辅助感知头 → Top-K 分类别 Query 提议 → 6 层解耦层级迭代解码器 → GNN 拓扑头；
成对连接轨迹头按需对指定车道对调用（训练期：真值正对；推理期：约束解码激活对）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.config import load_config
from src.graph.lane_graph import (CONN_MAX, LANE_ROLES, M_MAX, N_MAX, NODE_TYPES, P_MAX, PERMISSIONS, POLY_K,
                                  GraphTensors)
from src.models.bev_encoder import BevEncoder, count_params
from src.models.common import MLP, Frame, sine_embed
from src.models.decoder import DenseHead, MapDecoder, QueryProposer, dense_activations
from src.models.fusion import FusionLayer, bev_positions
from src.models.topology import ConnTrajHead, TopologyHead
from src.models.vector_encoder import VectorEncoder

STRIDES = {"F2": 2, "F4": 4, "F8": 8, "F16": 16}


class B2SNet(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or load_config("model")
        self.cfg = cfg
        d, h, dff = cfg["d_model"], cfg["n_heads"], cfg["ffn"]
        dc = cfg["decoder"]
        assert dc["p_q"] == P_MAX and dc["n_q"] == N_MAX and dc["m_q"] == M_MAX, "Query 数须与图契约单窗上限一致"
        self.res = load_config("data")["resolution_m"]
        self.pos_scale = dc["pos_scale_m"]
        self.bev = BevEncoder(cfg["in_channels"], d, cfg["d2"])
        self.vec = VectorEncoder(d, h, dff, cfg["vector_encoder"], self.pos_scale)
        fc = cfg["fusion"]
        self.fusion = nn.ModuleList(FusionLayer(d, h, dff, fc["n_points"], fc["v2b_levels"], self.pos_scale)
                                    for _ in range(fc["layers"]))
        self.f2_fuse = nn.Conv2d(d, cfg["d2"], 1)
        self.dense = DenseHead(d, cfg["d2"], cfg["dense_head"]["hidden"])
        self.proposer = QueryProposer(d, dc)
        self.decoder = MapDecoder(d, h, dff, dc)
        self.vec_pos = MLP(d, d, d, 2)
        self.topo = TopologyHead(d, h, dff, cfg["topology"])
        self.conn = ConnTrajHead(d, h, dff, cfg["conn_head"], self.pos_scale, dc["step_scale_m"])

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        x = batch["bev"].float()
        hw = batch["hw"]
        B, _, Hc, Wc = x.shape
        frame = Frame(hw, (Hc, Wc), self.res)
        feats = self.bev(x)
        tok = {k: batch[k] for k in ("tok_cls", "tok_pts", "tok_tan", "tok_size", "tok_heading", "tok_turns",
                                     "tok_mask")}
        tok = {k: (v.float() if v.is_floating_point() else v) for k, v in tok.items()}
        vec, vctr, vmask = self.vec(tok)
        d = vec.shape[-1]
        bpos = bev_positions(feats, frame, {k: STRIDES[k] for k in self.cfg["fusion"]["v2b_levels"]}, d,
                             self.pos_scale)
        for layer in self.fusion:
            feats, vec = layer(feats, vec, vctr, vmask, frame, bpos)
        feats["F2"] = feats["F2"] + nn.functional.interpolate(self.f2_fuse(feats["F4"]), size=feats["F2"].shape[-2:],
                                                               mode="bilinear", align_corners=False)
        dense_raw = self.dense(feats["F2"], feats["F4"])
        dense = {k: v.detach() for k, v in dense_activations(dense_raw).items()}
        h2, w2 = dense_raw.shape[-2:]
        rr = torch.arange(h2, device=x.device)[None, :, None]
        cc = torch.arange(w2, device=x.device)[None, None, :]
        valid_px = (rr < (hw[:, 0] // 2)[:, None, None]) & (cc < (hw[:, 1] // 2)[:, None, None])
        init = self.proposer(feats["F4"], dense, frame, valid_px)
        vpos = self.vec_pos(sine_embed(vctr, d, self.pos_scale))
        layers = self.decoder(init, feats, frame, vec, vctr, vmask, vpos)
        topo = self.topo(layers[-1])
        out = {"dense_raw": dense_raw, "layers": layers, "init": init, **topo, **{k: v for k, v in layers[-1].items()}}
        out["_cache"] = {"feats": {k: feats[k] for k in ("F4", "F8", "F16")}, "frame": frame}
        return out

    def connections(self, out: dict, pairs: torch.Tensor) -> torch.Tensor:
        """pairs [B, Kp, 2]（车道槽位）→ K_conn [B, Kp, 8, 2]（米）。"""
        c = out["_cache"]
        return self.conn(out["h_lane_topo"], out["lane_pts"], pairs, c["feats"], c["frame"])


def build_model(cfg: dict | None = None) -> B2SNet:
    return B2SNet(cfg)


# ====================================================================== 网络输出 → GraphTensors（约束解码输入）
@torch.no_grad()
def to_graph_tensors(out: dict, b: int, origin=(0.0, 0.0)) -> GraphTensors:
    t = GraphTensors.empty()
    sig = torch.sigmoid
    f = lambda x: x[b].detach().float().cpu().numpy()  # noqa: E731
    t.node_score[:] = f(sig(out["node_score"]))
    t.node_type[:] = f(out["node_type"].softmax(-1))
    t.node_pos[:, :2] = f(out["node_pos"])
    t.node_poly[:] = f(out["node_poly"])
    t.port_embed[:] = f(out["port_embed"])
    t.lane_pts[:, :, :2] = f(out["lane_pts"])
    t.lane_width[:] = f(out["lane_w"])
    t.lane_role[:] = f(out["lane_role"].softmax(-1))
    t.lane_perm[:] = f(out["lane_perm"].softmax(-1))
    t.lane_lc[:] = f(sig(out["lane_lc"]))
    t.lane_stop[:] = f(sig(out["lane_stop"]))
    t.lane_score[:] = f(sig(out["lane_score"]))
    t.E_adj[:] = f(sig(out["E_adj"]))
    t.E_succ[:] = f(sig(out["E_succ"]))
    t.E_from[:] = f(out["E_from"].softmax(-1))
    t.E_to[:] = f(out["E_to"].softmax(-1))
    t.cw_endpoints[:, :, :2] = f(out["cw_end"])
    t.cw_width[:] = f(out["cw_w"])
    t.cw_score[:] = f(sig(out["cw_score"]))
    t.E_cross[:] = f(sig(out["E_cross"]))
    t.E_cross_node[:] = f(out["E_cross_node"].softmax(-1))
    t.origin = tuple(origin)
    return t


__all__ = ["B2SNet", "build_model", "to_graph_tensors", "count_params", "CONN_MAX", "LANE_ROLES", "NODE_TYPES",
           "PERMISSIONS", "POLY_K"]
