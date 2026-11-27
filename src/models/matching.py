"""训练期 V / L / C 独立 Hungarian 集合匹配（docs/DESIGN.md §4.2 阶段 C）。

有向折线固定行驶方向（车道点序直接对应，不做翻转）；人行横道中心轴端点无向（取两种顺序的较小代价）。
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _focal_cls_cost(logit: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    p = logit.sigmoid()
    neg = (1 - alpha) * p ** gamma * (-(1 - p + 1e-8).log())
    pos = alpha * (1 - p) ** gamma * (-(p + 1e-8).log())
    return pos - neg


@torch.no_grad()
def match_layer(o: dict, tgt: dict, b: int, cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mc = cfg["match"]
    res = {}
    # ---------------- 车道
    gl = torch.nonzero(tgt["lane_valid"][b]).flatten()
    if len(gl):
        P = o["lane_pts"][b].float()
        G = tgt["lane_pts"][b, gl].float()
        c = mc["lane"]["cls"] * _focal_cls_cost(o["lane_score"][b].float())[:, None] \
            + mc["lane"]["pts"] * torch.cdist(P.flatten(1), G.flatten(1), p=1) / P.shape[1] / 2 \
            + mc["lane"]["width"] * (o["lane_w"][b].float()[:, None] - tgt["lane_w"][b, gl].float()[None]).abs()
        r, cidx = linear_sum_assignment(c.cpu().numpy())
        res["lane"] = (r, gl.cpu().numpy()[cidx])
    else:
        res["lane"] = (np.zeros(0, int), np.zeros(0, int))
    # ---------------- 节点
    gn = torch.nonzero(tgt["node_valid"][b]).flatten()
    if len(gn):
        prob = o["node_type"][b].float().softmax(-1)
        c = mc["node"]["cls"] * _focal_cls_cost(o["node_score"][b].float())[:, None] \
            - mc["node"]["type"] * prob[:, tgt["node_type"][b, gn]] \
            + mc["node"]["pos"] * torch.cdist(o["node_pos"][b].float(), tgt["node_pos"][b, gn].float(), p=1) / 2
        r, cidx = linear_sum_assignment(c.cpu().numpy())
        res["node"] = (r, gn.cpu().numpy()[cidx])
    else:
        res["node"] = (np.zeros(0, int), np.zeros(0, int))
    # ---------------- 人行横道
    gc = torch.nonzero(tgt["cw_valid"][b]).flatten()
    if len(gc):
        P = o["cw_end"][b].float()
        G = tgt["cw_end"][b, gc].float()
        d1 = torch.cdist(P.flatten(1), G.flatten(1), p=1)
        d2 = torch.cdist(P.flatten(1), G.flip(1).flatten(1), p=1)
        c = mc["crosswalk"]["cls"] * _focal_cls_cost(o["cw_score"][b].float())[:, None] \
            + mc["crosswalk"]["end"] * torch.minimum(d1, d2) / 4
        r, cidx = linear_sum_assignment(c.cpu().numpy())
        res["cw"] = (r, gc.cpu().numpy()[cidx])
    else:
        res["cw"] = (np.zeros(0, int), np.zeros(0, int))
    return res


def match_all(out: dict, tgt: dict, cfg: dict) -> list[list[dict]]:
    """返回 matches[layer][b]（末层下标为 -1）。"""
    B = tgt["lane_valid"].shape[0]
    return [[match_layer(o, tgt, b, cfg) for b in range(B)] for o in out["layers"]]


def conn_pairs(match_last: list[dict], tgt: dict, max_pairs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """真值正连接对 → 预测车道槽位对（训练期成对轨迹头输入）。返回 (pairs [B, K, 2], gt_idx [B, K], mask [B, K])。"""
    B = tgt["conn_valid"].shape[0]
    pairs = torch.zeros(B, max_pairs, 2, dtype=torch.long)
    gidx = torch.zeros(B, max_pairs, dtype=torch.long)
    mask = torch.zeros(B, max_pairs, dtype=torch.bool)
    for b in range(B):
        pr, gt = match_last[b]["lane"]
        g2p = {int(g): int(p) for p, g in zip(pr, gt)}
        cv = torch.nonzero(tgt["conn_valid"][b]).flatten().tolist()
        k = 0
        for ci in cv:
            i, j = (int(x) for x in tgt["conn_ij"][b, ci])
            if i in g2p and j in g2p and k < max_pairs:
                pairs[b, k] = torch.tensor([g2p[i], g2p[j]])
                gidx[b, k] = ci
                mask[b, k] = True
                k += 1
    return pairs, gidx, mask
