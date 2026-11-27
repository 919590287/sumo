"""Layer 1 预训练多任务损失 L_sim（docs/DESIGN.md §5 Layer 1-4）：

L_sim = L_dense + λ_V (L_node + L_poly_end) + λ_L (L_pts + L_dir + L_width + L_cls/perm/lc)
        + λ_K L_conn_traj + λ_C L_cw + λ_topo L_topo(E_adj, E_from, E_to, E_succ, E_cross, E_cross_node)
        + λ_port L_port_match

全部权重见 configs/losses.yaml。中间解码层按 ``aux_layer_weight`` 计辅助几何 / 存在性损失。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.common.config import load_config
from src.data.targets import DENSE_CHANNELS
from src.graph.lane_graph import NODE_TYPES

DCH = {n: k for k, n in enumerate(DENSE_CHANNELS)}
J_CORE = NODE_TYPES.index("junction_core")


def sigmoid_focal(logit, target, alpha=0.25, gamma=2.0, reduction="sum"):
    p = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma
    loss = w * ce
    return loss.sum() if reduction == "sum" else loss


def heat_focal(logit, target, mask, a=2.0, b=4.0):
    """CenterNet 式惩罚降低焦点损失（高斯软目标；峰值邻域为正样本）。"""
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.9).float() * mask
    neg = (1 - pos) * mask
    lp = -((1 - p) ** a) * p.log() * pos
    ln = -((1 - target) ** b) * (p ** a) * (1 - p).log() * neg
    return (lp.sum() + ln.sum()) / pos.sum().clamp_min(1.0)


def point_to_ring_distance(q: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """q [Q, 2] 到闭合折线 ring [K, 2] 的最近距离 [Q]（可微）。"""
    a = ring
    b = torch.roll(ring, -1, 0)
    ab = b - a
    t = ((q[:, None] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)[None]
    proj = a[None] + t.clamp(0, 1)[..., None] * ab[None]
    return (q[:, None] - proj).norm(dim=-1).min(1).values


class SimCriterion:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config("losses")

    # ------------------------------------------------------------------
    def dense(self, raw: torch.Tensor, tgt: torch.Tensor, hw: torch.Tensor) -> dict:
        c = self.cfg["dense"]
        t = tgt.float()
        B, _, H, W = raw.shape
        rr = torch.arange(H, device=raw.device)[None, :, None]
        cc = torch.arange(W, device=raw.device)[None, None, :]
        m = ((rr < (hw[:, 0] // 2)[:, None, None]) & (cc < (hw[:, 1] // 2)[:, None, None])).float()
        out = {}
        bce = F.binary_cross_entropy_with_logits(raw[:, DCH["valid"]], t[:, DCH["valid"]], reduction="none")
        out["d_valid"] = c["w_valid"] * (bce * m).sum() / m.sum()
        out["d_center"] = c["w_center"] * heat_focal(raw[:, DCH["center_heat"]], t[:, DCH["center_heat"]], m,
                                                     c["focal_alpha"], c["focal_beta"])
        an = [k for k in DENSE_CHANNELS if k.startswith("a_")]
        out["d_anchor"] = c["w_anchor"] * sum(heat_focal(raw[:, DCH[k]], t[:, DCH[k]], m, c["focal_alpha"],
                                                         c["focal_beta"]) for k in an) / len(an)
        v = (t[:, DCH["valid"]] > 0.5).float() * m
        nv = v.sum().clamp_min(1.0)
        out["d_sdf"] = c["w_sdf"] * ((torch.tanh(raw[:, DCH["center_sdf"]]) - t[:, DCH["center_sdf"]]).abs() * v).sum() / nv
        dv = torch.stack([raw[:, DCH["dir_cos"]], raw[:, DCH["dir_sin"]]], 1)
        dv = dv / dv.norm(dim=1, keepdim=True).clamp_min(1e-4)
        dt = torch.stack([t[:, DCH["dir_cos"]], t[:, DCH["dir_sin"]]], 1)
        out["d_dir"] = c["w_dir"] * ((1 - (dv * dt).sum(1)) * v).sum() / nv
        out["d_width"] = c["w_width"] * ((F.softplus(raw[:, DCH["width"]]) - t[:, DCH["width"]]).abs() * v).sum() / nv
        return out

    # ------------------------------------------------------------------
    def layer(self, o: dict, tgt: dict, matches: list[dict], last: bool) -> dict:
        cl, cn, cc = self.cfg["lane"], self.cfg["node"], self.cfg["crosswalk"]
        fa, fg = self.cfg["score_focal"]["alpha"], self.cfg["score_focal"]["gamma"]
        dev = o["lane_pts"].device
        B = o["lane_pts"].shape[0]
        L = {k: torch.zeros((), device=dev) for k in
             ("l_score", "l_pts", "l_dir", "l_width", "n_score", "n_type", "n_pos", "n_poly", "c_score", "c_end",
              "c_width")}
        if last:
            for k in ("l_role", "l_perm", "l_lc", "l_stop", "n_poly_end"):
                L[k] = torch.zeros((), device=dev)
        n_l = n_n = n_c = 0
        for b in range(B):
            m = matches[b]
            # ---------------- 车道
            pi, gi = (torch.as_tensor(x, device=dev, dtype=torch.long) for x in m["lane"])
            tgt_s = torch.zeros_like(o["lane_score"][b])
            tgt_s[pi] = 1.0
            L["l_score"] = L["l_score"] + sigmoid_focal(o["lane_score"][b], tgt_s, fa, fg)
            n_l += len(pi)
            if len(pi):
                P = o["lane_pts"][b, pi]
                G = tgt["lane_pts"][b, gi]
                L["l_pts"] = L["l_pts"] + (P - G).abs().sum(-1).mean(-1).sum()
                dp = P[:, 1:] - P[:, :-1]
                dg = G[:, 1:] - G[:, :-1]
                cos = (dp * dg).sum(-1) / (dp.norm(dim=-1) * dg.norm(dim=-1)).clamp_min(1e-6)
                L["l_dir"] = L["l_dir"] + (1 - cos).mean(-1).sum()
                L["l_width"] = L["l_width"] + (o["lane_w"][b, pi] - tgt["lane_w"][b, gi]).abs().sum()
                if last:
                    L["l_role"] = L["l_role"] + F.cross_entropy(o["lane_role"][b, pi], tgt["lane_role"][b, gi], reduction="sum")
                    L["l_perm"] = L["l_perm"] + F.cross_entropy(o["lane_perm"][b, pi], tgt["lane_perm"][b, gi], reduction="sum")
                    L["l_lc"] = L["l_lc"] + F.binary_cross_entropy_with_logits(
                        o["lane_lc"][b, pi], tgt["lane_lc"][b, gi], reduction="sum") / 2
                    L["l_stop"] = L["l_stop"] + F.binary_cross_entropy_with_logits(
                        o["lane_stop"][b, pi], tgt["lane_stop"][b, gi], reduction="sum")
            # ---------------- 节点
            pn, gn = (torch.as_tensor(x, device=dev, dtype=torch.long) for x in m["node"])
            tgt_s = torch.zeros_like(o["node_score"][b])
            tgt_s[pn] = 1.0
            L["n_score"] = L["n_score"] + sigmoid_focal(o["node_score"][b], tgt_s, fa, fg)
            n_n += len(pn)
            if len(pn):
                L["n_type"] = L["n_type"] + F.cross_entropy(o["node_type"][b, pn], tgt["node_type"][b, gn], reduction="sum")
                L["n_pos"] = L["n_pos"] + (o["node_pos"][b, pn] - tgt["node_pos"][b, gn]).abs().sum()
                jc = tgt["node_type"][b, gn] == J_CORE
                if jc.any():
                    Pp = o["node_poly"][b, pn[jc]]                                 # [J, K, 2]
                    Gp = tgt["node_poly"][b, gn[jc]]
                    K = Pp.shape[1]
                    shifts = torch.stack([torch.roll(Gp, -s, 1) for s in range(K)], 1)   # [J, K, K, 2]
                    d = (Pp[:, None] - shifts).abs().sum(-1).mean(-1)               # [J, K]
                    L["n_poly"] = L["n_poly"] + d.min(1).values.sum()
                    if last:
                        L["n_poly_end"] = L["n_poly_end"] + self._poly_end(o, tgt, b, pn[jc], gn[jc], matches[b])
            # ---------------- 人行横道
            pc, gc = (torch.as_tensor(x, device=dev, dtype=torch.long) for x in m["cw"])
            tgt_s = torch.zeros_like(o["cw_score"][b])
            tgt_s[pc] = 1.0
            L["c_score"] = L["c_score"] + sigmoid_focal(o["cw_score"][b], tgt_s, fa, fg)
            n_c += len(pc)
            if len(pc):
                P = o["cw_end"][b, pc]
                G = tgt["cw_end"][b, gc]
                d1 = (P - G).abs().sum((-1, -2))
                d2 = (P - G.flip(1)).abs().sum((-1, -2))
                L["c_end"] = L["c_end"] + torch.minimum(d1, d2).sum() / 2
                L["c_width"] = L["c_width"] + (o["cw_w"][b, pc] - tgt["cw_w"][b, gc]).abs().sum()
        nl, nn_, nc = max(n_l, 1), max(n_n, 1), max(n_c, 1)
        out = {
            "l_score": cl["w_score"] * L["l_score"] / nl, "l_pts": cl["w_pts"] * L["l_pts"] / nl,
            "l_dir": cl["w_dir"] * L["l_dir"] / nl, "l_width": cl["w_width"] * L["l_width"] / nl,
            "n_score": cn["w_score"] * L["n_score"] / nn_, "n_type": cn["w_type"] * L["n_type"] / nn_,
            "n_pos": cn["w_pos"] * L["n_pos"] / nn_, "n_poly": cn["w_poly"] * L["n_poly"] / nn_,
            "c_score": cc["w_score"] * L["c_score"] / nc, "c_end": cc["w_end"] * L["c_end"] / nc,
            "c_width": cc["w_width"] * L["c_width"] / nc,
        }
        if last:
            out.update({"l_role": cl["w_role"] * L["l_role"] / nl, "l_perm": cl["w_perm"] * L["l_perm"] / nl,
                        "l_lc": cl["w_lc"] * L["l_lc"] / nl, "l_stop": cl["w_stop"] * L["l_stop"] / nl,
                        "n_poly_end": cn["w_poly_end"] * L["n_poly_end"] / nn_})
        return out

    def _poly_end(self, o, tgt, b, pn_j, gn_j, m) -> torch.Tensor:
        """L_poly_end：junction_core 轮廓边界经过其接入（预测、停梯度）车道的起止横断面角点。"""
        pl, gl = m["lane"]
        g2p = {int(g): int(p) for p, g in zip(pl, gl)}
        lp = o["lane_pts"][b].detach()
        lw = o["lane_w"][b].detach()
        total = torch.zeros((), device=lp.device)
        for pn, gn in zip(pn_j.tolist(), gn_j.tolist()):
            corners = []
            for g in torch.nonzero(tgt["lane_valid"][b]).flatten().tolist():
                if g not in g2p:
                    continue
                p = g2p[g]
                for at_end, node in ((True, tgt["lane_to"][b, g]), (False, tgt["lane_from"][b, g])):
                    if int(node) != gn:
                        continue
                    k = -1 if at_end else 0
                    t = lp[p, -1] - lp[p, -2] if at_end else lp[p, 1] - lp[p, 0]
                    t = t / t.norm().clamp_min(1e-6)
                    n = torch.stack([-t[1], t[0]])
                    corners += [lp[p, k] + n * lw[p] / 2, lp[p, k] - n * lw[p] / 2]
            if corners:
                q = torch.stack(corners)
                total = total + point_to_ring_distance(q, o["node_poly"][b, pn]).mean()
        return total

    # ------------------------------------------------------------------
    def topo(self, out: dict, tgt: dict, matches: list[dict]) -> dict:
        c = self.cfg["topo"]
        g, a = c["focal_gamma"], c["pos_alpha"]
        dev = out["E_adj"].device
        B, N = out["E_adj"].shape[:2]
        M = out["E_cross"].shape[1]
        L = {k: torch.zeros((), device=dev) for k in ("t_adj", "t_succ", "t_from", "t_to", "t_cross", "t_cross_node")}
        cnt = {"lane": 0, "cw": 0}
        for b in range(B):
            pl, gl = (torch.as_tensor(x, device=dev, dtype=torch.long) for x in matches[b]["lane"])
            pn, gn = matches[b]["node"]
            pc, gc = (torch.as_tensor(x, device=dev, dtype=torch.long) for x in matches[b]["cw"])
            g2n = torch.full((tgt["node_valid"].shape[1],), -1, dtype=torch.long, device=dev)
            g2n[torch.as_tensor(gn, device=dev, dtype=torch.long)] = torch.as_tensor(pn, device=dev, dtype=torch.long)
            T_adj = torch.zeros(N, N, device=dev)
            T_succ = torch.zeros(N, N, device=dev)
            if len(pl):
                T_adj[pl[:, None], pl[None]] = tgt["E_adj"][b][gl[:, None], gl[None]].float()
                T_succ[pl[:, None], pl[None]] = tgt["E_succ"][b][gl[:, None], gl[None]].float()
                cnt["lane"] += len(pl)
                tf = g2n[tgt["lane_from"][b, gl]]
                tt = g2n[tgt["lane_to"][b, gl]]
                ok = tf >= 0
                if ok.any():
                    L["t_from"] = L["t_from"] + F.cross_entropy(out["E_from"][b, pl[ok]], tf[ok], reduction="sum")
                ok = tt >= 0
                if ok.any():
                    L["t_to"] = L["t_to"] + F.cross_entropy(out["E_to"][b, pl[ok]], tt[ok], reduction="sum")
            L["t_adj"] = L["t_adj"] + sigmoid_focal(out["E_adj"][b], T_adj, a, g)
            L["t_succ"] = L["t_succ"] + sigmoid_focal(out["E_succ"][b], T_succ, a, g)
            T_cross = torch.zeros(M, N, device=dev)
            if len(pc) and len(pl):
                T_cross[pc[:, None], pl[None]] = tgt["E_cross"][b][gc[:, None], gl[None]].float()
            L["t_cross"] = L["t_cross"] + sigmoid_focal(out["E_cross"][b], T_cross, a, g)
            if len(pc):
                tn = g2n[tgt["cw_node"][b, gc]]
                ok = tn >= 0
                if ok.any():
                    L["t_cross_node"] = L["t_cross_node"] + F.cross_entropy(out["E_cross_node"][b, pc[ok]], tn[ok],
                                                                            reduction="sum")
                cnt["cw"] += len(pc)
        nl, nc = max(cnt["lane"], 1), max(cnt["cw"], 1)
        return {"t_adj": c["w_adj"] * L["t_adj"] / nl, "t_succ": c["w_succ"] * L["t_succ"] / nl,
                "t_from": c["w_from"] * L["t_from"] / nl, "t_to": c["w_to"] * L["t_to"] / nl,
                "t_cross": c["w_cross"] * L["t_cross"] / nc, "t_cross_node": c["w_cross_node"] * L["t_cross_node"] / nc}

    # ------------------------------------------------------------------
    def conn(self, pred: torch.Tensor, gidx: torch.Tensor, mask: torch.Tensor, tgt: dict) -> dict:
        if not mask.any():
            return {"k_pts": pred.sum() * 0}
        B = pred.shape[0]
        G = tgt["conn_pts"][torch.arange(B, device=pred.device)[:, None], gidx]          # [B, K, 8, 2]
        d = (pred[:, :, 1:-1] - G[:, :, 1:-1]).abs().sum(-1).mean(-1)
        return {"k_pts": self.cfg["conn"]["w_pts"] * (d * mask).sum() / mask.sum()}

    def port(self, out: dict, tgt: dict, matches: list[dict], pair_id: torch.Tensor) -> dict:
        dev = out["port_embed"].device
        total, n = torch.zeros((), device=dev), 0
        T = self.cfg["port"]["temperature"]
        ids = pair_id.tolist()
        seen = set()
        for b, pid in enumerate(ids):
            if pid < 0 or pid in seen:
                continue
            seen.add(pid)
            bb = [k for k, x in enumerate(ids) if x == pid]
            if len(bb) != 2:
                continue
            emb, keys = [], []
            for k in bb:
                pn, gn = matches[k]["node"]
                kk = tgt["port_key"][k, torch.as_tensor(gn, dtype=torch.long, device=tgt["port_key"].device)]
                sel = (kk >= 0).nonzero().flatten().tolist()
                emb.append(out["port_embed"][k, torch.as_tensor(pn, dtype=torch.long, device=dev)[sel]])
                keys.append(kk[sel].tolist())
            common = sorted(set(keys[0]) & set(keys[1]))
            if not common:
                continue
            ia = torch.as_tensor([keys[0].index(x) for x in common], device=dev)
            ib = torch.as_tensor([keys[1].index(x) for x in common], device=dev)
            S = emb[0][ia] @ emb[1].T / T
            lab_b = ib
            l1 = F.cross_entropy(S, lab_b)
            S2 = emb[1][ib] @ emb[0].T / T
            l2 = F.cross_entropy(S2, ia)
            total = total + (l1 + l2) / 2
            n += 1
        return {"port": self.cfg["node"]["w_port"] * total / max(n, 1)}

    # ------------------------------------------------------------------
    def __call__(self, out: dict, tgt: dict, matches: list[list[dict]], conn_pred=None, conn_gidx=None,
                 conn_mask=None, pair_id=None) -> tuple[torch.Tensor, dict]:
        c = self.cfg["sim"]
        parts = {}
        dense = self.dense(out["dense_raw"], tgt["dense"], tgt["hw"])
        parts.update(dense)
        nl = len(out["layers"])
        for li, o in enumerate(out["layers"]):
            last = li == nl - 1
            w = 1.0 if last else c["aux_layer_weight"]
            lo = self.layer(o if not last else out, tgt, matches[li], last)
            for k, v in lo.items():
                key = k if last else f"aux_{k}"
                parts[key] = parts.get(key, 0.0) + w * v
        parts.update(self.topo(out, tgt, matches[-1]))
        if conn_pred is not None:
            parts.update(self.conn(conn_pred, conn_gidx, conn_mask, tgt))
        if pair_id is not None and (pair_id >= 0).any():
            parts.update(self.port(out, tgt, matches[-1], pair_id))
        group = {"d_": c["lambda_dense"], "l_": c["lambda_L"], "n_": c["lambda_V"], "c_": c["lambda_C"],
                 "t_": c["lambda_topo"], "k_": c["lambda_K"], "port": c["lambda_port"]}
        total = 0.0
        for k, v in parts.items():
            base = k[4:] if k.startswith("aux_") else k
            lam = next((w for pre, w in group.items() if base.startswith(pre)), 1.0)
            total = total + lam * v
        return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}
