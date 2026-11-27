"""训练期快速验证指标（基于末层 Hungarian 匹配，无需约束解码 / 编译）。

完整的里程碑评测（约束解码 + SUMO 往返编译 + 仿真 + 双轨拓扑）见 ``src/eval/synth_eval.py``。
"""
from __future__ import annotations

import numpy as np
import torch

from src.common.config import load_config

_EV = None


def _cfg():
    global _EV
    if _EV is None:
        _EV = load_config("eval")
    return _EV


def lateral_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """预测中心线各点到真值中心线折线的距离（米）。"""
    a, b = gt[:-1], gt[1:]
    ab = b - a
    t = np.clip(((pred[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pred[:, None] - proj, axis=-1).min(1)


@torch.no_grad()
def quick_metrics(out: dict, tgt: dict, matches: list[dict]) -> dict:
    ev = _cfg()["match"]
    B = tgt["lane_valid"].shape[0]
    tp = fp = fn = 0
    errs = []
    ntp = nfp = nfn = 0
    ctp = cfp = cfn = 0
    s_tp = s_fp = s_fn = 0
    for b in range(B):
        sc = torch.sigmoid(out["lane_score"][b]).cpu().numpy()
        pl, gl = matches[b]["lane"]
        P = out["lane_pts"][b].cpu().numpy()
        G = tgt["lane_pts"][b].cpu().numpy()
        ok = {}
        for p, g in zip(pl, gl):
            if sc[p] < 0.5:
                continue
            e = lateral_errors(P[p], G[g])
            d = np.abs(P[p] - G[g]).sum(-1).mean() / 2
            if d < ev["lane_m"]:
                ok[p] = g
                errs.append(e)
        tp += len(ok)
        fp += int((sc >= 0.5).sum()) - len(ok)
        fn += int(tgt["lane_valid"][b].sum()) - len(ok)
        # 节点 / 人行横道
        nsc = torch.sigmoid(out["node_score"][b]).cpu().numpy()
        pn, gn = matches[b]["node"]
        good = sum(1 for p, g in zip(pn, gn) if nsc[p] >= 0.5 and
                   np.abs(out["node_pos"][b, p].cpu().numpy() - tgt["node_pos"][b, g].cpu().numpy()).sum() / 2 < ev["node_m"])
        ntp += good
        nfp += int((nsc >= 0.5).sum()) - good
        nfn += int(tgt["node_valid"][b].sum()) - good
        csc = torch.sigmoid(out["cw_score"][b]).cpu().numpy()
        pc, gc = matches[b]["cw"]
        good = 0
        for p, g in zip(pc, gc):
            if csc[p] < 0.5:
                continue
            P2 = out["cw_end"][b, p].cpu().numpy()
            G2 = tgt["cw_end"][b, g].cpu().numpy()
            d = min(np.abs(P2 - G2).sum(-1).mean(), np.abs(P2 - G2[::-1]).sum(-1).mean()) / 2
            good += d < ev["crosswalk_m"]
        ctp += good
        cfp += int((csc >= 0.5).sum()) - good
        cfn += int(tgt["cw_valid"][b].sum()) - good
        # E_succ（在已正确匹配车道之间，阈值 0.5，原始输出）
        es = torch.sigmoid(out["E_succ"][b]).cpu().numpy()
        T = tgt["E_succ"][b].cpu().numpy()
        items = list(ok.items())
        for pi, gi in items:
            for pj, gj in items:
                if pi == pj:
                    continue
                pr, gt = es[pi, pj] >= 0.5, T[gi, gj] > 0
                s_tp += pr and gt
                s_fp += pr and not gt
                s_fn += gt and not pr
    f1 = lambda a, b_, c: 2 * a / max(2 * a + b_ + c, 1)  # noqa: E731
    e = np.concatenate(errs) if errs else np.array([np.nan])
    return {"lane_f1": f1(tp, fp, fn), "node_f1": f1(ntp, nfp, nfn), "cw_f1": f1(ctp, cfp, cfn),
            "succ_f1": f1(s_tp, s_fp, s_fn), "lat_p50": float(np.nanpercentile(e, 50)),
            "lat_p90": float(np.nanpercentile(e, 90))}
