"""推理：网络输出 → GraphTensors → 阶段 E 结构化约束图解码（成对轨迹头按激活对批量调用）→ LaneGraph。

不含任何几何 / 拓扑后处理规则；解码失败如实返回失败原因（计入评测）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.common.device import batch_to
from src.graph.decoder import DecodeResult, decode_graph
from src.models.b2s_net import to_graph_tensors


@dataclass
class Prediction:
    ok: bool
    result: DecodeResult | None
    tensors: object
    error: str = ""
    info: dict = field(default_factory=dict)


class _ConnFn:
    batched = True

    def __init__(self, model, out, b: int):
        self.model, self.out, self.b = model, out, b

    @torch.no_grad()
    def __call__(self, pairs: list[tuple[int, int]]) -> dict:
        dev = self.out["lane_pts"].device
        sub = {k: (v[self.b:self.b + 1] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == self.out["lane_pts"].shape[0]
                   else v) for k, v in self.out.items() if k != "_cache"}
        c = self.out["_cache"]
        feats = {k: v[self.b:self.b + 1] for k, v in c["feats"].items()}
        fr = c["frame"]
        frame = type(fr).__new__(type(fr))
        frame.__dict__.update(fr.__dict__)
        frame.h_true = fr.h_true[self.b:self.b + 1]
        res = {}
        mp = self.model.cfg["conn_head"]["max_pairs"] if hasattr(self.model, "cfg") else 256
        for s in range(0, len(pairs), mp):
            chunk = pairs[s:s + mp]
            pt = torch.tensor(chunk, dtype=torch.long, device=dev)[None]
            k = self.model.conn(sub["h_lane_topo"], sub["lane_pts"], pt, feats, frame)[0].float().cpu().numpy()
            for (i, j), p in zip(chunk, k):
                res[(i, j)] = p
        return res


@torch.no_grad()
def predict_batch(model, batch: dict, dev: torch.device, autocast_dtype=None) -> tuple[list[Prediction], dict]:
    model.eval()
    batch = batch_to(batch, dev)
    if autocast_dtype is not None:
        with torch.autocast(dev.type, dtype=autocast_dtype):
            out = model(batch)
    else:
        out = model(batch)
    preds = []
    for b in range(batch["bev"].shape[0]):
        t = to_graph_tensors(out, b)
        try:
            r = decode_graph(t, conn_pts=_ConnFn(model, out, b))
            preds.append(Prediction(True, r, t, info=r.info))
        except Exception as e:  # noqa: BLE001
            preds.append(Prediction(False, None, t, error=repr(e)[:200]))
    return preds, out
