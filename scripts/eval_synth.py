"""M1 合成留出测试集评测（docs/DESIGN.md §7.2 M1 门槛）。

用法（仓库根目录）：
    /opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.eval_synth --ckpt <best.pt> [--split test] [--n 1000]
        [--no-sumo] [--tiles] [--out runs/m1/eval_synth]

输出：report.json（汇总指标 + 门禁判定 + 双轨拓扑 + 逐样本记录）与典型样例底图叠图 figures/。
``--tiles``：在 tile_test（>172.8 m 走廊场景）上执行分块推理 + 端口拼接验证。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import shapely
import shapely.affinity
import torch
from shapely.geometry import box

sys.dont_write_bytecode = True

from src.common.config import REPO_ROOT, cache_dir, load_config  # noqa: E402
from src.common.device import select_device  # noqa: E402
from src.data.dataset import FeatureIndex, SynthWindowDataset, WindowSpec, build_input, collate  # noqa: E402
from src.data.synth.library import SynthLibrary  # noqa: E402
from src.data.targets import crop_graph_to_roi, gt_region, node_areas  # noqa: E402
from src.eval.overlay import render_overlay  # noqa: E402
from src.eval.synth_eval import aggregate, clean, eval_geometry, eval_sumo  # noqa: E402
from src.graph.stitching import plan_tiles, plan_tiles_avoiding, stitch_graphs  # noqa: E402
from src.infer.predict import predict_batch  # noqa: E402
from src.train.trainer import load_checkpoint_model  # noqa: E402


def oracle_pred(gt):
    """评测框架自检：以真值图张量走同一约束解码 / 指标 / SUMO 管线（结果只用于校验评测代码，不是模型成绩）。"""
    from src.graph.decoder import decode_graph
    from src.graph.stitching import _port_embed
    from src.infer.predict import Prediction
    for n in gt.nodes:
        if n.node_type in ("roi_entry_port", "roi_exit_port"):
            n.port_embed = _port_embed((getattr(n, "port_key", n.id),), None, 0.0)
    t = gt.to_tensors()

    class _Fallback(dict):                       # 解码器按非零度公理补出的连接对：直线占位（仅自检）
        def __missing__(self, key):
            i, j = key
            return np.linspace(t.lane_pts[i, -1], t.lane_pts[j, 0], 8)
    try:
        r = decode_graph(t, conn_pts=_Fallback(t.conn_pts))
        return Prediction(True, r, t, info=r.info)
    except Exception as e:  # noqa: BLE001
        return Prediction(False, None, t, error=repr(e)[:200])


def _topo(rec: dict, pred) -> None:
    for track in ("raw", "decoded"):
        st = getattr(pred.result, track) if pred.ok else None
        rec[f"{track}_dead_end"] = st.dead_end if st else 0
        rec[f"{track}_unfed"] = st.unfed if st else 0
        rec[f"{track}_crossings"] = st.same_approach_crossings if st else 0


def run_windows(model, dev, args, out_dir: Path) -> dict:
    ds = SynthWindowDataset(args.split, train=False, seed=args.seed, pair=False, randomize=True)
    n = min(args.n, len(ds))
    seeds = load_config("eval")["sim_seeds"]
    tol = load_config("eval")["gates_m1"]["roundtrip_hausdorff_m"]
    work = cache_dir("eval_synth_work")
    recs, t0 = [], time.time()
    for s0 in range(0, n, args.batch):
        items = [ds[i] for i in range(s0, min(s0 + args.batch, n))]
        batch = collate(items)
        if args.oracle:
            preds = [oracle_pred(m["graph"]) for m in batch["meta"]]
        else:
            preds, _ = predict_batch(model, batch, dev)
        for k, (pred, meta) in enumerate(zip(preds, batch["meta"])):
            idx = s0 + k
            g = pred.result.graph if pred.ok else None
            rec = eval_geometry(g, meta["graph"], meta, batch["hw"][k].tolist(), 0.2)
            rec.update({"idx": idx, "seed": meta["seed"], "kind": meta["kind"], "decode_ok": pred.ok,
                        "decode_error": pred.error})
            _topo(rec, pred)
            if args.sumo:
                if g is not None:
                    w = work / f"s{idx}"
                    rec["sumo"] = eval_sumo(g, w, seeds, tol)
                    clean(w)
                else:
                    rec["sumo"] = {"compile_ok": False, "error": "decode_failed"}
            if idx < args.figures:
                render_overlay(out_dir / "figures" / f"{args.split}_{idx:04d}.png", batch["bev"][k].float().numpy(),
                               batch["hw"][k].tolist(), g, meta["graph"],
                               title=f"{args.split} #{idx} {meta['kind']} seed={meta['seed']} decode={pred.ok}")
            recs.append(rec)
        print(f"[eval {args.split}] {len(recs)}/{n}  {time.time() - t0:.0f}s", flush=True)
    summ = aggregate(recs)
    for r in recs:
        r["lat_p50"] = float(np.percentile(r["lat"], 50)) if len(r["lat"]) else None
        r.pop("lat")
    return {"summary": summ, "samples": recs}


def run_tiles(model, dev, args, out_dir: Path) -> dict:
    lib = SynthLibrary("tile_test")
    res = 0.2
    cfg_data = load_config("data")
    seeds = load_config("eval")["sim_seeds"]
    tol = load_config("eval")["gates_m1"]["roundtrip_hausdorff_m"]
    work = cache_dir("eval_tiles_work")
    recs = []
    for i in range(min(args.n_tiles, len(lib))):
        rec_s = lib[i]
        fidx = FeatureIndex(rec_s)
        cores = [n.pos[:2] for n in rec_s.graph.nodes if n.node_type == "junction_core"]
        c = np.asarray(cores)
        roi = box(*(c.min(0) - 45.0), *(c.max(0) + 45.0))
        def infer_tiles(tiles):
            graphs, fails = [], 0
            for t in tiles:
                spec = WindowSpec((float(t.origin[0]), float(t.origin[1])), t.shape_px, box(*t.core))
                if args.oracle:
                    sub = crop_graph_to_roi(rec_s.graph, box(*t.core), shift=spec.origin)
                    pred = oracle_pred(sub)
                else:
                    x0, y0 = spec.origin
                    rows = fidx.select(x0, y0, x0 + t.shape_px[1] * res, y0 + t.shape_px[0] * res)
                    _, sample = build_input(rows, spec, cfg_data)
                    b = collate([[{**sample, **_dummy_targets(t.shape_px), "meta": {}}]])
                    pred = predict_batch(model, b, dev)[0][0]
                if pred.ok:
                    g = pred.result.graph
                    g.origin = spec.origin
                    graphs.append(g)
                else:
                    fails += 1
            return graphs, fails

        # 第一遍：等分分块推理，取模型输出的（非端口）节点区域；第二遍：切线避开这些区域后重新推理
        tiles = plan_tiles(roi.bounds)
        graphs, fails = infer_tiles(tiles)
        avoid = []
        for g in graphs:
            for nid, a in node_areas(g).items():
                if next(n for n in g.nodes if n.id == nid).node_type not in ("roi_entry_port", "roi_exit_port"):
                    avoid.append(shapely.affinity.translate(a, *g.origin))
        if len(tiles) > 1 and avoid:
            tiles = plan_tiles_avoiding(roi.bounds, avoid)
            graphs, fails = infer_tiles(tiles)
        cut_lines = shapely.union_all([box(*t.core).exterior for t in tiles]).difference(roi.exterior.buffer(0.01))
        gnt = {n.id: n.node_type for n in rec_s.graph.nodes}
        cut_nodes = sum(a.intersects(cut_lines) for nid, a in node_areas(rec_s.graph).items()
                        if gnt[nid] not in ("roi_entry_port", "roi_exit_port"))
        rec = {"idx": i, "seed": rec_s.meta["seed"], "n_tiles": len(tiles), "tile_decode_fail": fails,
               "tile_boundaries_cutting_nodes": int(cut_nodes), "decode_ok": False}
        base = (min(t.origin[0] for t in tiles), min(t.origin[1] for t in tiles))
        gt = crop_graph_to_roi(rec_s.graph, gt_region(rec_s.graph, roi), shift=base)
        try:
            st, rep = stitch_graphs(graphs, base_origin=base) if graphs and not fails else (None, None)
        except Exception as e:  # noqa: BLE001
            st, rep = None, None
            rec["stitch_error"] = repr(e)[:200]
        Wm = max(t.window[2] for t in tiles) - base[0]
        Hm = max(t.window[3] for t in tiles) - base[1]
        hw = (int(math.ceil(Hm / res)), int(math.ceil(Wm / res)))
        g_rec = eval_geometry(st, gt, {}, hw, res)
        rec.update(g_rec)
        rec["decode_ok"] = st is not None
        if rep is not None:
            rec.update({"ports_matched": len(rep.matched), "ports_unmatched": len(rep.unmatched_exit) + len(rep.unmatched_entry),
                        "lane_count_mismatch": len(rep.lane_count_mismatch)})
        for track in ("raw", "decoded"):
            rec[f"{track}_dead_end"] = rec[f"{track}_unfed"] = rec[f"{track}_crossings"] = 0
        if args.sumo:
            if st is not None:
                w = work / f"t{i}"
                rec["sumo"] = eval_sumo(st, w, seeds, tol)
                clean(w)
            else:
                rec["sumo"] = {"compile_ok": False, "error": "stitch_failed"}
        recs.append(rec)
        print(f"[eval tiles] {i + 1}  tiles={len(tiles)} stitched={st is not None}", flush=True)
    summ = aggregate(recs)
    summ["mean_tiles"] = float(np.mean([r["n_tiles"] for r in recs])) if recs else 0.0
    summ["stitch_ok_rate"] = float(np.mean([r["decode_ok"] for r in recs])) if recs else 0.0
    summ["samples_with_node_cut_by_tile_boundary"] = int(sum(r["tile_boundaries_cutting_nodes"] > 0 for r in recs))
    for r in recs:
        r.pop("lat", None)
    return {"summary": summ, "samples": recs}


def _dummy_targets(hw) -> dict:
    """推理无真值：以空图占位使 collate 结构一致。"""
    from src.graph.lane_graph import LaneGraph
    from src.data.targets import DENSE_CHANNELS, graph_targets
    return {**graph_targets(LaneGraph([], [], [], [], [])),
            "dense": np.zeros((len(DENSE_CHANNELS), hw[0] // 2, hw[1] // 2), np.float16)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-tiles", type=int, default=120)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--figures", type=int, default=12)
    ap.add_argument("--no-sumo", dest="sumo", action="store_false")
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="评测框架自检：以真值张量代替模型输出")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/m1/eval_synth")
    args = ap.parse_args()
    dev = select_device(args.device)
    model, ck = (None, {}) if args.oracle else load_checkpoint_model(Path(args.ckpt), args.device)
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"ckpt": str(args.ckpt), "step": ck.get("step"), "device": str(dev), "args": vars(args)}
    report["windows"] = run_windows(model, dev, args, out_dir)
    if args.tiles:
        report["tiles"] = run_tiles(model, dev, args, out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in report.items() if isinstance(v, dict) and "summary" in v},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
