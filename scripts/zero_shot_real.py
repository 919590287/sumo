"""M1 真实验证集 Zero-shot 底图叠图与域差诊断报告（docs/DESIGN.md §7.2 M1「真实验证集」）。

用法：
    /opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.zero_shot_real --ckpt <best.pt> [--n 24] [--split val]
        [--half 70] [--gpkg city|dev] [--out runs/m1/zero_shot_real]

- 样本：广州全库 `信号灯点位` 按 60 m 连通聚类得到候选路口中心，经 split_v1 空间切分清单取 val 区块
  （隔离带与 dev 区块自动排除）；固定种子抽样。仅评测，不参与训练与调参。
- 每口：任意尺寸 ROI（默认 140 m 方形）→ 同一套光栅化 / Token 化 → 模型 → 约束解码 → SUMO 往返编译尝试 → 底图叠图；
- 域差诊断：真实 vs 合成验证窗口的 36 通道覆盖率均值、矢量 Token 类别分布、模型存在置信度与解码 / 编译成功率。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyogrio
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.dont_write_bytecode = True

from src.common.config import REPO_ROOT, cache_dir, load_config, path_setting  # noqa: E402
from src.common.device import select_device  # noqa: E402
from src.data.dataset import SynthWindowDataset, collate, tokens_dict  # noqa: E402
from src.data.rasterize import rasterize  # noqa: E402
from src.data.roi import RoiWindow, read_roi_features, square_roi, wgs84_to_utm  # noqa: E402
from src.data.splits import SplitManifest  # noqa: E402
from src.data.tokenize import TOKEN_CLASSES, tokenize  # noqa: E402
from src.eval.overlay import render_overlay  # noqa: E402
from src.infer.predict import predict_batch  # noqa: E402
from src.train.trainer import load_checkpoint_model  # noqa: E402


def junction_centers(gpkg: Path, link_m: float = 60.0) -> np.ndarray:
    df = pyogrio.read_dataframe(gpkg, layer=load_config("data")["layers"]["signal"], columns=[])
    xy = np.array([wgs84_to_utm(p.x, p.y) for p in df.geometry.values])
    t = cKDTree(xy)
    pairs = t.query_pairs(link_m, output_type="ndarray")
    from scipy.sparse import coo_matrix
    n = len(xy)
    a = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, lab = connected_components(a, directed=False)
    return np.array([xy[lab == i].mean(0) for i in range(k)])


def sample_input(gpkg: Path, cx: float, cy: float, half: float) -> tuple[dict, RoiWindow]:
    from scripts.eval_synth import _dummy_targets
    win = RoiWindow.from_roi(square_roi(cx, cy, half))
    feat = read_roi_features(gpkg, win)
    bev = rasterize(feat).tensor
    tok = tokenize(feat)
    s = {"bev": bev.astype(np.float16), "hw": np.array(win.shape, np.int64), **tokens_dict(tok),
         **_dummy_targets(win.shape), "meta": {"center": (cx, cy), "origin": win.origin}}
    return s, win


def channel_stats(bev: np.ndarray, hw) -> np.ndarray:
    H, W = int(hw[0]), int(hw[1])
    return bev[:, :H, :W].astype(np.float32).reshape(bev.shape[0], -1).mean(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--split", default="val")
    ap.add_argument("--half", type=float, default=70.0)
    ap.add_argument("--gpkg", default="city", choices=["city", "dev"])
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-sumo", dest="sumo", action="store_false")
    ap.add_argument("--out", default="runs/m1/zero_shot_real")
    args = ap.parse_args()
    gpkg = path_setting("gpkg_city" if args.gpkg == "city" else "gpkg_dev")
    out = REPO_ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    dev = select_device(args.device)
    model, ck = load_checkpoint_model(Path(args.ckpt), args.device)
    t0 = time.time()
    centers = junction_centers(gpkg)
    man = SplitManifest.load()
    lab = np.array([man.assign(x, y) for x, y in centers])
    pool = centers[lab == args.split] if args.gpkg == "city" else centers
    rng = np.random.default_rng(args.seed)
    pick = pool[rng.choice(len(pool), min(args.n, len(pool)), replace=False)]
    print(f"信号灯聚类路口 {len(centers)}，{args.split} 可选 {len(pool)}，抽样 {len(pick)}（{time.time() - t0:.0f}s）", flush=True)
    recs, real_stats, real_tok = [], [], Counter()
    names = [c["name"] for c in load_config("data")["channels"]]
    for k, (cx, cy) in enumerate(pick):
        s, win = sample_input(gpkg, float(cx), float(cy), args.half)
        b = collate([[s]])
        preds, o = predict_batch(model, b, dev)
        p = preds[0]
        rec = {"k": k, "center_utm": [float(cx), float(cy)], "hw": list(win.shape), "decode_ok": p.ok,
               "error": p.error, "n_lane_conf": int((p.tensors.lane_score >= 0.5).sum()),
               "n_node_conf": int((p.tensors.node_score >= 0.5).sum()),
               "n_cw_conf": int((p.tensors.cw_score >= 0.5).sum()),
               "lane_score_mean_top32": float(np.sort(p.tensors.lane_score)[-32:].mean())}
        real_stats.append(channel_stats(s["bev"], s["hw"]))
        real_tok.update(int(c) for c, m in zip(s["tok_cls"], s["tok_mask"]) if m)
        g = p.result.graph if p.ok else None
        if g is not None:
            g.origin = win.origin
            rec.update({"lanes": len(g.lanes), "edges": len(g.edges), "conns": len(g.connections),
                        "crosswalks": len(g.crosswalks), "raw": vars(p.result.raw), "decoded": vars(p.result.decoded)})
            if args.sumo:
                from src.eval.synth_eval import clean, eval_sumo
                w = cache_dir("zero_shot_work") / f"z{k}"
                rec["sumo"] = eval_sumo(g, w, load_config("eval")["sim_seeds"][:1], 0.20)
                clean(w)
        render_overlay(out / "figures" / f"real_{args.split}_{k:02d}.png", s["bev"], s["hw"], g,
                       title=f"zero-shot {args.split} #{k} UTM({cx:.0f},{cy:.0f}) decode={p.ok}")
        recs.append(rec)
        print(f"[{k + 1}/{len(pick)}] decode={p.ok} lanes≥0.5={rec['n_lane_conf']} nodes≥0.5={rec['n_node_conf']}", flush=True)
    # ---------------- 合成对照
    ds = SynthWindowDataset("val", train=False, seed=99, pair=False)
    syn_stats, syn_tok, syn_conf = [], Counter(), []
    for i in range(min(len(pick), len(ds))):
        it = ds[i][0]
        syn_stats.append(channel_stats(it["bev"], it["hw"]))
        syn_tok.update(int(c) for c, m in zip(it["tok_cls"], it["tok_mask"]) if m)
        pr, _ = predict_batch(model, collate([[it]]), dev)
        syn_conf.append(int((pr[0].tensors.lane_score >= 0.5).sum()))
    rs, ss = np.mean(real_stats, 0), np.mean(syn_stats, 0)
    rt = np.array([real_tok[i] for i in range(len(TOKEN_CLASSES))], float)
    st = np.array([syn_tok[i] for i in range(len(TOKEN_CLASSES))], float)
    rt, st = rt / max(rt.sum(), 1), st / max(st.sum(), 1)
    kl = float(np.sum(rt * np.log((rt + 1e-6) / (st + 1e-6))))
    diag = {"channel_mean_real": dict(zip(names, rs.round(5).tolist())),
            "channel_mean_synth": dict(zip(names, ss.round(5).tolist())),
            "channel_ratio_real_over_synth": dict(zip(names, (rs / np.maximum(ss, 1e-6)).round(3).tolist())),
            "token_dist_real": dict(zip(TOKEN_CLASSES, rt.round(4).tolist())),
            "token_dist_synth": dict(zip(TOKEN_CLASSES, st.round(4).tolist())), "token_kl_real_synth": kl,
            "decode_ok_rate_real": float(np.mean([r["decode_ok"] for r in recs])),
            "compile_ok_rate_real": float(np.mean([r.get("sumo", {}).get("compile_ok", False) for r in recs])),
            "lanes_conf_real_mean": float(np.mean([r["n_lane_conf"] for r in recs])),
            "lanes_conf_synth_mean": float(np.mean(syn_conf)) if syn_conf else None}
    report = {"ckpt": args.ckpt, "step": ck.get("step"), "gpkg": str(gpkg), "split": args.split, "n": len(recs),
              "diagnostics": diag, "samples": recs}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _plot_diag(out / "domain_gap_channels.png", names, rs, ss)
    print(json.dumps({k: v for k, v in diag.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1))
    return 0


def _plot_diag(path: Path, names, rs, ss) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, rs, 0.4, label="real (val)")
    ax.bar(x + 0.2, ss, 0.4, label="synthetic (val)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, fontsize=7)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend()
    ax.set_title("36-channel mean coverage: real vs synthetic")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
