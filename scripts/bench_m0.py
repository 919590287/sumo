"""M0 本机 M4 Pro 实测基准（docs/DESIGN.md §6 性能实测门禁 + §7.2 M0 验收第 3 条）。

用法：/opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.bench_m0 [--quick]

1. 光栅化：选择性抗锯齿（窄线 ss4 vs 解析覆盖率）单样本 CPU 耗时 + 8 进程有效吞吐；
2. MPS 原生双线性采样：相对 CPU grid_sample 数值误差 + 反向零 CPU fallback；
3. BEV 编码器 + 2 层 MSDeformAttn：896² 训练前反向、1024² 推理前向延迟与显存；fp16/bf16 AMP 稳定性；
4. 6–8 进程并行 netconvert + 往返核验 + sumo 的路口/秒吞吐。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")   # 任何 CPU 回退直接报错

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src.common.config import REPO_ROOT, cache_dir, load_config, path_setting


# ====================================================================== 1. 光栅化
def _junction_centers(n: int) -> list[tuple[float, float]]:
    import pyogrio
    from src.data.roi import wgs84_to_utm
    gp = path_setting("gpkg_dev")
    sig = pyogrio.read_dataframe(gp, layer="信号灯点位", columns=["分类代码"])
    cw = pyogrio.read_dataframe(gp, layer="南沙道路要素", columns=["分类代码"], where="分类代码='crosswalk_area'")
    pts = list(zip(sig.geometry.x.values, sig.geometry.y.values))
    pts += [(c.x, c.y) for c in cw.geometry.centroid.values[::3]]
    x, y = wgs84_to_utm(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return list(zip(x.tolist(), y.tolist()))[:n]


def _raster_job(args):
    cx, cy, half, method = args
    from src.data.rasterize import rasterize
    from src.data.roi import RoiWindow, read_roi_features, square_roi
    from src.data.tokenize import tokenize
    win = RoiWindow.from_roi(square_roi(cx, cy, half))
    t0 = time.perf_counter()
    f = read_roi_features(path_setting("gpkg_dev"), win)
    t1 = time.perf_counter()
    s = rasterize(f, method=method)
    t2 = time.perf_counter()
    tk = tokenize(f)
    t3 = time.perf_counter()
    return {"shape": list(win.shape), "read": t1 - t0, "raster": t2 - t1, "narrow": s.timing["narrow"],
            "tokens": t3 - t2, "n_tokens": tk.n}


def bench_raster(quick: bool) -> dict:
    centers = _junction_centers(8 if quick else 24)
    out = {}
    for px in (512, 896, 1024):
        half = (px * 0.2 - 32) / 2
        for method in ("analytic", "ss4"):
            rs = [_raster_job((cx, cy, half, method)) for cx, cy in centers]
            out[f"{px}_{method}"] = {
                "window_px": px, "method": method, "n": len(rs),
                "raster_s_mean": float(np.mean([r["raster"] for r in rs])),
                "raster_s_p90": float(np.percentile([r["raster"] for r in rs], 90)),
                "narrow_s_mean": float(np.mean([r["narrow"] for r in rs])),
                "read_s_mean": float(np.mean([r["read"] for r in rs])),
                "tokens_s_mean": float(np.mean([r["tokens"] for r in rs])),
            }
            print(f"[raster] {px}px {method:8s} raster={out[f'{px}_{method}']['raster_s_mean']:.3f}s "
                  f"(p90 {out[f'{px}_{method}']['raster_s_p90']:.3f}) read={out[f'{px}_{method}']['read_s_mean']:.3f}s "
                  f"tokens={out[f'{px}_{method}']['tokens_s_mean']:.3f}s", flush=True)
    # 8 进程有效吞吐（896² 训练窗，读 + 光栅化 + Token 化全流程）
    half = (896 * 0.2 - 32) / 2
    for method in ("analytic", "ss4"):
        jobs = [(cx, cy, half, method) for cx, cy in centers] * (2 if quick else 4)
        ctx = mp.get_context("spawn")
        with ctx.Pool(8) as pool:
            pool.map(_raster_job, jobs[:8])
            t0 = time.perf_counter()
            pool.map(_raster_job, jobs)
            dt = time.perf_counter() - t0
        out[f"pool8_896_{method}"] = {"samples": len(jobs), "seconds": dt, "samples_per_s": len(jobs) / dt}
        print(f"[raster] 8 进程 896px {method}: {len(jobs) / dt:.1f} 样本/s", flush=True)
    return out


# ====================================================================== 2/3. MPS
def bench_mps(quick: bool) -> dict:
    import torch
    from src.models.bev_encoder import BevEncoder, count_params
    from src.models.sampling import MSDeformAttn, bilinear_sample, grid_sample_reference
    res: dict = {"torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
                 "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
    dev = torch.device("mps")
    torch.manual_seed(0)
    # ---- 采样精度
    errs = {"fwd": 0.0, "grad_feat": 0.0, "grad_grid_rel": 0.0}
    for B, C, H, W, Q in ((2, 32, 48, 64, 500), (1, 128, 224, 224, 4096), (4, 64, 56, 56, 2000)):
        f = torch.randn(B, C, H, W)
        g = torch.rand(B, Q, 2) * 2.4 - 1.2
        fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
        ref = grid_sample_reference(fr, gr)
        w = torch.randn_like(ref)
        (ref * w).sum().backward()
        fm, gm = f.to(dev).requires_grad_(), g.to(dev).requires_grad_()
        o = bilinear_sample(fm, gm)
        (o * w.to(dev)).sum().backward()
        errs["fwd"] = max(errs["fwd"], (o.cpu() - ref).abs().max().item())
        errs["grad_feat"] = max(errs["grad_feat"], (fm.grad.cpu() - fr.grad).abs().max().item())
        errs["grad_grid_rel"] = max(errs["grad_grid_rel"],
                                    ((gm.grad.cpu() - gr.grad).abs().max() / gr.grad.abs().max()).item())
    try:
        x = torch.randn(1, 4, 8, 8, device=dev, requires_grad=True)
        gg = torch.rand(1, 4, 1, 2, device=dev, requires_grad=True)
        torch.nn.functional.grid_sample(x, gg, align_corners=False).sum().backward()
        native = "ok"
    except NotImplementedError as e:
        native = f"NotImplementedError: {str(e)[:80]}"
    res["sampling"] = {**errs, "pass_fwd_lt_1e-5": errs["fwd"] < 1e-5, "native_grid_sample_backward_on_mps": native}
    print(f"[mps] 采样误差 fwd={errs['fwd']:.2e} grad_feat={errs['grad_feat']:.2e} "
          f"grad_grid_rel={errs['grad_grid_rel']:.2e}；原生 grid_sample 反向：{native}", flush=True)

    # ---- 模型前反向
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = BevEncoder()
            self.q = torch.nn.Parameter(torch.randn(1, 216 + 512, 256) * 0.02)
            self.attn = torch.nn.ModuleList([MSDeformAttn(256, 3, 8, 4) for _ in range(2)])
            self.head = torch.nn.Conv2d(128, 4, 1)

        def forward(self, x):
            f = self.enc(x)
            B = x.shape[0]
            q = self.q.repeat(B, 1, 1)
            ref = torch.sigmoid(q[..., :2])
            for a in self.attn:
                q = q + a(q, ref, [f["F4"], f["F8"], f["F16"]])
            return self.head(f["F2"]).float().pow(2).mean() + q.float().pow(2).mean()

    model = Probe().to(dev)
    res["probe_params_M"] = count_params(model) / 1e6
    rows = []
    cases = [("train", 896, 1), ("train", 896, 2), ("train", 896, 4), ("infer", 1024, 1), ("train", 512, 8)]
    if quick:
        cases = cases[:2] + cases[3:4]
    for mode, px, bs in cases:
        for amp in ("fp32", "fp16", "bf16"):
            torch.mps.empty_cache()
            x = torch.randn(bs, 36, px, px, device=dev)
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
            model.train(mode == "train")
            ts = []
            ok = True
            try:
                for it in range(4):
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast("mps", dtype=dtype, enabled=dtype is not None):
                        if mode == "train":
                            loss = model(x)
                        else:
                            with torch.no_grad():
                                loss = model(x)
                    if mode == "train":
                        model.zero_grad(set_to_none=True)
                        loss.backward()
                    torch.mps.synchronize()
                    if it > 0:
                        ts.append(time.perf_counter() - t0)
                mem = torch.mps.driver_allocated_memory() / 2**30
            except Exception as e:  # noqa
                ok, mem = False, float("nan")
                ts = [float("nan")]
                print(f"[mps] {mode} {px} bs{bs} {amp} 失败：{e}", flush=True)
            rows.append({"mode": mode, "px": px, "batch": bs, "amp": amp, "ok": ok,
                         "latency_s": float(np.mean(ts)), "driver_mem_GiB": mem})
            print(f"[mps] {mode:5s} {px} bs{bs} {amp}: {np.mean(ts):.3f}s/iter  mem {mem:.2f} GiB", flush=True)
            del x
    res["latency"] = rows
    # ---- AMP 数值稳定性（同一输入：loss 相对误差、梯度余弦相似度、全部有限）
    x = torch.randn(2, 36, 512, 512, device=dev)
    model.train()
    grads = {}
    losses = {}
    for amp in ("fp32", "fp16", "bf16"):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        model.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=dtype, enabled=dtype is not None):
            loss = model(x)
        loss.backward()
        losses[amp] = loss.item()
        grads[amp] = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
    amp_res = {}
    for amp in ("fp16", "bf16"):
        cos = torch.nn.functional.cosine_similarity(grads[amp], grads["fp32"], dim=0).item()
        amp_res[amp] = {"loss_rel_err": abs(losses[amp] - losses["fp32"]) / abs(losses["fp32"]),
                        "grad_cosine": cos, "all_finite": bool(torch.isfinite(grads[amp]).all().item())}
        print(f"[mps] AMP {amp}: loss 相对误差 {amp_res[amp]['loss_rel_err']:.2e} 梯度余弦 {cos:.5f} "
              f"有限 {amp_res[amp]['all_finite']}", flush=True)
    res["amp"] = amp_res
    return res


# ====================================================================== 4. 仿真吞吐
def _sim_job(args):
    name, k = args
    from src.data.standard_graphs import STANDARD_GRAPHS
    from src.sumo_io.pipeline import compile_graph
    from src.sumo_io.simulate import run_sumo, write_demand
    work = cache_dir("m0_bench_sim", f"{name}_{k}")
    t0 = time.perf_counter()
    g = STANDARD_GRAPHS[name]()
    r = compile_graph(g, work)
    ok = bool(r.ok)
    rou = write_demand(g, r.plain.ids, work / "probe.rou.xml", "probe", 42)
    s = run_sumo(r.eval_sim.net_path, rou, work, 42, "probe")
    return {"ok": ok and s.collisions == 0 and s.teleports == 0, "seconds": time.perf_counter() - t0}


def bench_sim(quick: bool) -> dict:
    from src.data.standard_graphs import STANDARD_GRAPHS
    names = list(STANDARD_GRAPHS)
    reps = 1 if quick else 3
    jobs = [(n, k) for k in range(reps) for n in names]
    out = {}
    for nproc in (1, 6, 8):
        ctx = mp.get_context("spawn")
        with ctx.Pool(nproc) as pool:
            pool.map(_sim_job, jobs[:nproc])
            t0 = time.perf_counter()
            rs = pool.map(_sim_job, jobs)
            dt = time.perf_counter() - t0
        out[f"proc{nproc}"] = {"junctions": len(jobs), "seconds": dt, "junctions_per_s": len(jobs) / dt,
                               "all_ok": all(r["ok"] for r in rs),
                               "mean_job_s": float(np.mean([r["seconds"] for r in rs]))}
        print(f"[sim] {nproc} 进程：{len(jobs) / dt:.2f} 路口/s（编译+往返核验+测试台编译+探测仿真），"
              f"全部通过 {out[f'proc{nproc}']['all_ok']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/m0/benchmark.json")
    ap.add_argument("--only", default="raster,mps,sim")
    args = ap.parse_args()
    parts = args.only.split(",")
    out = REPO_ROOT / args.out
    rep = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    rep.update({"host": {"machine": platform.machine(), "platform": platform.platform(), "cpu_count": os.cpu_count()},
                "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    if "raster" in parts:
        rep["raster"] = bench_raster(args.quick)
    if "mps" in parts:
        rep["mps"] = bench_mps(args.quick)
    if "sim" in parts:
        rep["sim"] = bench_sim(args.quick)
    # ---- 定型判定（DESIGN §6：ss4 单样本 > 0.25 s 或 8 进程吞吐低于 GPU 喂食速率 → 切换解析覆盖率）
    if "raster" in rep:
        r = rep["raster"]
        feed = None
        if "mps" in rep:
            tr = [x for x in rep["mps"]["latency"] if x["mode"] == "train" and x["px"] == 896 and x["amp"] == "fp32"
                  and x["ok"]]
            if tr:
                best = min(tr, key=lambda x: x["latency_s"] / x["batch"])
                feed = best["batch"] / best["latency_s"]
        ss4_single = r["896_ss4"]["raster_s_mean"]
        ss4_pool = r["pool8_896_ss4"]["samples_per_s"]
        rep["decision"] = {
            "gpu_feed_samples_per_s_896": feed,
            "ss4_single_s_896": ss4_single,
            "ss4_pool8_samples_per_s": ss4_pool,
            "analytic_pool8_samples_per_s": r["pool8_896_analytic"]["samples_per_s"],
            "switch_to_analytic": bool(ss4_single > 0.25 or (feed is not None and ss4_pool < feed)),
            "configured_narrow_method": load_config("data")["raster"]["narrow_method"],
        }
        print("[decision]", json.dumps(rep["decision"], ensure_ascii=False), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"基准报告已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
