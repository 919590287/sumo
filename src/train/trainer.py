"""设备无关训练框架（docs/DESIGN.md §6.1–6.2）：同一套代码服务 Mac MPS 与 Linux CUDA。

- 设备：B2S_DEVICE（auto → cuda → mps → cpu）；AMP：fp32 / bf16 / fp16（auto 读取 calibration.json 判定）；
- 优化：AdamW（主干较低学习率）+ 线性 warmup + 余弦退火；梯度累积与裁剪；
- 可复现落盘（runs/<run_id>/）：config_snapshot.json、seed、split / 合成库清单、calibration.json、
  train.log（人类可读）、metrics.jsonl（逐步结构化指标）、checkpoints/{last,best}.pt（CPU state_dict + 配置快照 + MANIFEST）；
- 断点续训：从 last.pt 恢复模型 / 优化器 / 调度 / 缩放器 / 随机状态与步数；
- 终端实时进度：阶段 / epoch / step、损失分量、学习率、样本/s、已用时间与 ETA、显存与 GPU 利用率、最近验证指标。
"""
from __future__ import annotations

from functools import partial
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.device import amp_dtype, batch_to, device_info, gpu_util, memory_gib, select_device, synchronize
from src.data.dataset import SynthWindowDataset, collate
from src.models.b2s_net import build_model, count_params
from src.models.losses import SimCriterion
from src.models.matching import conn_pairs, match_all

HEADLINE = ("d_valid", "d_center", "l_pts", "l_score", "n_pos", "t_succ", "t_adj", "k_pts", "port")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def float_out(out: dict) -> dict:
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
    f["layers"] = [{k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in o.items()}
                   for o in out["layers"]]
    return f


def train_step(model, batch, crit, loss_cfg, dev, autocast_dtype):
    """一次前向 + 匹配 + 成对轨迹 + 损失（不含反向）。"""
    ctx = torch.autocast(dev.type, dtype=autocast_dtype) if autocast_dtype is not None else _null()
    with ctx:
        out = model(batch)
    fo = float_out(out)
    matches = match_all(fo, batch, loss_cfg)
    pairs, gidx, pmask = conn_pairs(matches[-1], batch, model.cfg["conn_head"]["max_pairs"])
    kmax = max(int(pmask.sum(1).max()), 1)
    pairs, gidx, pmask = pairs[:, :kmax].to(dev), gidx[:, :kmax].to(dev), pmask[:, :kmax].to(dev)
    with ctx:
        kc = model.connections(out, pairs)
    total, parts = crit(fo, batch, matches, kc.float(), gidx, pmask, batch.get("pair_id"))
    return total, parts, fo, matches


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Trainer:
    def __init__(self, run_dir: Path, stage: str = "layer1", overrides: dict | None = None, resume: bool = False,
                 device: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.cfg = dict(load_config("train")[stage])
        self.cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
        self.stage = stage
        self.dev = select_device(device)
        self.loss_cfg = load_config("losses")
        self.amp = self._amp_mode()
        seed_everything(self.cfg["seed"])
        self.log_f = open(self.run_dir / "train.log", "a", encoding="utf-8")
        self.met_f = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        self.model = build_model().to(self.dev)
        self.crit = SimCriterion(self.loss_cfg)
        bb = [p for n, p in self.model.named_parameters() if n.startswith("bev.")]
        rest = [p for n, p in self.model.named_parameters() if not n.startswith("bev.")]
        self.opt = torch.optim.AdamW([{"params": bb, "lr": self.cfg["lr_backbone"]},
                                      {"params": rest, "lr": self.cfg["lr"]}], weight_decay=self.cfg["weight_decay"])
        self.base_lr = [self.cfg["lr_backbone"], self.cfg["lr"]]
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp == "fp16" and self.dev.type == "cuda"))
        self.train_ds = SynthWindowDataset(self.cfg["train_split"], train=True, seed=self.cfg["seed"],
                                           fixed_px=self.cfg.get("fixed_px"),
                                           length=self.cfg.get("train_length"))
        self.val_ds = SynthWindowDataset(self.cfg["val_split"], train=False, seed=self.cfg["seed"] + 1,
                                         pair=False, fixed_px=self.cfg.get("fixed_px"))
        self.step, self.epoch, self.best = 0, 0, float("inf")
        self.last_val: dict = {}
        steps_per_epoch = math.ceil(len(self.train_ds) / (self.cfg["batch_size"] * self.cfg["grad_accum"]))
        self.total_steps = self.cfg.get("max_steps") or steps_per_epoch * self.cfg["epochs"]
        if resume:
            self.load(self.run_dir / "checkpoints" / "last.pt")
        self._snapshot()

    # ------------------------------------------------------------------ 配置 / 落盘
    def _amp_mode(self) -> str:
        mode = self.cfg.get("amp", "auto")
        if mode != "auto":
            return mode
        cal = self.run_dir / "calibration.json"
        if self.dev.type == "cuda" and cal.exists():
            return json.loads(cal.read_text())["amp"]["decision"]
        return "fp32"

    def _snapshot(self) -> None:
        from src.data.synth.library import library_root
        snap = {"stage": self.stage, "train_cfg": self.cfg, "amp": self.amp, "device": device_info(self.dev),
                "configs": {k: load_config(k) for k in ("model", "losses", "graph", "data", "synth", "decoder")},
                "params": count_params(self.model), "synth_dir": str(library_root()),
                "total_steps": self.total_steps, "argv": sys.argv, "env_B2S": {k: v for k, v in os.environ.items()
                                                                               if k.startswith("B2S_")}}
        man = Path(library_root()) / "MANIFEST.json"
        if man.exists():
            snap["synth_manifest"] = json.loads(man.read_text())
        bman = os.environ.get("B2S_BUNDLE_MANIFEST")
        if bman and Path(bman).exists():
            snap["bundle_manifest"] = json.loads(Path(bman).read_text())
        (self.run_dir / "config_snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1, default=str),
                                                           encoding="utf-8")
        self.snapshot = snap

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.log_f.write(line + "\n")
        self.log_f.flush()
        print(line, flush=True)

    def save(self, name: str) -> Path:
        state = {"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                 "optimizer": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                 "step": self.step, "epoch": self.epoch, "best": self.best, "amp": self.amp,
                 "config_snapshot": self.snapshot, "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                                            "python": random.getstate()}}
        p = self.run_dir / "checkpoints" / name
        tmp = p.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(p)
        return p

    def load(self, path: Path) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.scaler.load_state_dict(st["scaler"])
        self.step, self.epoch, self.best = st["step"], st["epoch"], st["best"]
        torch.set_rng_state(st["rng"]["torch"])
        np.random.set_state(st["rng"]["numpy"])
        random.setstate(st["rng"]["python"])
        self.log(f"从 {path} 恢复：step={self.step} epoch={self.epoch}")

    # ------------------------------------------------------------------ 调度
    def lr_scale(self) -> float:
        w = self.cfg["warmup_steps"]
        if self.step < w:
            return (self.step + 1) / w
        p = (self.step - w) / max(self.total_steps - w, 1)
        m = self.cfg["min_lr_ratio"]
        return m + (1 - m) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    def loader(self, ds, shuffle: bool, n: int | None = None) -> DataLoader:
        nw = self.cfg["num_workers"]
        max_pairs = self.cfg.get("max_pairs_per_batch") if shuffle else None
        cfn = partial(collate, max_pairs=max_pairs) if max_pairs is not None else collate
        kw = {"num_workers": nw, "collate_fn": cfn, "pin_memory": self.dev.type == "cuda",
              "persistent_workers": False}
        if nw > 0:
            kw["prefetch_factor"] = self.cfg.get("prefetch_factor", 2)
        if n is not None:
            ds = torch.utils.data.Subset(ds, list(range(min(n, len(ds)))))
        return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=shuffle, drop_last=shuffle, **kw)

    def close(self) -> None:
        for f in (self.log_f, self.met_f):
            if not f.closed:
                f.close()

    # ------------------------------------------------------------------ 主循环
    def fit(self) -> dict:
        self.log(f"开始训练 {self.stage}：设备 {device_info(self.dev)}，AMP={self.amp}，参数 {count_params(self.model) / 1e6:.1f}M，"
                 f"总步数 {self.total_steps}，训练样本 {len(self.train_ds)}，验证样本 {len(self.val_ds)}")
        ad = amp_dtype(self.amp)
        accum = self.cfg["grad_accum"]
        t_start = time.time()
        start_step = self.step
        seen = 0
        recent: list[dict] = []
        done = self.step >= self.total_steps
        while not done:
            self.train_ds.set_epoch(self.epoch)
            self.model.train()
            micro = 0
            for batch in self.loader(self.train_ds, shuffle=True):
                batch = batch_to(batch, self.dev)
                total, parts, _, _ = train_step(self.model, batch, self.crit, self.loss_cfg, self.dev, ad)
                if not torch.isfinite(total):
                    self.log(f"警告：step {self.step} 损失非有限，跳过该批并重置梯度累积计数")
                    self.opt.zero_grad(set_to_none=True)
                    micro = 0
                    continue
                self.scaler.scale(total / accum).backward()
                micro += 1
                seen += int(batch["bev"].shape[0])
                recent.append(parts)
                if micro % accum:
                    continue
                for g, base in zip(self.opt.param_groups, self.base_lr):
                    g["lr"] = base * self.lr_scale()
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % self.cfg["log_every_steps"] == 0 or self.step == 1:
                    self._progress(recent, seen, t_start, float(gn), start_step)
                    recent = []
                if self.step % self.cfg["ckpt_every_steps"] == 0:
                    self.save("last.pt")
                if self.step % self.cfg["val_every_steps"] == 0:
                    self.validate()
                if self.step >= self.total_steps:
                    done = True
                    break
            self.epoch += 1
        self.save("last.pt")
        v = self.last_val if (self.last_val and self.step % self.cfg["val_every_steps"] == 0) else self.validate()
        try:
            self.figures(self.cfg.get("figures", 6))
        except Exception as e:  # noqa: BLE001
            self.log(f"验证样例叠图生成失败：{e!r}")
        summary = {"steps": self.step, "epochs": self.epoch, "seconds": time.time() - t_start, "best_val": self.best,
                   "final_val": v}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        self.log(f"训练结束：{summary}")
        return summary

    def _progress(self, recent, seen, t_start, gn, start_step: int = 0) -> None:
        avg = {k: float(np.mean([r[k] for r in recent if k in r])) for k in recent[-1]} if recent else {}
        el = time.time() - t_start
        sps = seen / max(el, 1e-6)
        done_steps = max(self.step - start_step, 1)
        eta = max(self.total_steps - self.step, 0) * el / done_steps
        mem = memory_gib(self.dev)
        rec = {"step": self.step, "epoch": self.epoch, "lr": self.opt.param_groups[1]["lr"], "grad_norm": gn,
               "samples_per_s": sps, "elapsed_s": el, "eta_s": eta, "mem_gib": mem,
               "loss": sum(avg.values()), **{f"loss/{k}": v for k, v in avg.items()}}
        if self.dev.type == "cuda" and self.step % (self.cfg["log_every_steps"] * 5) == 0:
            rec.update(gpu_util())
        self.met_f.write(json.dumps(rec) + "\n")
        self.met_f.flush()
        head = " ".join(f"{k}={avg[k]:.3f}" for k in HEADLINE if k in avg)
        vtxt = f" | val {self.last_val.get('val_loss', float('nan')):.3f} F1 {self.last_val.get('lane_f1', float('nan')):.3f}" \
            if self.last_val else ""
        gpu = f" util {rec['gpu_util']:.0f}%" if "gpu_util" in rec else ""
        self.log(f"[{self.stage}] ep {self.epoch} step {self.step}/{self.total_steps} loss {rec['loss']:.3f} ({head}) "
                 f"lr {rec['lr']:.2e} {sps:.2f} 样本/s 已用 {_hms(el)} ETA {_hms(eta)} 显存 {mem:.1f}GiB{gpu}{vtxt}")

    # ------------------------------------------------------------------ 验证
    @torch.no_grad()
    def validate(self) -> dict:
        from src.eval.quick import quick_metrics
        self.model.eval()
        ad = amp_dtype(self.amp)
        losses, qm = [], []
        for batch in self.loader(self.val_ds, shuffle=False, n=self.cfg["val_samples"]):
            batch = batch_to(batch, self.dev)
            total, parts, fo, matches = train_step(self.model, batch, self.crit, self.loss_cfg, self.dev, ad)
            losses.append(float(total))
            qm.append(quick_metrics(fo, batch, matches[-1]))
        self.model.train()
        v = {"val_loss": float(np.mean(losses)) if losses else float("nan")}
        if qm:
            for k in qm[0]:
                vals = [q[k] for q in qm if q[k] == q[k]]
                v[k] = float(np.mean(vals)) if vals else float("nan")
        self.last_val = v
        self.met_f.write(json.dumps({"step": self.step, "epoch": self.epoch, **{f"val/{k}": x for k, x in v.items()}})
                         + "\n")
        self.met_f.flush()
        if v["val_loss"] < self.best:
            self.best = v["val_loss"]
            self.save("best.pt")
        self.log(f"验证 step {self.step}: " + " ".join(f"{k}={x:.4f}" for k, x in v.items()))
        return v


@torch.no_grad()
def _figures(self, n: int) -> None:
    """验证样例底图叠图（约束解码图 + 真值虚线），写入 runs/<run_id>/figures/。"""
    from src.eval.overlay import render_overlay
    from src.infer.predict import predict_batch
    out = self.run_dir / "figures"
    k = 0
    for batch in self.loader(self.val_ds, shuffle=False, n=n):
        preds, _ = predict_batch(self.model, batch, self.dev)
        for b, (p, m) in enumerate(zip(preds, batch["meta"])):
            render_overlay(out / f"val_{k:03d}_step{self.step}.png", batch["bev"][b].float().numpy(), batch["hw"][b].tolist(),
                           p.result.graph if p.ok else None, m["graph"],
                           title=f"val #{k} step {self.step} decode={'ok' if p.ok else p.error[:60]}")
            k += 1
    self.model.train()


Trainer.figures = _figures


def _hms(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_checkpoint_model(path: Path, device: str | None = None):
    """加载 checkpoint（CPU state_dict）到当前设备，返回 (model, ckpt)。"""
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st.get("config_snapshot", {}).get("configs", {}).get("model")
    m = build_model(cfg)
    m.load_state_dict(st["model"])
    dev = select_device(device)
    synchronize(dev)
    return m.to(dev).eval(), st
