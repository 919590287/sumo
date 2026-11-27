"""合成轻量矢量库：生成（含 netconvert 往返一致性编译校验）、分片存储与读取（docs/DESIGN.md §5 Layer 1-3）。

目录结构（位于 APFS 缓存 ``cache_dir/<library_dir>``，训练包内由 ``B2S_SYNTH_DIR`` 重写）::

    <lib>/<split>/shard_00000.pkl      # list[bytes]，每项为 codec.encode_sample 的输出
    <lib>/<split>/index.json           # 分片清单、逐样本种子 / 场景类型 / 规模
    <lib>/<split>/stats.json           # 生成统计（接受率、编译失败原因、耗时）
    <lib>/MANIFEST.json                # 版本、配置哈希、各 split 样本数与 sha256
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
import traceback
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from src.common.config import cache_dir, load_config
from src.data.synth.codec import SampleRecord, decode_sample, encode_sample


def library_root() -> Path:
    """合成库根目录：B2S_SYNTH_DIR > configs/paths.yaml 的 synth_dir（训练包内重写为包内相对路径）> 缓存目录。"""
    env = os.environ.get("B2S_SYNTH_DIR")
    if env:
        return Path(env)
    if "synth_dir" in load_config("paths"):
        from src.common.config import path_setting
        return path_setting("synth_dir")
    return cache_dir(load_config("synth")["library_dir"])


def config_hash() -> str:
    cfg = {k: load_config(k) for k in ("synth", "graph", "data", "sumo_export")}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# ====================================================================== 生成
def generate_one(seed: int, work: Path, compile_check: bool = True, force_kind: str | None = None):
    """生成单个通过往返一致性编译校验的合成样本；返回 (bytes | None, info)。"""
    from src.data.synth.render import render_scene
    from src.data.synth.scene import SceneGenerator, SceneReject
    gen = _GEN.get("gen") or _GEN.setdefault("gen", SceneGenerator())
    info = {"seed": seed}
    t0 = time.time()
    try:
        if force_kind:
            gen.cfg = dict(gen.cfg, scene_mix={force_kind: 1.0})
        sc = gen.generate(seed)
    except SceneReject as e:
        info["fail"] = "scene:" + str(e)[:60]
        return None, info
    info["t_gen"] = time.time() - t0
    g = sc.graph
    if compile_check:
        from src.sumo_io.pipeline import compile_graph
        from src.sumo_io.roundtrip import classify_warnings
        d = work / f"s{seed}"
        try:
            r = compile_graph(g, d, with_eval_sim=False)
            _, unexpected = classify_warnings(g, r.plain.ids, r.site.warnings)
            ok = r.ok and not unexpected
            if not ok:
                why = (r.roundtrip.failures[:1] if r.roundtrip else r.site.errors[:1]) or unexpected[:1]
                info["fail"] = "compile:" + (str(why[0])[:60] if why else "?")
                return None, info
            info["hausdorff"] = max(r.roundtrip.lane_hausdorff_max, r.roundtrip.conn_hausdorff_max)
        except Exception as e:  # noqa: BLE001
            info["fail"] = "compile_exc:" + repr(e)[:60]
            return None, info
        finally:
            shutil.rmtree(d, ignore_errors=True)
    info["t_compile"] = time.time() - t0 - info["t_gen"]
    rng = np.random.default_rng([seed, 7])
    fs, meta = render_scene(sc, gen, rng)
    meta.update({"seed": seed, "kind": sc.kind, "attempt": sc.meta.get("attempt", 0)})
    blob = encode_sample(g, fs, meta)
    info.update({"kind": sc.kind, "bytes": len(blob), "lanes": len(g.lanes), "conns": len(g.connections),
                 "nodes": len(g.nodes), "cws": len(g.crosswalks), "t_total": time.time() - t0})
    return blob, info


_GEN: dict = {}


def _worker(args):
    seeds, work, compile_check, force_kind = args
    out = []
    for s in seeds:
        try:
            out.append(generate_one(s, Path(work), compile_check, force_kind))
        except Exception as e:  # noqa: BLE001
            out.append((None, {"seed": s, "fail": "exc:" + repr(e)[:80], "tb": traceback.format_exc()[-400:]}))
    return out


def build_split(split: str, count: int, root: Path | None = None, workers: int = 8, compile_check: bool = True,
                force_kind: str | None = None, seed_base: int | None = None, progress=print) -> dict:
    """生成 ``count`` 个合格样本（种子从 seed_base 递增，失败种子跳过），写入分片。"""
    import multiprocessing as mp
    cfg = load_config("synth")
    root = Path(root or library_root())
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    base = cfg["seeds"].get(split, 0) if seed_base is None else seed_base
    work = cache_dir("synth_work", split)
    shard_size = cfg["shard_size"]
    shard, index, fails = [], [], Counter()
    n_shard, next_seed, t0 = 0, base, time.time()
    chunk = 4
    ctx = mp.get_context("spawn")
    total_bytes = 0
    with ctx.Pool(workers) as pool:
        while len(index) < count:
            need = count - len(index)
            n_jobs = max(workers, int(need * 1.25 / chunk) + 1)
            jobs = []
            for _ in range(min(n_jobs, workers * 8)):
                jobs.append((list(range(next_seed, next_seed + chunk)), str(work), compile_check, force_kind))
                next_seed += chunk
            for res in pool.imap(_worker, jobs):
                for blob, info in res:
                    if blob is None:
                        fails[info.get("fail", "?").split(":")[0] + ":" + info.get("fail", "?").split(":", 1)[-1][:40]] += 1
                        continue
                    if len(index) >= count:
                        break
                    shard.append(blob)
                    total_bytes += len(blob)
                    index.append({k: info[k] for k in ("seed", "kind", "lanes", "conns", "nodes", "cws", "bytes")}
                                 | {"shard": n_shard, "k": len(shard) - 1})
                    if len(shard) >= shard_size:
                        _write_shard(out_dir, n_shard, shard)
                        shard, n_shard = [], n_shard + 1
                if len(index) and len(index) % 200 < chunk * workers:
                    el = time.time() - t0
                    progress(f"[{split}] {len(index)}/{count}  {len(index) / el:.1f} 样本/s  失败 {sum(fails.values())}"
                             f"  平均 {total_bytes / max(len(index), 1) / 1024:.1f} KB")
    if shard:
        _write_shard(out_dir, n_shard, shard)
        n_shard += 1
    stats = {"split": split, "count": len(index), "seed_base": base, "seed_end": next_seed,
             "attempted": len(index) + sum(fails.values()), "fails": dict(fails.most_common()),
             "seconds": time.time() - t0, "bytes_total": total_bytes,
             "bytes_mean": total_bytes / max(len(index), 1), "kinds": dict(Counter(r["kind"] for r in index)),
             "config_hash": config_hash(), "compile_check": compile_check, "force_kind": force_kind}
    (out_dir / "index.json").write_text(json.dumps({"n_shards": n_shard, "samples": index}), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    write_manifest(root)
    return stats


def _write_shard(out_dir: Path, k: int, blobs: list[bytes]) -> None:
    tmp = out_dir / f"shard_{k:05d}.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blobs, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out_dir / f"shard_{k:05d}.pkl")


def write_manifest(root: Path) -> dict:
    root = Path(root)
    man = {"version": load_config("synth")["version"], "config_hash": config_hash(), "splits": {}}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        idx = d / "index.json"
        if not idx.exists():
            continue
        files = sorted(d.glob("shard_*.pkl"))
        man["splits"][d.name] = {
            "count": len(json.loads(idx.read_text(encoding="utf-8"))["samples"]),
            "shards": {f.name: sha256_file(f) for f in files},
        }
    (root / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def sha256_file(p: Path, bs: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ====================================================================== 读取
class SynthLibrary:
    """按需加载分片（每进程 LRU 缓存若干分片）。"""

    def __init__(self, split: str, root: Path | None = None, cache_shards: int = 4):
        self.root = Path(root or library_root())
        self.dir = self.root / split
        idx = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.samples = idx["samples"]
        self.cache: OrderedDict[int, list[bytes]] = OrderedDict()
        self.cache_shards = cache_shards

    def __len__(self) -> int:
        return len(self.samples)

    def raw(self, i: int) -> bytes:
        s = self.samples[i]
        k = s["shard"]
        if k not in self.cache:
            with open(self.dir / f"shard_{k:05d}.pkl", "rb") as f:
                self.cache[k] = pickle.load(f)
            while len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(k)
        return self.cache[k][s["k"]]

    def __getitem__(self, i: int) -> SampleRecord:
        return decode_sample(self.raw(i))
