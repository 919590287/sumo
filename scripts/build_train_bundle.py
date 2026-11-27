"""构建一键训练包（docs/DESIGN.md §6.2-1/2）。

用法（仓库根目录，本机）：
    /opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.build_train_bundle --stage layer1 --out ~/cache/baidu2sumo/bundles
        [--allow-dirty] [--skip-tests] [--splits train,val,test,tile_test] [--max-train N] [--no-tar]

前置条件：本机单元测试全部通过（构建时执行；--skip-tests 仅用于构建脚本自身的单测）、冒烟训练通过
（runs/smoke/smoke_<git短哈希>.json，pass=true 且提交号一致）、工作区已提交（--allow-dirty 例外并在 MANIFEST 中标记）。
包内只放代码、数据、依赖清单与脚本，不预先打包运行环境。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

from src.common.config import REPO_ROOT, cache_dir, load_config  # noqa: E402
from src.data.synth.library import config_hash, library_root, sha256_file  # noqa: E402

CODE_DIRS = ("src", "configs", "scripts", "tests")
CODE_FILES = ("AGENTS.md", "README.md", "pyproject.toml", "docs/DESIGN.md")
# 依赖清单：包名 → 服务器安装名（macOS 专属 / GUI 依赖剔除，OpenCV 换无头构建，SUMO 换 Linux 轮子）
REQS = [("numpy", "numpy"), ("scipy", "scipy"), ("shapely", "shapely"), ("geopandas", "geopandas"),
        ("pyogrio", "pyogrio"), ("pyproj", "pyproj"), ("pandas", "pandas"),
        ("opencv-python", "opencv-python-headless"), ("PyYAML", "PyYAML"), ("pytest", "pytest"), ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"), ("sumolib", "sumolib"), ("traci", "traci")]
SUMO_VERSION = "1.28.0"
TORCH_CUDA = {"tag": "cu128", "cuda": "12.8", "index_url": "https://download.pytorch.org/whl/cu128"}


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()


def code_files() -> list[str]:
    files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    out = [f for f in files if (f.split("/")[0] in CODE_DIRS or f in CODE_FILES)
           and not os.path.basename(f).startswith("._") and (REPO_ROOT / f).is_file()]
    return sorted(set(out))


def requirements() -> tuple[str, str, dict]:
    vers = {}
    lines = []
    for src, dst in REQS:
        v = md.version(src)
        lines.append(f"{dst}=={v}")
        vers[dst] = v
    lines += [f"eclipse-sumo=={SUMO_VERSION}"]
    vers["eclipse-sumo"] = SUMO_VERSION
    tv = md.version("torch").split("+")[0]
    tvv = md.version("torchvision").split("+")[0]
    torch_req = f"torch=={tv}\ntorchvision=={tvv}\n"
    vers.update({"torch": tv, "torchvision": tvv})
    return torch_req, "\n".join(lines) + "\n", vers


def preflight(args, head: str, dirty: bool) -> dict:
    pre = {"dirty": dirty}
    if dirty and not args.allow_dirty:
        raise SystemExit("工作区有未提交改动：请先提交，或显式 --allow-dirty（将在 MANIFEST 中标记）")
    smoke = REPO_ROOT / "runs" / "smoke" / f"smoke_{head}.json"
    if smoke.exists():
        rep = json.loads(smoke.read_text(encoding="utf-8"))
        pre["smoke"] = {"file": str(smoke.relative_to(REPO_ROOT)), "pass": rep.get("pass"), "git": rep.get("git"),
                        "dirty_at_smoke": rep.get("dirty")}
        if not rep.get("pass"):
            raise SystemExit(f"冒烟训练未通过：{smoke}")
    elif not args.allow_dirty:
        raise SystemExit(f"缺少当前提交 {head} 的冒烟训练记录：先运行 python -B -m scripts.smoke_train")
    else:
        pre["smoke"] = {"missing": True}
    if args.skip_tests:
        pre["tests"] = "skipped(--skip-tests)"
        if not args.allow_dirty:
            raise SystemExit("--skip-tests 仅允许与 --allow-dirty 一起使用（构建脚本自测）")
    else:
        print("运行本机单元测试 ...", flush=True)
        r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        pre["tests"] = {"returncode": r.returncode, "tail": tail}
        if r.returncode != 0:
            raise SystemExit("本机单元测试未全部通过：\n" + "\n".join(tail))
    return pre


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="layer1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--splits", default="train,val,test,tile_test")
    ap.add_argument("--max-train", type=int, default=None, help="仅用于构建脚本自测：截取训练分片数")
    ap.add_argument("--no-tar", action="store_true")
    args = ap.parse_args()
    head = git("rev-parse", "--short", "HEAD")
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    pre = preflight(args, head, dirty)
    date = time.strftime("%Y%m%d")
    name = f"b2s_train_{args.stage}_{head}{'_dirty' if dirty else ''}_{date}"
    out_root = Path(os.path.expanduser(args.out)) if args.out else cache_dir("bundles")
    bdir = out_root / name
    if bdir.exists():
        shutil.rmtree(bdir)
    (bdir / "env").mkdir(parents=True)
    (bdir / "runs").mkdir()
    (bdir / "cache").mkdir()
    # ---------------- 代码
    files = code_files()
    for f in files:
        dst = bdir / "code" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / f, dst)
    paths = (bdir / "code/configs/paths.yaml")
    paths.write_text("# 训练包内路径（构建时重写为包内相对路径；运行时环境变量 B2S_* 仍可覆盖）\n"
                     "cache_dir: \"../cache\"\nsynth_dir: \"../data/synth_v1\"\nruns_dir: \"../runs\"\n"
                     "sumo_bin: \"../cache/sumo_bin_unset\"   # run_train.sh 以 eclipse-sumo 轮子路径设置 B2S_SUMO_BIN\n"
                     "gpkg_city: \"\"\ngpkg_dev: \"\"\nlegacy_eval_dir: \"\"\ndistricts_geojson: \"\"\n", encoding="utf-8")
    for sh in ("run_train.sh", "watch.sh", "pack_results.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / "bundle" / sh, bdir / sh)
        os.chmod(bdir / sh, 0o755)
    # ---------------- 数据
    lib = library_root()
    synth_man = json.loads((lib / "MANIFEST.json").read_text(encoding="utf-8"))
    for sp in args.splits.split(","):
        src = lib / sp
        dst = bdir / "data" / "synth_v1" / sp
        dst.mkdir(parents=True)
        idx = json.loads((src / "index.json").read_text(encoding="utf-8"))
        shards = sorted(src.glob("shard_*.pkl"))
        if args.max_train and sp == "train":
            shards = shards[: args.max_train]
            keep = {int(p.stem.split("_")[1]) for p in shards}
            idx = {"n_shards": len(shards), "samples": [s for s in idx["samples"] if s["shard"] in keep]}
        for p in shards:
            shutil.copy2(p, dst / p.name)
        (dst / "index.json").write_text(json.dumps(idx), encoding="utf-8")
        shutil.copy2(src / "stats.json", dst / "stats.json")
    from src.data.synth.library import write_manifest
    write_manifest(bdir / "data" / "synth_v1")
    # ---------------- 依赖清单
    torch_req, req, vers = requirements()
    (bdir / "env" / "environment.yml").write_text(
        "name: baidu2sumo\nchannels:\n  - conda-forge\n  - defaults\ndependencies:\n  - python=3.12\n  - pip\n",
        encoding="utf-8")
    (bdir / "env" / "requirements-torch.txt").write_text(torch_req, encoding="utf-8")
    (bdir / "env" / "requirements.txt").write_text(req, encoding="utf-8")
    # ---------------- MANIFEST + SHA256SUMS
    sums = {}
    for p in sorted(bdir.rglob("*")):
        if p.is_file():
            sums[str(p.relative_to(bdir))] = sha256_file(p)
    man = {
        "name": name, "stage": args.stage, "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"commit": commit, "short": head, "dirty": dirty, "allow_dirty": bool(args.allow_dirty)},
        "preflight": pre, "config_hash": config_hash(),
        "configs_sha256": {k[5:]: v for k, v in sums.items() if k.startswith("code/configs/")},
        "torch": {"version": vers["torch"], "torchvision": vers["torchvision"], **TORCH_CUDA},
        "requirements": vers, "sumo": SUMO_VERSION,
        "data": {"synth_version": synth_man.get("version"), "synth_config_hash": synth_man.get("config_hash"),
                 "splits": {sp: json.loads((bdir / "data/synth_v1/MANIFEST.json").read_text())["splits"][sp]["count"]
                            for sp in args.splits.split(",")},
                 "split_manifest": load_config("splits")["version"]},
        "train_cfg": load_config("train")[args.stage],
        "env_sha256": {k: v for k, v in sums.items() if k.startswith("env/")},
        "n_files": len(sums),
    }
    (bdir / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    sums["MANIFEST.json"] = sha256_file(bdir / "MANIFEST.json")
    (bdir / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())), encoding="utf-8")
    size = sum(p.stat().st_size for p in bdir.rglob("*") if p.is_file())
    print(f"训练包目录：{bdir}（{size / 2 ** 30:.2f} GiB，{len(sums)} 个文件，提交 {head}{'，含未提交改动' if dirty else ''}）")
    if not args.no_tar:
        tar = out_root / f"{name}.tar.gz"
        with tarfile.open(tar, "w:gz", compresslevel=1) as tf:
            tf.add(bdir, arcname=name)
        print(f"训练包：{tar}（{tar.stat().st_size / 2 ** 30:.2f} GiB）")
        print(f"上传服务器后：tar xzf {tar.name} && cd {name} && ./run_train.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
