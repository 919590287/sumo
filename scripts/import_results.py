"""导入服务器训练成果（docs/DESIGN.md §6.2-5）。

用法：
    /opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.import_results results_<run_id>.tar.gz [--eval] [--n 1000]

步骤：校验 SHA256SUMS 与 MANIFEST（训练包提交号须为本机仓库已有提交，与 HEAD 不同时给出警告）→
checkpoint 放入 APFS 缓存（cache_dir/checkpoints/<run_id>/）→ 轻量报告与叠图放入仓库 runs/<run_id>/ →
本机跨设备前向一致性检查（CUDA 参考输出 vs 本机 MPS / CPU；fp32）→（可选）在 MPS 上执行 M1 合成评测。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from src.common.config import REPO_ROOT, cache_dir  # noqa: E402

TOL_REL = 1e-3


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(root: Path) -> list[str]:
    bad = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        p = root / rel
        if not p.exists() or sha256(p) != digest:
            bad.append(rel)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    tb = Path(args.tarball).expanduser().resolve()
    stage = cache_dir("import_tmp")
    with tarfile.open(tb) as tf:
        names = tf.getnames()
        top = names[0].split("/")[0]
        for n in names:                                  # 防路径穿越
            if n.startswith("/") or ".." in Path(n).parts:
                raise SystemExit(f"成果包含非法路径：{n}")
        tf.extractall(stage, filter="data")
    root = stage / top
    bad = verify(root)
    if bad:
        raise SystemExit(f"sha256 校验失败：{bad[:5]}")
    man = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    commit = man["git"]["commit"]
    known = subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=REPO_ROOT).returncode == 0
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    if not known:
        raise SystemExit(f"训练包提交 {commit} 不在本机仓库中，无法确认代码一致性")
    if commit != head:
        print(f"警告：训练包提交 {commit[:8]} ≠ 本机 HEAD {head[:8]}（模型结构若有变化将无法加载）")
    if man["git"].get("dirty"):
        print("警告：训练包构建时工作区含未提交改动（MANIFEST 已标记）")
    run_id = top.removeprefix("results_")
    ck_dir = cache_dir("checkpoints", run_id)
    for p in (root / "run" / "checkpoints").glob("*.pt"):
        shutil.copy2(p, ck_dir / p.name)
    rep_dir = REPO_ROOT / "runs" / run_id
    rep_dir.mkdir(parents=True, exist_ok=True)
    for f in ("config_snapshot.json", "calibration.json", "metrics.jsonl", "summary.json", "train.log"):
        if (root / "run" / f).exists():
            shutil.copy2(root / "run" / f, rep_dir / f)
    if (root / "run" / "figures").exists():
        shutil.copytree(root / "run" / "figures", rep_dir / "figures", dirs_exist_ok=True)
    shutil.copy2(root / "MANIFEST.json", rep_dir / "bundle_MANIFEST.json")
    best = ck_dir / ("best.pt" if (ck_dir / "best.pt").exists() else "last.pt")
    # ---------------- 跨设备前向一致性
    from scripts.export_reference import compare, forward_outputs
    from src.common.device import select_device
    from src.train.trainer import load_checkpoint_model
    cons = {"checkpoint": str(best)}
    dev = select_device(args.device)
    model, ck = load_checkpoint_model(best, args.device)
    local = forward_outputs(model, dev)
    cpu_model, _ = load_checkpoint_model(best, "cpu")
    cons["local_vs_cpu"] = compare(forward_outputs(cpu_model, select_device("cpu")), local)
    ref = root / "run" / "reference_cuda.npz"
    if ref.exists():
        r = dict(np.load(ref))
        cons["cuda_vs_local"] = compare({k: r[k] for k in local}, local)
    checks = [v["rel"] for d in (cons.get("cuda_vs_local", {}), cons["local_vs_cpu"]) for v in d.values()]
    cons["max_rel"] = float(max(checks)) if checks else None
    cons["pass"] = bool(checks) and max(checks) <= TOL_REL
    cons["device"] = str(dev)
    cons["step"] = ck.get("step")
    (rep_dir / "import_consistency.json").write_text(json.dumps(cons, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "checkpoints": str(ck_dir), "reports": str(rep_dir),
                      "consistency_pass": cons["pass"], "max_rel": cons["max_rel"]}, ensure_ascii=False, indent=1))
    shutil.rmtree(stage, ignore_errors=True)
    if not cons["pass"]:
        print("跨设备前向一致性未通过：不得用于评测（见 import_consistency.json）")
        return 1
    if args.eval:
        return subprocess.run([sys.executable, "-B", "-m", "scripts.eval_synth", "--ckpt", str(best), "--n", str(args.n),
                               "--tiles", "--out", f"runs/{run_id}/eval_synth"], cwd=REPO_ROOT).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
