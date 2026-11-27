"""训练包环境核验（docs/DESIGN.md §6.2-3 第 2 步）：已装版本与依赖清单一致、CUDA 可用、GPU 上完成一次前反向。"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.manifest).parent
    want = {}
    for f in ("env/requirements-torch.txt", "env/requirements.txt"):
        for line in (root / f).read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
            if m:
                want[m.group(1)] = m.group(2)
    rep = {"mismatch": {}, "ok": True}
    for k, v in want.items():
        try:
            got = md.version(k)
        except md.PackageNotFoundError:
            got = None
        if got is None or got.split("+")[0] != v.split("+")[0]:
            rep["mismatch"][k] = {"want": v, "got": got}
    import torch
    rep["torch"] = torch.__version__
    rep["torch_cuda"] = torch.version.cuda
    rep["cuda_available"] = torch.cuda.is_available()
    if rep["cuda_available"]:
        x = torch.randn(256, 256, device="cuda", requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        rep["gpu"] = torch.cuda.get_device_name(0)
        rep["gpu_fwd_bwd"] = bool(torch.isfinite(x.grad).all())
    rep["ok"] = not rep["mismatch"] and rep["cuda_available"] and rep.get("gpu_fwd_bwd", False)
    rep["manifest_commit"] = man.get("git", {}).get("commit")
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if not rep["ok"]:
        print("环境核验未通过", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
