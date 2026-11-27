"""设备选择与设备无关工具（docs/DESIGN.md §6.1-2）：B2S_DEVICE ∈ {auto, cuda, mps, cpu}，auto 顺序 cuda → mps → cpu。"""
from __future__ import annotations

import os
import platform
import subprocess

import torch


def select_device(pref: str | None = None) -> torch.device:
    pref = (pref or os.environ.get("B2S_DEVICE") or "auto").lower()
    if pref not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"B2S_DEVICE 取值非法：{pref}")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B2S_DEVICE=cuda 但 CUDA 不可用")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("B2S_DEVICE=mps 但 MPS 不可用")
    return torch.device(pref)


def synchronize(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def memory_gib(dev: torch.device) -> float:
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2 ** 30
    if dev.type == "mps":
        return torch.mps.driver_allocated_memory() / 2 ** 30
    return 0.0


def gpu_util() -> dict:
    """nvidia-smi 只读查询（利用率 / 显存）；非 NVIDIA 环境返回空。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        u, used, tot = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return {"gpu_util": u, "gpu_mem_used_mib": used, "gpu_mem_total_mib": tot}
    except Exception:  # noqa: BLE001
        return {}


def device_info(dev: torch.device) -> dict:
    info = {"device": str(dev), "torch": torch.__version__, "platform": platform.platform(),
            "python": platform.python_version()}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "vram_gib": round(p.total_memory / 2 ** 30, 2), "cuda": torch.version.cuda,
                     "capability": f"{p.major}.{p.minor}"})
    elif dev.type == "mps":
        info.update({"gpu": "Apple MPS"})
    return info


def batch_to(batch: dict, dev: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=dev.type == "cuda")
        else:
            out[k] = v
    return out


def amp_dtype(mode: str) -> torch.dtype | None:
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
