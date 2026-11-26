"""netconvert 编译接口（交付网 site.net.xml 与测试台副本 eval_sim.net.xml）。"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import load_config, sumo_tool


@dataclass
class CompileResult:
    ok: bool
    net_path: Path
    returncode: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    log: str = ""


def _parse_log(text: str) -> tuple[list[str], list[str]]:
    warnings, errors = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Warning:"):
            warnings.append(s)
        elif s.startswith("Error:") or s.startswith("Quitting"):
            errors.append(s)
    return warnings, errors


def run_netconvert(paths: dict[str, Path], out_net: Path, tls_set: list[str] | None = None,
                   extra_args: list[str] | None = None, timeout: float = 120.0) -> CompileResult:
    cfg = load_config("sumo_export")
    cmd = [sumo_tool("netconvert"),
           "-n", str(paths["nod"]), "-e", str(paths["edg"]), "-x", str(paths["con"]),
           *cfg["netconvert_args"], "-o", str(out_net)]
    if tls_set:
        cmd.append(f"--tls.set={','.join(tls_set)}")
    if extra_args:
        cmd.extend(extra_args)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log = (p.stdout or "") + (p.stderr or "")
    warnings, errors = _parse_log(log)
    ok = p.returncode == 0 and not errors and Path(out_net).exists()
    return CompileResult(ok=ok, net_path=Path(out_net), returncode=p.returncode,
                         warnings=warnings, errors=errors, log=log)
