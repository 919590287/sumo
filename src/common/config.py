"""配置加载：configs/*.yaml + 环境变量覆盖（纯 I/O，无任何几何裁决）。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """读取 configs/<name>.yaml（带缓存，返回只读语义的 dict）。"""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def path_setting(key: str) -> Path:
    """读取 configs/paths.yaml 中的路径项，允许 B2S_<KEY> 环境变量覆盖，并展开 ~。"""
    env = os.environ.get(f"B2S_{key.upper()}")
    raw = env if env else load_config("paths")[key]
    p = Path(os.path.expanduser(str(raw)))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def cache_dir(*parts: str) -> Path:
    """APFS 缓存目录（自动创建）。"""
    d = path_setting("cache_dir").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sumo_tool(name: str) -> str:
    """返回 SUMO 可执行文件路径（netconvert / sumo）。"""
    p = path_setting("sumo_bin") / name
    if not p.exists():
        raise FileNotFoundError(f"找不到 SUMO 工具 {name}: {p}（可用 B2S_SUMO_BIN 覆盖）")
    return str(p)
