import os
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from src.common.config import path_setting  # noqa: E402


def _has_sumo() -> bool:
    return (path_setting("sumo_bin") / "netconvert").exists() and (path_setting("sumo_bin") / "sumo").exists()


def _has_gpkg() -> bool:
    return path_setting("gpkg_dev").exists()


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip = {
        "sumo": (_has_sumo(), "缺少 netconvert / sumo"),
        "gpkg": (_has_gpkg(), "缺少开创大道 GPKG"),
        "mps": (_has_mps(), "缺少 MPS 设备"),
    }
    for item in items:
        for mark, (ok, why) in skip.items():
            if mark in item.keywords and not ok:
                item.add_marker(pytest.mark.skip(reason=why))


@pytest.fixture(scope="session")
def standard_graphs():
    from src.data.standard_graphs import build_all
    return build_all()


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    return tmp_path
