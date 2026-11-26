"""生成 ≥ 500 m 隔离带的跨区空间切分清单 configs/splits/split_v1.json（只读访问全市 GPKG）。

用法：/opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.build_split_manifest
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time

import numpy as np
import pyogrio

from src.common.config import REPO_ROOT, load_config, path_setting
from src.data.roi import wgs84_to_utm
from src.data.splits import SplitManifest, assign_blocks


def main() -> int:
    cfg = load_config("splits")
    bs = float(cfg["block_size_m"])
    city = path_setting("gpkg_city")
    t0 = time.time()
    import geopandas as gpd
    ref = pyogrio.read_dataframe(city, layer=cfg["district_source_layer"], columns=["分类代码"])
    c = ref.geometry.centroid
    rx, ry = wgs84_to_utm(c.x.values, c.y.values)
    # GPKG 内仅南沙 / 番禺填有 行政区名称，故以 11 区行政边界对道路参考线质心做空间连接
    districts = gpd.read_file(path_setting("districts_geojson"))[["name", "geometry"]]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(c.x.values, c.y.values), crs="EPSG:4326")
    j = gpd.sjoin(pts, districts, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    dist_name = j["name"].fillna("").values
    sig = pyogrio.read_dataframe(city, layer=cfg["density_source_layer"], columns=["分类代码"])
    sx, sy = wgs84_to_utm(sig.geometry.x.values, sig.geometry.y.values)
    print(f"读取 {len(ref)} 条道路参考线 + {len(sig)} 个信号灯点位，用时 {time.time() - t0:.1f}s", flush=True)

    ox = math.floor(min(rx.min(), sx.min()) / bs) * bs
    oy = math.floor(min(ry.min(), sy.min()) / bs) * bs
    bix = np.floor((rx - ox) / bs).astype(int)
    biy = np.floor((ry - oy) / bs).astype(int)
    votes: dict = collections.defaultdict(collections.Counter)
    for i, j, d in zip(bix, biy, dist_name):
        if d:
            votes[(int(i), int(j))][d] += 1
    block_district = {b: v.most_common(1)[0][0] for b, v in votes.items()}
    weight = collections.Counter(zip(np.floor((sx - ox) / bs).astype(int).tolist(),
                                     np.floor((sy - oy) / bs).astype(int).tolist()))
    # 开创大道 dev 样本外包框（外扩）覆盖的区块 → dev
    info = pyogrio.read_info(path_setting("gpkg_dev"), layer="南沙道路要素")
    b = info["total_bounds"]
    x0, y0 = wgs84_to_utm(b[0], b[1])
    x1, y1 = wgs84_to_utm(b[2], b[3])
    m = cfg["dev_gpkg_bounds_buffer_m"]
    dev_blocks = {(i, j) for i in range(int((x0 - m - ox) // bs), int((x1 + m - ox) // bs) + 1)
                  for j in range(int((y0 - m - oy) // bs), int((y1 + m - oy) // bs) + 1)}
    for db in dev_blocks:
        block_district.setdefault(db, "开创大道dev")
    split = assign_blocks(block_district, weight, dev_blocks, cfg)
    margin = cfg["gap_m"] / 2 + cfg["window_half_extent_m"]
    blocks = [{"ix": i, "iy": j, "split": split[(i, j)], "district": block_district[(i, j)],
               "n_signal": int(weight.get((i, j), 0)), "n_ref_lines": int(sum(votes[(i, j)].values()))}
              for (i, j) in sorted(split)]
    man = {"version": cfg["version"], "crs": "EPSG:32649", "block_size_m": bs, "origin": [ox, oy],
           "gap_m": cfg["gap_m"], "window_half_extent_m": cfg["window_half_extent_m"], "buffer_margin_m": margin,
           "fractions": cfg["fractions"], "seed": cfg["seed"], "dev_blocks": sorted([list(x) for x in dev_blocks]),
           "blocks": blocks}
    path = REPO_ROOT / cfg["manifest_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 统计与自检
    sm = SplitManifest.load(path)
    lab = [sm.assign(x, y) for x, y in zip(sx, sy)]
    cnt = collections.Counter(lab)
    per_d = collections.defaultdict(collections.Counter)
    for bb in blocks:
        per_d[bb["district"]][bb["split"]] += 1
    # 隔离带自检：任意两个不同集合（train/val/test/dev）信号灯样本中心距离 ≥ gap + 2·窗口半跨度
    pts = np.c_[sx, sy]
    keep = np.array([l in ("train", "val", "test", "dev") for l in lab])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts[keep])
    labs = np.array(lab)[keep]
    need = cfg["gap_m"] + 2 * cfg["window_half_extent_m"]
    viol = 0
    for i, j in tree.query_pairs(need):
        viol += labs[i] != labs[j]
    legacy = {}
    for name in ("校准集.json", "留出集.json"):
        p = path_setting("legacy_eval_dir") / name
        if p.exists():
            pts_l = json.loads(p.read_text(encoding="utf-8"))
            lx, ly = wgs84_to_utm(np.array([q["lon"] for q in pts_l]), np.array([q["lat"] for q in pts_l]))
            legacy[name] = dict(collections.Counter(sm.assign(a, b) for a, b in zip(lx, ly)))
    stats = {"signals_by_split": dict(cnt), "blocks_by_split": dict(collections.Counter(bb["split"] for bb in blocks)),
             "blocks_by_district": {k: dict(v) for k, v in per_d.items()},
             "cross_split_pairs_within_gap": int(viol), "required_center_distance_m": need,
             "legacy_eval_points": legacy}
    man["stats"] = stats
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0 if viol == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
