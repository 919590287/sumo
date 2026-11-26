"""36 通道米制 BEV 在线光栅化（docs/DESIGN.md §3.2 分支 A）。

选择性抗锯齿：
- 窄线 / 窄漆面通道：``analytic``（解析精确面积覆盖率：按像素网格切分多边形边，累加有向梯形面积，
  零超采样、结果为精确面积分数）或 ``ss4``（4×4 子像素超采样 + 均值池化），由 M0 基准定型；
- 宽面通道：1× 像素中心精确光栅化；
- 派生场：全标线截断 UDF、标线无向主轴场 (cos 2φ, sin 2φ)、箭头朝向与 6 维转向通道。
纯确定性张量化，不含任何几何裁决。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from src.common.config import load_config
from src.data.roi import RoiFeatures, RoiWindow

TURN_ORDER = ("L", "S", "R", "U", "FL", "FR")


# ====================================================================== 几何 → 像素环
def polygons_of(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(polygons_of(g))
        return out
    return []


def pixel_polys(geoms, window: RoiWindow) -> list[list[np.ndarray]]:
    """几何序列 → 每个多边形的像素坐标环列表 [外环(CCW 世界坐标), 内环...]。"""
    xmin, ymin, xmax, ymax = window.bounds
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        g = shapely.clip_by_rect(g, xmin - 1, ymin - 1, xmax + 1, ymax + 1)
        for p in polygons_of(g):
            if p.area <= 0:
                continue
            p = orient(p, 1.0)
            rings = [window.world_to_pixel(np.asarray(p.exterior.coords))]
            rings += [window.world_to_pixel(np.asarray(r.coords)) for r in p.interiors]
            out.append(rings)
    return out


# ====================================================================== 解析精确覆盖率
def coverage_analytic(polys: list[list[np.ndarray]], H: int, W: int) -> np.ndarray:
    """精确面积覆盖率：每条边在整数网格线处切分为单元内子段，
    子段对所在单元贡献 dy·(1−f)、对右邻单元贡献 dy·f（f 为子段中点在单元内的横向分数），
    行内前缀和即得每像素被多边形覆盖的有向面积（重叠按绕数累加后截断到 [0, 1]）。"""
    segs = []
    for rings in polys:
        for r in rings:
            r = np.asarray(r, dtype=np.float64)
            if len(r) < 3:
                continue
            if not np.allclose(r[0], r[-1]):
                r = np.vstack([r, r[:1]])
            segs.append(np.hstack([r[:-1], r[1:]]))
    if not segs:
        return np.zeros((H, W), np.float32)
    S = np.vstack(segs)
    x0, y0, x1, y1 = S.T
    keep = y0 != y1
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    n = len(x0)
    ids = np.arange(n)

    def crossings(a0, a1):
        lo = np.floor(np.minimum(a0, a1)) + 1
        hi = np.ceil(np.maximum(a0, a1)) - 1
        cnt = np.maximum(hi - lo + 1, 0).astype(np.int64)
        tot = int(cnt.sum())
        if tot == 0:
            return np.zeros(0, np.int64), np.zeros(0)
        seg = np.repeat(ids, cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        k = np.arange(tot) - start + np.repeat(lo, cnt)
        da = (a1 - a0)[seg]
        t = (k - a0[seg]) / np.where(da == 0, 1, da)
        return seg, t

    sx, tx = crossings(x0, x1)
    sy, ty = crossings(y0, y1)
    seg_all = np.concatenate([ids, ids, sx, sy])
    t_all = np.concatenate([np.zeros(n), np.ones(n), tx, ty])
    order = np.lexsort((t_all, seg_all))
    seg_all, t_all = seg_all[order], t_all[order]
    same = seg_all[:-1] == seg_all[1:]
    s = seg_all[:-1][same]
    ta, tb = t_all[:-1][same], t_all[1:][same]
    dx, dy = (x1 - x0)[s], (y1 - y0)[s]
    xa, ya = x0[s] + ta * dx, y0[s] + ta * dy
    xb, yb = x0[s] + tb * dx, y0[s] + tb * dy
    sdy = yb - ya
    xm, ym = (xa + xb) / 2, (ya + yb) / 2
    cx = np.floor(xm).astype(np.int64)
    cy = np.floor(ym).astype(np.int64)
    f = xm - cx
    ok = (cy >= 0) & (cy < H) & (cx < W) & (sdy != 0)
    cx, cy, f, sdy = cx[ok], cy[ok], f[ok], sdy[ok]
    left = cx < 0
    acc = np.zeros(H * (W + 1))
    idx_l = cy[left] * (W + 1)
    acc += np.bincount(idx_l, weights=sdy[left], minlength=H * (W + 1))
    m = ~left
    i0 = cy[m] * (W + 1) + cx[m]
    acc += np.bincount(i0, weights=sdy[m] * (1 - f[m]), minlength=H * (W + 1))
    acc += np.bincount(i0 + 1, weights=sdy[m] * f[m], minlength=H * (W + 1))
    cov = np.abs(np.cumsum(acc.reshape(H, W + 1), axis=1)[:, :W])
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


# ====================================================================== cv2 填充（1× / 超采样）
_SHIFT = 8


def _local_masks(polys, H: int, W: int, scale: int):
    """逐多边形在外包框内局部填充（偶奇规则处理内环），产出 (y0, y1, x0, x1, mask)。"""
    for rings in polys:
        pts = [(np.asarray(r) * scale - 0.5) for r in rings]
        allp = np.vstack(pts)
        x0 = max(int(math.floor(allp[:, 0].min())) - 1, 0)
        y0 = max(int(math.floor(allp[:, 1].min())) - 1, 0)
        x1 = min(int(math.ceil(allp[:, 0].max())) + 2, W * scale)
        y1 = min(int(math.ceil(allp[:, 1].max())) + 2, H * scale)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv_pts = [np.round((p - (x0, y0)) * (1 << _SHIFT)).astype(np.int32) for p in pts]
        cv2.fillPoly(sub, cv_pts, 1, lineType=cv2.LINE_8, shift=_SHIFT)
        yield y0, y1, x0, x1, sub


def _fill_polys(polys, H: int, W: int, scale: int) -> np.ndarray:
    """逐多边形局部填充后取并，像素中心采样。"""
    canvas = np.zeros((H * scale, W * scale), np.uint8)
    for y0, y1, x0, x1, sub in _local_masks(polys, H, W, scale):
        np.maximum(canvas[y0:y1, x0:x1], sub, out=canvas[y0:y1, x0:x1])
    return canvas


def fill_center(polys, H: int, W: int) -> np.ndarray:
    return _fill_polys(polys, H, W, 1).astype(np.float32)


def coverage_supersample(polys, H: int, W: int, ss: int = 4) -> np.ndarray:
    big = _fill_polys(polys, H, W, ss).astype(np.float32)
    return big.reshape(H, ss, W, ss).mean(axis=(1, 3))


def narrow_coverage(polys, H: int, W: int, method: str, ss: int = 4) -> np.ndarray:
    if method == "analytic":
        return coverage_analytic(polys, H, W)
    if method == "ss4":
        return coverage_supersample(polys, H, W, ss)
    raise ValueError(method)


# ====================================================================== 派生场
def udf_field(marking: np.ndarray, res: float, trunc_m: float, seed_cov: float):
    """全标线截断无符号距离场（归一化至 [0, 1]）+ 最近标线像素索引。"""
    seed = (marking >= seed_cov)
    if not seed.any():
        H, W = marking.shape
        return np.ones_like(marking), None
    src = np.where(seed, 0, 1).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    udf = np.minimum(dist * res, trunc_m) / trunc_m
    return udf.astype(np.float32), (labels, seed)


def axis_field(marking: np.ndarray, udf: np.ndarray, nearest, sigma: float):
    """标线无向主轴场：结构张量主方向（梯度正交方向）在标线像素上取值，并沿最近标线像素
    传播至截断半径内、按 (1 − UDF) 衰减。"""
    H, W = marking.shape
    if nearest is None:
        return np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    gx = cv2.Sobel(marking, cv2.CV_32F, 1, 0, ksize=3)
    gy = -cv2.Sobel(marking, cv2.CV_32F, 0, 1, ksize=3)       # 行向下 → y 向上
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    a, b = jxx - jyy, 2 * jxy
    nrm = np.sqrt(a * a + b * b) + 1e-12
    c2, s2 = -a / nrm, -b / nrm                               # 线方向 = 梯度方向 + 90°
    labels, seed = nearest
    ys, xs = np.nonzero(seed)
    lab_at = labels[ys, xs]
    lut_c = np.zeros(labels.max() + 1, np.float32)
    lut_s = np.zeros(labels.max() + 1, np.float32)
    lut_c[lab_at] = c2[ys, xs]
    lut_s[lab_at] = s2[ys, xs]
    fade = 1.0 - udf
    return (lut_c[labels] * fade).astype(np.float32), (lut_s[labels] * fade).astype(np.float32)


# ====================================================================== 主入口
@dataclass
class BevSample:
    tensor: np.ndarray          # [36, H, W] float32
    names: list[str]
    window: RoiWindow
    timing: dict


def rasterize(feat: RoiFeatures, method: str | None = None, cfg: dict | None = None) -> BevSample:
    import time
    cfg = cfg or load_config("data")
    rc = cfg["raster"]
    method = method or rc["narrow_method"]
    win = feat.window
    H, W = win.shape
    names = [c["name"] for c in cfg["channels"]]
    out = np.zeros((len(names), H, W), np.float32)
    ch = {n: k for k, n in enumerate(names)}
    road = feat.road
    codes = road["分类代码"].values
    timing = {"narrow": 0.0, "wide": 0.0, "derived": 0.0}

    def geoms_for(code_list):
        return road.geometry.values[np.isin(codes, code_list)]

    road_surf = geoms_for(["road_surface"])
    boundary = None
    for spec in cfg["channels"]:
        name, mode = spec["name"], spec["mode"]
        t0 = time.perf_counter()
        if mode == "derived":
            continue
        if spec.get("source") == "roi":
            out[ch[name]] = fill_center(pixel_polys([win.roi], win), H, W)
        elif spec.get("source") == "road_boundary":
            if len(road_surf):
                u = shapely.union_all(shapely.clip_by_rect(road_surf, *_pad(win.bounds, 2.0)))
                boundary = shapely.buffer(u.boundary, rc["boundary_width_m"] / 2, cap_style="flat")
                out[ch[name]] = narrow_coverage(pixel_polys([boundary], win), H, W, method, rc["supersample"])
        elif spec.get("source") == "stripes":
            out[ch[name]] = narrow_coverage(pixel_polys(feat.stripes.geometry.values, win), H, W, method,
                                            rc["supersample"])
        elif spec.get("source") == "separation":
            g = shapely.buffer(feat.separation.geometry.values, rc["separation_width_m"] / 2, cap_style="flat")
            out[ch[name]] = narrow_coverage(pixel_polys(g, win), H, W, method, rc["supersample"])
        elif spec.get("source") == "arrows":
            _arrows(feat, cfg, win, out, ch)
        elif mode == "narrow":
            out[ch[name]] = narrow_coverage(pixel_polys(geoms_for(spec["codes"]), win), H, W, method,
                                            rc["supersample"])
        elif mode == "wide":
            gs = list(geoms_for(spec["codes"]))
            if spec.get("other_area_codes"):
                oa = feat.other_area
                gs += list(oa.geometry.values[np.isin(oa["分类代码"].values, spec["other_area_codes"])])
            out[ch[name]] = fill_center(pixel_polys(gs, win), H, W)
        elif mode == "mixed":
            a = fill_center(pixel_polys(geoms_for(spec["codes"]), win), H, W)
            b = narrow_coverage(pixel_polys(geoms_for(spec["narrow_codes"]), win), H, W, method, rc["supersample"])
            out[ch[name]] = np.maximum(a, b)
        timing["narrow" if mode in ("narrow", "mixed") else "wide"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    marking = np.max(out[[ch[n] for n in cfg["marking_channels"]]], axis=0)
    udf, nearest = udf_field(marking, win.resolution, rc["udf_trunc_m"], rc["udf_seed_coverage"])
    c2, s2 = axis_field(marking, udf, nearest, rc["axis_sigma_px"])
    out[ch["udf_marking"]] = udf
    out[ch["marking_axis_cos2"]] = c2
    out[ch["marking_axis_sin2"]] = s2
    # z_order_mean / z_order_span：Phase 1 平面场景置 0（Phase 2 接入 道路面_三维 显示层序）
    timing["derived"] += time.perf_counter() - t0
    return BevSample(out, names, win, timing)


def _pad(b, d):
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def _arrows(feat: RoiFeatures, cfg: dict, win: RoiWindow, out: np.ndarray, ch: dict) -> None:
    H, W = win.shape
    turns = cfg["arrow_turns"]
    road = feat.road
    sel = road[road["分类代码"].isin(list(turns.keys()))]
    for code, geom, hd in zip(sel["分类代码"].values, sel.geometry.values, sel["heading"].values):
        for y0, y1, x0, x1, sub in _local_masks(pixel_polys([geom], win), H, W, 1):
            m = sub > 0
            if not m.any():
                continue
            out[ch["arrow_mask"], y0:y1, x0:x1][m] = 1.0
            if np.isfinite(hd):
                out[ch["arrow_cos"], y0:y1, x0:x1][m] = math.cos(hd)
                out[ch["arrow_sin"], y0:y1, x0:x1][m] = math.sin(hd)
            for t in turns[code]:
                out[ch[f"turn_{t}"], y0:y1, x0:x1][m] = 1.0
