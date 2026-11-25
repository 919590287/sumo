"""取数：按种子点读 ROI，投影到局部横轴墨卡托，按路口所在层做 Z 过滤。"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pyogrio
import shapely
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString

# 数据源：环境变量 LANEGEN_GPKG 优先，其次默认路径；read_roi(gpkg=...) 可逐次覆盖
GPKG = os.environ.get("LANEGEN_GPKG",
                      str(Path(__file__).resolve().parents[2] / "爬取百度路网" / "output" / "广州市"
                          / "广州道路仿真要素_全市采集与立交分层_WGS84.gpkg"))

LAYERS = {
    "main": "南沙道路要素",
    "lines": "未分类源线",
    "ref": "道路参考线",
    "barrier": "隔离设施路径",
    "zebra": "斑马线白色条纹",
    "signal": "信号灯点位",
}

STOP_STYLE = 501775
Z_TOL = 1.5


def local_crs(lon, lat):
    return CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat:.8f} +lon_0={lon:.8f} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )


def read_roi(lon, lat, radius_m=280, gpkg=None):
    dlat = radius_m / 110900.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    crs = local_crs(lon, lat)
    src = gpkg or GPKG
    if not os.path.exists(src):
        raise FileNotFoundError(f"找不到数据源 {src}（设置环境变量 LANEGEN_GPKG 或传 gpkg=）")
    out = {}
    for key, layer in LAYERS.items():
        df = pyogrio.read_dataframe(src, layer=layer, bbox=bbox, fid_as_index=True).reset_index()
        out[key] = df.to_crs(crs) if len(df) else df.set_crs(4326).to_crs(crs)
    return out, crs


def _native(j):
    try:
        d = json.loads(j)
    except Exception:
        return None
    return d.get("native", d)


def _zclip_line(geom, zvec, z0):
    """按逐顶点 Z 保留 |z-z0|<=Z_TOL 的连续段；zvector 为常数时整体判断。"""
    parts = list(getattr(geom, "geoms", [geom]))
    ncoords = sum(len(p.coords) for p in parts)
    if not zvec:
        return geom
    if len(zvec) != ncoords:
        z = float(np.mean(zvec))
        return geom if abs(z - z0) <= Z_TOL else None
    keep, k = [], 0
    for p in parts:
        cs = list(p.coords)
        zs = zvec[k:k + len(cs)]
        k += len(cs)
        run = []
        for c, z in zip(cs, zs):
            if abs(z - z0) <= Z_TOL:
                run.append(c)
            else:
                if len(run) >= 2:
                    keep.append(LineString(run))
                run = []
        if len(run) >= 2:
            keep.append(LineString(run))
    if not keep:
        return None
    return keep[0] if len(keep) == 1 else MultiLineString(keep)


def estimate_z0(lines, seed_xy, r=70.0):
    sx, sy = seed_xy
    st = lines[lines["源样式ID"] == STOP_STYLE]
    if len(st):
        d = st.geometry.distance(shapely.Point(sx, sy))
        near = st[(d < r) & st["源显示Z最大"].notna()]
        if len(near):
            # 以离种子最近的 3 条停止线的 Z 中位数为路口层
            near = near.assign(_d=d[near.index]).sort_values("_d").head(3)
            return float(np.median(near["源显示Z最大"].values))
    # 无停止线（无信号小路口）：取种子 r 内全部标线按长度加权的 Z 众数（0.5 m 分箱），
    # 再在离种子最近的 15 条里看这个层是否存在，避免被上方高架的大量标线带偏
    if len(lines):
        d = lines.geometry.distance(shapely.Point(sx, sy))
        near = lines[(d < r) & lines["源显示Z最大"].notna()]
        if len(near):
            z = np.round(near["源显示Z最大"].astype(float).values * 2) / 2
            wts = near.geometry.length.values
            closest = set(np.round(near.assign(_d=d[near.index]).sort_values("_d").head(15)["源显示Z最大"].astype(float).values * 2) / 2)
            vals, idx = np.unique(z, return_inverse=True)
            score = np.bincount(idx, weights=wts)
            for k in np.argsort(-score):
                if vals[k] in closest:
                    return float(np.median(near["源显示Z最大"].astype(float).values[z == vals[k]]))
    return 0.0


def zfilter(roi, z0):
    """返回 Z 过滤后的副本。线按逐顶点 Z 裁剪，面按代表 Z 判断。"""
    out = dict(roi)
    # 线（未分类源线、参考线）
    for key in ("lines", "ref"):
        df = roi[key].copy()
        geoms = []
        for g, j, zmin, zmax in zip(df.geometry, df["原始属性JSON"], df["源显示Z最小"], df["源显示Z最大"]):
            if zmin is not None and zmax is not None and not (np.isnan(zmin) or np.isnan(zmax)):
                if zmin >= z0 - Z_TOL and zmax <= z0 + Z_TOL:
                    geoms.append(g)
                    continue
                if zmax < z0 - Z_TOL or zmin > z0 + Z_TOL:
                    geoms.append(None)
                    continue
            n = _native(j) or {}
            geoms.append(_zclip_line(g, n.get("zvector", []), z0))
        df["geometry"] = geoms
        out[key] = df[df.geometry.notna()].copy()
    # 隔离设施：无 Z 列，用 JSON zvector
    df = roi["barrier"].copy()
    if len(df):
        df["geometry"] = [_zclip_line(g, (_native(j) or {}).get("zvector", []), z0)
                          for g, j in zip(df.geometry, df["原始属性JSON"])]
        out["barrier"] = df[df.geometry.notna()].copy()
    # 面
    m = roi["main"].copy()
    zr = m["源显示Z代表"].astype(float)
    zmin = m["源显示Z最小"].astype(float)
    zmax = m["源显示Z最大"].astype(float)
    keep = zr.isna() | ((zr - z0).abs() <= Z_TOL)
    # 跨层道路面（隧道口、匝道：代表 Z 不在本层，但 Z 范围跨过本层）：面没有逐顶点 Z，
    # 用落在该面内的标线/参考线逐顶点 Z 插值出 Z 场，只保留 Z 场落在本层的部分。
    # 旧做法只看“最低点 ≤ 本层”，地下隧道（-6 → -0.1 m）整块混进地面层。
    is_road = m["分类代码"] == "road_surface"
    span = is_road & ~keep & zmin.notna() & zmax.notna() & (zmin <= z0 + Z_TOL) & (zmax >= z0 - Z_TOL)
    # 向上跨层（上高架的匝道，代表 Z 在本层之上）：它的上半段与高架主线标线在平面上重叠，
    # 普通 IDW Z 场会把两层混在一起。改用“面内样本的下包络”：只取落在该匝道面内的标线点，
    # 邻域取最低 Z —— 起坡段（贴地）保留，悬空段（下方无本层标线）裁掉。
    # 旧做法整块放行，把 +12 m 的引桥投影混进地面可行驶面。
    up = span & (zr > z0)
    down = span & ~up
    level = [g for g, k, r in zip(m.geometry, keep, is_road) if k and r]
    for sel, mode in ((down, "idw"), (up, "low")):
        if sel.any():
            clipped = _clip_spanning(list(m.geometry[sel]), level, roi, z0, mode=mode)
            m.loc[sel[sel].index, "geometry"] = clipped
            keep = keep | (sel & m.geometry.notna() & ~m.geometry.is_empty)
    out["main"] = m[keep & m.geometry.notna()].copy()
    # 斑马线白条：本身无 Z，跟随父人行横道面（父面被 Z 过滤掉 → 白条也去掉；父面不在 ROI 内的保留）
    zb = roi.get("zebra")
    if zb is not None and len(zb) and "父范围FID" in zb.columns and "fid" in m.columns:
        raw_fids = set(m["fid"].tolist())
        kept_fids = set(out["main"]["fid"].tolist())
        par = zb["父范围FID"]
        drop = par.notna() & par.isin(raw_fids) & ~par.isin(kept_fids)
        out["zebra"] = zb[~drop].copy()
    # 信号灯点：JSON points 第 3 个分量为显示 Z（厘米级整数，0 = 未给），给了的按层过滤
    sg = roi.get("signal")
    if sg is not None and len(sg):
        keep_s = []
        for j in sg["原始属性JSON"]:
            n = _native(j) or {}
            pts = n.get("points") or []
            z = pts[2] / 100.0 if len(pts) >= 3 and pts[2] else None
            keep_s.append(z is None or abs(z - z0) <= Z_TOL)
        out["signal"] = sg[np.array(keep_s, bool)].copy()
    return out


def _zfield(roi, step=2.0):
    """原始（未过滤）标线、参考线按逐顶点 Z 加密成点云 (xy, z)。"""
    import shapely
    P, Z = [], []
    for key in ("lines", "ref"):
        df = roi[key]
        for g, j in zip(df.geometry, df["原始属性JSON"]):
            zv = (_native(j) or {}).get("zvector", [])
            if not zv or g is None:
                continue
            parts = list(getattr(g, "geoms", [g]))
            if sum(len(p.coords) for p in parts) != len(zv):
                continue
            k = 0
            for p in parts:
                c = np.asarray(p.coords)[:, :2]
                z = np.asarray(zv[k:k + len(c)], float)
                k += len(c)
                if len(c) < 2:
                    continue
                s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
                t = np.arange(0, s[-1] + 1e-9, step)
                P.append(np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1))
                Z.append(np.interp(t, s, z))
    if not P:
        return None, None
    return np.vstack(P), np.concatenate(Z)


def _clip_spanning(polys, level, roi, z0, cell=1.0, mode="idw"):
    """跨层道路面只保留：与本层道路面重叠的部分 + Z 场落在本层的部分。
    Z 场只取不在本层道路面内的点（本层路面下方的隧道段被本层路面盖住，取哪一层都不改变并集）。
    mode="low"（上高架匝道）：每个面只用落在面内（外扩 0.5 m）的样本，邻域取最低 Z。"""
    import shapely
    from scipy.spatial import cKDTree
    from shapely.ops import unary_union
    level = [shapely.make_valid(g) for g in level if g is not None]
    polys = [shapely.make_valid(g) if g is not None else g for g in polys]
    G = unary_union(level) if level else None
    XY, Z = _zfield(roi)
    if XY is not None and G is not None and not G.is_empty:
        off = ~shapely.contains_xy(G.buffer(-0.5), XY[:, 0], XY[:, 1])
        XY, Z = XY[off], Z[off]
    tree = cKDTree(XY) if XY is not None and len(XY) else None
    out = []
    for P in polys:
        extra = P.difference(G) if G is not None else P
        if extra.is_empty or extra.area < 1.0:
            out.append(P)  # 全被本层路面盖住，保留与否不改变并集
            continue
        x0, y0, x1, y1 = extra.bounds
        gx, gy = np.meshgrid(np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell))
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        pts = pts[shapely.contains_xy(extra, pts[:, 0], pts[:, 1])]
        has = np.zeros(len(pts), bool)
        zi = np.zeros(len(pts))
        if mode == "low" and XY is not None and len(XY) and len(pts):
            inside = shapely.contains_xy(P.buffer(0.5), XY[:, 0], XY[:, 1])
            if inside.any():
                t_in, z_in = cKDTree(XY[inside]), Z[inside]
                d, i = t_in.query(pts, k=min(6, int(inside.sum())), distance_upper_bound=12.0)
                d, i = d.reshape(len(pts), -1), i.reshape(len(pts), -1)
                valid = np.isfinite(d)
                zz = np.where(valid, z_in[np.minimum(i, len(z_in) - 1)], np.inf)
                has = valid.any(1)
                zi = np.where(has, zz.min(1), 0.0)
        elif tree is not None and len(pts):
            d, i = tree.query(pts, k=6, distance_upper_bound=12.0)
            valid = np.isfinite(d)
            wgt = np.where(valid, 1.0 / (d + 0.5), 0.0)
            zz = np.where(valid, Z[np.minimum(i, len(Z) - 1)], 0.0)
            has = wgt.sum(1) > 0
            zi = (wgt * zz).sum(1) / np.where(has, wgt.sum(1), 1)
        ok = has & (np.abs(zi - z0) <= Z_TOL)
        if G is not None and len(pts):  # 没有 Z 证据、紧贴本层路面的接缝（< 3 m）保留
            ok |= ~has & (shapely.distance(G, shapely.points(pts)) < 3.0)
        if ok.any():
            boxes = shapely.box(pts[ok, 0] - cell / 2 - 0.05, pts[ok, 1] - cell / 2 - 0.05,
                                pts[ok, 0] + cell / 2 + 0.05, pts[ok, 1] + cell / 2 + 0.05)
            # 闭运算抹平格子台阶，开运算去掉 < 2.5 m 宽的细条（匝道路面至少一个车道宽）
            keep_area = shapely.union_all(boxes).buffer(1.5).buffer(-1.5).buffer(-1.25).buffer(1.25)
            # 只留成片的、或贴着本层路面的部分（Z 场边缘的零碎台阶块丢掉）
            # 只留与本层路面相连的部分（匝道在本层的一段必然接着本层路网；孤立小块是 Z 插值的边角）
            ka = [q for q in getattr(keep_area, "geoms", [keep_area]) if q.geom_type == "Polygon"
                  and (G is None or q.distance(G) < 0.5)]
            keep_area = shapely.union_all(ka) if ka else shapely.Polygon()
            part = P.intersection(keep_area.union(G) if G is not None else keep_area)
        else:
            part = P.intersection(G) if G is not None else None
        out.append(part if part is not None and not part.is_empty else None)
    return out
