"""无标注自检指标 + 留出集批量评测。

python -m lanegen.evaluate <标签> [--limit N] [--no-shift]

指标（每个路口）：
  arrow_recall    进口箭头覆盖率：前向射线 60 m 内打到本路口停止线的箭头，有多少落在方向一致的车道上
  stop_cover      停止线覆盖：止于该停止线的进口车道宽度之和 / 停止线长度，取各停止线中位数
  stop_good       覆盖率在 [0.75, 1.3] 的停止线占比
  crossings       车道之间非端点处的交叉次数（应为 0）
  duplicates      相距 < 0.4 W_ref、平行重叠 > 10 m 的车道对（应为 0）
  frag_per_lane   未归类端点（open / lane_add / lane_drop）每车道个数，偏大说明断裂
  unknown_dir     方向未定车道数
  width_bad       宽度不在 [2.6, 4.2] 的车道占比
  merged_flag     路口核心里有同向、前后相距 > 30 m 的两条停止线（疑似把相邻路口并进来）
  shift_stable    种子平移 20 m 后，两次结果车道重合（0.3 m 内）的长度占比
  （v16 起）Step 4 契约审计：contract_errors（双向引用/分段一致性错误数，门禁 0）、orphan_lanes（不在车行道/渠化右转里的
  车道，含附加支路）、cw_arrow_dir_conflict（车行道内反向箭头数，门禁 0）、narrow_cw_count（< 2.2 m 却有车道的车行道，门禁 0）、
  jump_transitions（分段间跨 ≥2 条车道的跳变，门禁 0）、dp_fallback_rate（退回射线车道的车行道占比）、
  dir_conflicts（每枚箭头最近车道与箭头反向的车道数，实算）
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform

from . import pipeline


def _tan(line, s, h=1.0):
    a = line.interpolate(max(0.0, s - h))
    b = line.interpolate(min(line.length, s + h))
    v = np.array([b.x - a.x, b.y - a.y])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def approach_arrows(res, reach=80.0, with_hit=False):
    """评测用（独立于算法）：箭头前向射线 reach 米内碰到的第一条停止线（全部停止线中），
    若属于本路口则计为本路口进口箭头。返回 [(p, v, 核心停止线序号[, 落点])]。"""
    from .junction import arrows_of
    out = []
    allst = res["B"]["stops"]
    core_ids = {id(s): k for k, s in enumerate(res["core_stops"])}
    for p, v, code in arrows_of(res["roi"]["main"]):
        ray = LineString([p, p + v * reach])
        best = None
        for s in allst:
            x = ray.intersection(s)
            if x.is_empty:
                continue
            q = x if x.geom_type == "Point" else x.representative_point()
            d = Point(p).distance(q)
            if best is None or d < best[0]:
                best = (d, s, np.asarray(q.coords[0]))
        if best and id(best[1]) in core_ids:
            k = core_ids[id(best[1])]
            out.append((p, v, k, best[2]) if with_hit else (p, v, k))
    return out


def stop_direction(res):
    """每条核心停止线的行驶方向（指向路口），由打到它的箭头决定；没有箭头的为 None。"""
    dirs = {}
    for p, v, k in approach_arrows(res):
        dirs.setdefault(k, []).append(v)
    return {k: np.mean(vs, 0) / np.linalg.norm(np.mean(vs, 0)) for k, vs in dirs.items()}


def metrics(res):
    lanes = res["lanes"]
    w_ref = res["w_ref"]
    m = {}
    # 箭头覆盖
    aa = approach_arrows(res)
    hit = 0
    for p, v, _ in aa:
        pt = Point(p)
        ok = False
        for ln in lanes:
            g = ln["geom"]
            if g.distance(pt) <= 0.5 * w_ref + 0.3 and np.dot(_tan(g, g.project(pt)), v) > 0.7:
                ok = True
                break
        hit += ok
    m["n_arrows"] = len(aa)
    # 不依赖聚类：种子 80 m 内全部箭头的覆盖率（评测用固定半径，不进入算法）
    from .junction import arrows_of
    near = [(p, v) for p, v, c in arrows_of(res["roi"]["main"]) if np.hypot(*p) < 80.0]
    h2 = 0
    for p, v in near:
        pt = Point(p)
        h2 += any(ln["geom"].distance(pt) <= 0.5 * w_ref + 0.3 and np.dot(_tan(ln["geom"], ln["geom"].project(pt)), v) > 0.7
                  for ln in lanes)
    m["n_arrows80"] = len(near)
    m["arrow_recall80"] = h2 / len(near) if near else np.nan
    m["arrow_recall"] = hit / len(aa) if aa else np.nan
    # 停止线覆盖（只评有箭头指向的停止线，排除渠化让行线等）
    sd = stop_direction(res)
    covs = []
    for k, s in enumerate(res["core_stops"]):
        if k not in sd or s.length < 2.5:
            continue
        wsum = 0.0
        for ln in lanes:
            e = Point(ln["geom"].coords[-1])
            if e.distance(s) < 2.0:
                wsum += ln["width"]
        covs.append(wsum / s.length)
    m["n_stops"] = len(covs)
    # 车道数对箭头组数：打到同一停止线的箭头按横向位置（容差 0.5 W）分组 ≈ 有箭头的车道数
    eq = ge = 0
    nst = 0
    aah = approach_arrows(res, with_hit=True)
    for k, s in enumerate(res["core_stops"]):
        av = [h for p, v, kk, h in aah if kk == k]
        if not av:
            continue
        nst += 1
        dv = np.asarray(s.coords[-1]) - np.asarray(s.coords[0])
        dv = dv / np.linalg.norm(dv)
        # 用射线在停止线上的落点分组，不用箭头本身的位置（弯道、斜交进口上箭头会偏出停止线范围）
        lat = sorted(float(np.dot(h - np.asarray(s.coords[0]), dv)) for h in av)
        groups = 1 + sum(1 for i in range(1, len(lat)) if lat[i] - lat[i - 1] > 0.5 * w_ref)
        nl = sum(1 for ln in lanes if Point(ln["geom"].coords[-1]).distance(s) < 0.5 * w_ref + 1.0)
        eq += nl == groups
        ge += nl >= groups
    m["stop_lane_eq"] = eq / nst if nst else np.nan
    m["stop_lane_ge"] = ge / nst if nst else np.nan
    m["stop_cover"] = float(np.median(covs)) if covs else np.nan
    m["stop_good"] = float(np.mean([(0.75 <= c <= 1.3) for c in covs])) if covs else np.nan
    # 交叉、重复
    cross = dup = 0
    G = [ln["geom"] for ln in lanes]
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            a, b = G[i], G[j]
            if a.distance(b) > 0.5 * w_ref:
                continue
            x = a.intersection(b)
            pts = [q for q in getattr(x, "geoms", [x]) if q.geom_type == "Point"]
            ends = [Point(a.coords[0]), Point(a.coords[-1]), Point(b.coords[0]), Point(b.coords[-1])]
            cross += sum(1 for q in pts if min(q.distance(e) for e in ends) > 3.0)
            ov = a.intersection(b.buffer(0.4 * w_ref)).length
            if ov > 10.0:
                dup += 1
    m["crossings"] = cross
    m["duplicates"] = dup
    n = max(1, len(lanes))
    m["n_lanes"] = len(lanes)
    m["frag_per_lane"] = sum((ln["start"] in ("open", "lane_add")) + (ln["end"] in ("open", "lane_drop")) for ln in lanes) / n
    m["unknown_dir"] = sum(ln["direction"] == "unknown" for ln in lanes)
    m["width_bad"] = sum(not (2.6 <= ln["width"] <= 4.2) for ln in lanes) / n
    # 疑似并入相邻路口
    flag = 0
    keys = list(sd.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            va, vb = sd[keys[a]], sd[keys[b]]
            if np.dot(va, vb) > 0.95:
                ca = np.array(res["core_stops"][keys[a]].centroid.coords[0])
                cb = np.array(res["core_stops"][keys[b]].centroid.coords[0])
                if abs(np.dot(cb - ca, va)) > 30.0:
                    flag = 1
    m["merged_flag"] = flag
    m["core_stops"] = len(res["core_stops"])
    m.update(structure_metrics(res))
    return m


# ---------------- 结构自检（v12 起）：出口/进口臂完整性、岛区侵入、离开路面、贴线误差 ----------------

def _nondrive_zone(main):
    """评测独立构造的不可行驶区：安全岛 + 导流区（斜纹条按 1.5 m 闭运算连成整块）。"""
    from shapely.ops import unary_union
    code = main["分类代码"]
    isl = [g for g, c in zip(main.geometry, code) if c == "traffic_island"]
    ch = [g for g, c in zip(main.geometry, code) if c == "channelization_area"]
    parts = []
    if isl:
        parts.append(unary_union(isl))
    if ch:
        parts.append(unary_union([g.buffer(1.5) for g in ch]).buffer(-1.5))
    return unary_union(parts) if parts else None


def _cast_both(P, T, bounds, reach=4.5, par_deg=25.0):
    """P 处沿法向两侧找第一条平行边界，返回 (左距离, 右距离)，没找到为 nan。"""
    import shapely
    from .lanes import _segments
    A, B, _ = _segments(bounds)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    out = []
    for sgn in (1, -1):
        rays = shapely.linestrings(np.stack([P + sgn * N * 0.05, P + sgn * N * reach], 1))
        segs = shapely.linestrings(np.stack([A, B], 1))
        ri, si = shapely.STRtree(segs).query(rays, predicate="intersects")
        D = B[si] - A[si]
        Dh = D / np.linalg.norm(D, axis=1)[:, None]
        n = sgn * N[ri]
        w = A[si] - P[ri]
        den = n[:, 0] * D[:, 1] - n[:, 1] * D[:, 0]
        s = (w[:, 0] * D[:, 1] - w[:, 1] * D[:, 0]) / np.where(np.abs(den) < 1e-12, np.nan, den)
        par = np.abs(T[ri, 0] * Dh[:, 1] - T[ri, 1] * Dh[:, 0]) < np.sin(np.radians(par_deg))
        d = np.full(len(P), np.nan)
        ok = np.isfinite(s) & (s > 0) & par
        for i, v in zip(ri[ok], s[ok] + 0.05):
            if not (v >= d[i]):
                d[i] = v
        out.append(d)
    return out


def _ref_arms(res, lanes):
    """道路参考线（顶点顺序 = 行驶方向）在路口外环带内的分段，按离开/驶向路口分为出口/进口。
    只统计两侧 7 m 内有纵向标线的分段（排除无标线支路、非道路参考线）。
    环带内缘 = 路口核心（扩张前）离种子最远处 + 10 m，宽 30 m。返回 {out: [覆盖?], in: [覆盖?]}。"""
    from .boundaries import explode_lines
    import shapely
    core = res.get("core0", res["core"])
    c = np.asarray(core.exterior.coords)
    r1 = max(45.0, float(np.max(np.hypot(c[:, 0], c[:, 1]))) + 10.0)
    r2 = r1 + 30.0
    marks = [l for l, t in res["B"]["bounds"] if t in ("white", "dash", "yellow", "bus")]
    mtree = shapely.STRtree(marks) if marks else None
    w = res["w_ref"]
    out = {"out": [], "in": []}
    for ref in explode_lines(res["roi"]["ref"].geometry):
        L = ref.length
        if L < 10:
            continue
        s = np.arange(1.0, L, 3.0)
        pts = shapely.get_coordinates(shapely.line_interpolate_point(ref, s))
        rr = np.hypot(pts[:, 0], pts[:, 1])
        inside = (rr >= r1) & (rr <= r2)
        if inside.sum() < 4:
            continue
        # 按连续段切开（一条参考线可能两次穿过环带）
        idx = np.nonzero(inside)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for run in runs:
            if len(run) < 4:
                continue
            P = pts[run]
            T = np.array([_tan(ref, s[i]) for i in run])
            radial = np.einsum("ij,ij->i", T, P / np.hypot(P[:, 0], P[:, 1])[:, None])
            if abs(np.mean(radial)) < 0.5:
                continue  # 环带内绕行的参考线（不是路口臂）
            kind = "out" if np.mean(radial) > 0 else "in"
            # 标线支撑
            sup = 0
            for p, t in zip(P, T):
                q = shapely.Point(p)
                ok = False
                if mtree is not None:
                    for k in mtree.query(q.buffer(7.0)):
                        mk = marks[k]
                        if mk.length < 3:
                            continue
                        tt = _tan(mk, mk.project(q))
                        if abs(np.dot(tt, t)) > 0.94:
                            ok = True
                            break
                sup += ok
            if sup < 0.5 * len(P):
                continue
            # 覆盖：7 m 内有同向车道
            cov = 0
            for p, t in zip(P, T):
                q = shapely.Point(p)
                cov += any(ln["geom"].distance(q) < 7.0 and np.dot(_tan(ln["geom"], ln["geom"].project(q)), t) > 0.85
                           for ln in lanes)
            out[kind].append(cov >= 0.5 * len(P))
    return out


def foreign_intrusion(res):
    """评测口径独立于算法：全部停止线中，不在本路口簇里、离本路口扩张前核心 ≥ 8 m、行驶方向背离本路口的
    （= 下一个路口的进口停止线）；车道穿过它之后伸进下一个路口的长度之和（米）。"""
    import numpy as np
    from shapely.geometry import Point
    core = res.get("core0", res["core"])
    cc = np.asarray(core.centroid.coords[0])
    cl = {id(s) for s in res["core_stops"]}
    dirs = res.get("stop_dirs") or []
    tot = 0.0
    for k, s in enumerate(res["B"]["stops"]):
        if id(s) in cl or s.length < 3.0 or k >= len(dirs) or dirs[k] is None or s.distance(core) < 8.0:
            continue
        m_ = np.asarray(s.interpolate(0.5, normalized=True).coords[0])
        v_ = m_ - cc
        if float(np.dot(dirs[k], v_)) <= 0.5 * float(np.linalg.norm(v_)):
            continue
        for ln in res["lanes"]:
            g = ln["geom"]
            if not g.crosses(s):
                continue
            rest = g.difference(s.buffer(0.3))
            parts = [p for p in getattr(rest, "geoms", [rest]) if p.geom_type == "LineString"]
            if len(parts) >= 2:
                near = min(parts, key=lambda p: p.distance(core))
                tot += sum(p.length for p in parts if p is not near)
    return round(tot, 1)


def structure_metrics(res):
    from shapely.ops import unary_union
    import shapely
    lanes = res["lanes"]
    m = {}
    arms = _ref_arms(res, lanes)
    m["n_ref_out"] = len(arms["out"])
    m["n_ref_in"] = len(arms["in"])
    m["ref_out_recall"] = float(np.mean(arms["out"])) if arms["out"] else np.nan
    m["ref_in_recall"] = float(np.mean(arms["in"])) if arms["in"] else np.nan
    m["n_exit_lanes"] = sum(ln.get("role") == "exit" for ln in lanes)
    # Step 2：车道穿进紧致核心面内部（离边界 1 m 以上）的长度，应为 0；路臂、车行道统计
    jm = res.get("model")
    if jm is not None:
        inner = jm["core_poly"].buffer(-1.0)
        m["core_cross_m"] = round(sum(ln["geom"].intersection(inner).length for ln in lanes), 1) if not inner.is_empty else 0.0
        m["n_arms"] = len(jm["arms"])
        m["n_candidate_mouths"] = len(jm.get("candidate_mouths", []))
        m["n_arms_empty"] = sum(1 for a in jm["arms"] if not a["in_carriageways"] and not a["out_carriageways"])
        m["n_carriageways"] = len(jm["carriageways"])
        m["n_cw_unmarked"] = sum(1 for c in jm["carriageways"] if c.get("unmarked"))
        m["n_slips_model"] = len(jm["slips"])
        m["lanes_no_arm"] = sum(1 for ln in lanes if ln.get("arm") is None and ln.get("role") != "slip")
        m["dir_conflicts"] = jm.get("dir_conflicts", 0)
        # Step 4 契约审计（v16 起）
        from .frenet import validate_model_contract
        errs = validate_model_contract(jm, lanes) if "extra_lanes" in jm else []
        m["contract_errors"] = len(errs)
        owned = {i for c in jm["carriageways"] for i in c.get("lanes", [])} | {i for s in jm["slips"] for i in s.get("lanes", [])}
        m["orphan_lanes"] = sum(1 for i in range(len(lanes)) if i not in owned)  # 含附加支路 extra_lanes
        m["cw_arrow_dir_conflict"] = sum(int(c.get("arrow_dir_conflicts", 0)) for c in jm["carriageways"])
        m["narrow_cw_count"] = sum(1 for c in jm["carriageways"] if c.get("lanes") and float(c.get("width_m") or 0) < 2.2)
        m["jump_transitions"] = sum(1 for c in jm["carriageways"] for sec in c.get("sections", [])
                                    if str(sec.get("transition") or "").startswith("jump"))
        with_lanes = [c for c in jm["carriageways"] if c.get("lanes")]
        m["dp_fallback_rate"] = (sum(1 for c in with_lanes if not c.get("used_frenet_dp")) / len(with_lanes)) if with_lanes else np.nan
        m["n_slip_dead_end"] = sum(1 for s in jm["slips"] if s.get("dead_end"))
        m["foreign_intrude_m"] = foreign_intrusion(res)
        # 物理车行道对参考线出口/进口臂的覆盖率（验证 Step 2 物理切分是否覆盖了无标线出口）
        cw_pseudo = [dict(geom=c["axis"]) for c in jm["carriageways"] if c.get("axis") is not None]
        if cw_pseudo:
            arms_cw = _ref_arms(res, cw_pseudo)
            m["cw_out_recall"] = float(np.mean(arms_cw["out"])) if arms_cw["out"] else np.nan
            m["cw_in_recall"] = float(np.mean(arms_cw["in"])) if arms_cw["in"] else np.nan
    m["n_add_drop"] = sum((ln["start"] == "lane_add") + (ln["end"] == "lane_drop") for ln in lanes)
    m["n_slip"] = sum(bool(ln.get("slip")) for ln in lanes)
    m["n_nonmotor"] = sum(bool(ln.get("nonmotor_suspect")) for ln in lanes)
    tot = sum(ln["geom"].length for ln in lanes) or 1e-9
    main = res["roi"]["main"]
    nd = _nondrive_zone(main)
    if nd is not None and lanes:
        inner = nd.buffer(-0.5)
        m["island_m"] = round(sum(ln["geom"].intersection(inner).length for ln in lanes), 1)
    else:
        m["island_m"] = 0.0
    m["island_frac"] = m["island_m"] / tot
    # 只认可行驶路面（原始类别 15）；类别 20 等是非行车模型面，不能算合法路面
    road = [g for g, c, cat in zip(main.geometry, main["分类代码"], main["原始类别"]) if c == "road_surface" and cat == 15]
    if road and lanes:
        ru = unary_union([shapely.make_valid(g) for g in road]).buffer(0.5)
        m["offroad_frac"] = sum(ln["geom"].difference(ru).length for ln in lanes) / tot
    else:
        m["offroad_frac"] = 0.0
    # 贴线误差：车道中心到左右平行边界距离之差的一半
    P, T = [], []
    for ln in lanes:
        g = ln["geom"]
        if g.length < 8:
            continue
        s = np.arange(3.0, g.length - 3.0, 2.0)
        P.append(shapely.get_coordinates(shapely.line_interpolate_point(g, s)))
        T.append(np.array([_tan(g, x) for x in s]))
    if P:
        P, T = np.vstack(P), np.vstack(T)
        dl, dr = _cast_both(P, T, res["B"]["bounds"])
        ok = np.isfinite(dl) & np.isfinite(dr) & (dl + dr >= 2.2) & (dl + dr <= 5.0)
        err = np.abs(dl - dr)[ok] / 2
        m["center_err_p50"] = float(np.median(err)) if len(err) else np.nan
        m["center_err_p90"] = float(np.percentile(err, 90)) if len(err) else np.nan
        m["center_bad"] = float(np.mean(err > 0.5)) if len(err) else np.nan
    m["on_mark_m"] = on_raw_marking_m(res)
    return m


def on_raw_marking_m(res, tol=0.6, step=1.0):
    """压线长度（米）：车道中线与百度原始纵向标线（白/虚/黄，不含公交字样锚线）平行（<20°）且相距 < tol 的长度。
    独立于算法自己的边界（B["bounds"] 经过补段、改类型），对照的是原始数据。核心 2 m 内、人行横道上不计。"""
    import shapely
    from shapely.ops import unary_union
    from . import boundaries as bd
    L = res["roi"]["lines"]
    sty = L["源样式ID"].astype(int).values
    bm = [g for g, s in zip(L.geometry, sty) if s in (bd.BUS - bd.YELLOW)]
    bu = shapely.union_all(bm).buffer(0.3) if bm else None
    marks = []
    for g, s in zip(L.geometry, sty):
        if s in (bd.WHITE | bd.DASH | bd.YELLOW):
            for q in bd.explode_lines([g]):
                if bu is None or q.intersection(bu).length <= 0.5 * q.length:
                    marks.append(q)
    if not marks:
        return 0.0
    tree = shapely.STRtree(marks)
    excl = res["core"].buffer(2.0)
    if res["B"]["crosswalks"]:
        excl = excl.union(unary_union([c.buffer(0.5) for c in res["B"]["crosswalks"]]))
    tot = 0.0
    for ln in res["lanes"]:
        g = ln["geom"]
        ss = np.arange(0.5, g.length, step)
        if not len(ss):
            continue
        pts = shapely.line_interpolate_point(g, ss)
        keep = ~shapely.contains(excl, pts)
        for s, p in zip(ss[keep], pts[keep]):
            t = _tan(g, float(s))
            for j in tree.query(p, predicate="dwithin", distance=tol):
                m_ = marks[j]
                if abs(float(np.dot(t, _tan(m_, m_.project(p), 0.5)))) > 0.94:
                    tot += step
                    break
    return round(tot, 1)


def _to_wgs(res):
    t = Transformer.from_crs(res["crs"], 4326, always_xy=True).transform
    return [shp_transform(t, ln["geom"]) for ln in res["lanes"]]


def shift_stability(res_a, lon, lat, dx=20.0):
    """种子向东北平移 dx 米再跑一次，比较两次车道（在 a 的坐标系里）。"""
    dlon = dx / 111320.0 / math.cos(math.radians(lat))
    dlat = dx / 110900.0
    res_b = pipeline.run(lon + dlon * 0.7, lat + dlat * 0.7, log=lambda *a: None)
    tb = Transformer.from_crs(4326, res_a["crs"], always_xy=True).transform
    gb = [shp_transform(tb, g) for g in _to_wgs(res_b)]
    ga = [ln["geom"] for ln in res_a["lanes"]]
    if not ga and not gb:
        return 1.0
    from shapely.ops import unary_union
    ua = unary_union(ga).buffer(0.3) if ga else None
    ub = unary_union(gb).buffer(0.3) if gb else None
    la = sum(g.length for g in ga)
    lb = sum(g.length for g in gb)
    in_a = sum(g.intersection(ub).length for g in ga) if ub is not None else 0
    in_b = sum(g.intersection(ua).length for g in gb) if ua is not None else 0
    return (in_a + in_b) / max(1e-9, la + lb)


def main(tag, limit=None, shift=True, cases_path="data/评测/留出集.json", sumo=True, conn="cpsat"):
    cases = json.load(open(cases_path, encoding="utf-8"))
    from .truth import load_truth
    truth = load_truth()
    if limit:
        cases = cases[:limit]
    out_dir = os.path.join("输出/评测", tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in cases:
        t0 = time.time()
        row = dict(district=c["district"], fid=c["fid"], lon=c["lon"], lat=c["lat"])
        try:
            res = pipeline.run(c["lon"], c["lat"], log=lambda *a: None)
            row.update(status=res["status"], status_reason=res["status_reason"])
            row.update(metrics(res))
            if sumo:  # SUMO 闭环（与启发式不同源的物理可行性判据）
                from .sumo_check import check
                sm = check(res, conn=conn)
                row.update({f"sumo_{k}": v for k, v in sm.items() if not isinstance(v, (list, str))})
            if truth is not None:
                from .truth import score
                key = f"{c['district']}_{c['fid']}"
                if key in truth:
                    row.update({f"gt_{k}": v for k, v in score(res, truth[key]).items()})
            if shift:
                row["shift_stable"] = shift_stability(res, c["lon"], c["lat"])
            row["params"] = json.dumps(res.get("params", {}), ensure_ascii=False)
        except Exception as e:
            row["error"] = repr(e)
            traceback.print_exc()
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"{c['district']} {c['fid']}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                                              if k not in ("district", "fid", "lon", "lat", "params")}, flush=True)
    return save_metrics(rows, out_dir)


def save_metrics(rows, out_dir):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    num = df.select_dtypes("number").drop(columns=["fid", "lon", "lat"], errors="ignore")
    summ = dict(n=len(df), errors=int(df.get("error", pd.Series(dtype=str)).notna().sum()),
                mean=num.mean().round(3).to_dict(), median=num.median().round(3).to_dict())
    summ["recognition"] = df.get("status", pd.Series(dtype=str)).value_counts().to_dict()
    for k in ("crossings", "duplicates", "merged_flag", "unknown_dir", "island_m", "contract_errors",
              "cw_arrow_dir_conflict", "narrow_cw_count", "jump_transitions", "dir_conflicts"):
        if k in df:
            summ[f"junctions_with_{k}"] = int((df[k] > 0).sum())
    for k in ("sumo_dead_lanes", "sumo_teleports", "sumo_collisions"):
        if k in df:
            summ[f"junctions_with_{k}"] = int((df[k] > 0).sum())
    if "sumo_net_ok" in df:
        summ["sumo_net_fail"] = int((~df["sumo_net_ok"].astype(bool)).sum())
    for k in ("arrow_recall", "arrow_recall80", "stop_good", "stop_lane_eq", "stop_lane_ge", "shift_stable",
              "ref_out_recall", "ref_in_recall", "sumo_mv_cover", "sumo_route_arrive_rate"):
        if k in df:
            summ[f"{k}_p10"] = float(np.nanpercentile(df[k].astype(float), 10))
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return df


if __name__ == "__main__":
    tag = sys.argv[1]
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    cp = sys.argv[sys.argv.index("--cases") + 1] if "--cases" in sys.argv else "data/评测/留出集.json"
    main(tag, lim, shift="--no-shift" not in sys.argv, cases_path=cp, sumo="--no-sumo" not in sys.argv,
         conn=sys.argv[sys.argv.index("--conn") + 1] if "--conn" in sys.argv else "cpsat")
