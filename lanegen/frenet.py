"""Step 3：Frenet (s, d) 曲线坐标系下的多证据 DP / Viterbi 全局车道剖分与增减切分引擎（v15 Pass-2）。

输入：Step 2 的 JunctionModel（物理单向车行道 polygon + 平滑有向 Frenet 基线 axis + 绕岛右转走廊 slips）
     + 清洗后的标线/停止线/人行横道/可行驶面 + 导向箭头 + 非机动车/停车符号
     + Step 1-2 射线中点车道 old_lanes（作为高精度标线中点观测锚点与多分支兜底）。
输出：
  1. 对每条物理车行道 cw，生成全局连续、起止对齐、宽度守恒、贴合标线、0 交叉的机动车道中心线，
     并在 cw["sections"] 中记录精确的展宽/收窄分段切点与车道对应关系（直供 Step 4 写 SUMO edge/node/connection）；
  2. 对每条绕岛渠化右转走廊 slip，生成贴合走廊中线的平滑右转车道；
  3. 无标线宽出口/缺失虚线段自动按物理宽度 + 箭头居中约束无缝补齐；
  4. 多分支/Y 型分叉车行道自动保留未被单一 axis 覆盖的支路车道，保证箭头与停止线覆盖零回退。
"""
import math
import os

import numpy as np
import shapely
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from .junction import movement
from . import ribbons as _ribbon


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _tan(g, s, h=1.0):
    a = np.asarray(g.interpolate(max(0.0, s - h)).coords[0])
    b = np.asarray(g.interpolate(min(g.length, s + h)).coords[0])
    return _unit(b - a)


def _parts(g, kind="LineString"):
    if g is None or g.is_empty:
        return []
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == kind]


def _smooth_1d(arr, win=7):
    arr = np.asarray(arr, float)
    n = len(arr)
    if n < 5:
        return arr.copy()
    w = min(win, n if n % 2 else n - 1)
    if w < 5:
        return arr.copy()
    out = savgol_filter(arr, w, 2, mode="interp")
    out[0], out[-1] = arr[0], arr[-1]
    return out


# ---------------------------------------------------------------- 1. Frenet 断面采样、走廊防夹尖去鼓包与多源观测提取

def _clean_corridor_bounds(d_R, d_L, ss, role, proj_arrows, old_obs, M):
    """消除三类物理多边形几何失真：
    1) 斜交停止线/路口端头的“毛笔尖收拢”（一侧先碰口导致断面宽度骤缩，迫使所有车道向一点夹尖）；
    2) 路侧出入口/中央掉头缺口造成的瞬时外鼓（宽度突增 >1.0 m 且在 24 m 内回落，且鼓包内无箭头/标线车道）；
    3) 路口转角倒圆（curb fillet）在端头 12 m 内无标线/无箭头的外甩。"""
    d_R = d_R.copy()
    d_L = d_L.copy()
    if M < 6:
        return d_R, d_L

    W = d_L - d_R

    # 1a. 端头防夹尖（un-pinching）：分别检查高 s 端与低 s 端
    end_span = min(10, M // 3)
    if end_span >= 2:
        # 高 s 端（进口停止线端或出口远端）
        win_hi = W[max(0, M - end_span - 4):M]
        w_ref_hi = float(np.percentile(win_hi, 80))
        ref_hi = M - 1
        for i in range(M - 2, max(0, M - end_span - 4) - 1, -1):
            if W[i] >= 0.88 * w_ref_hi:
                ref_hi = i
                break
        for i in range(ref_hi + 1, M):
            # 若某侧边界突然向内塌缩（斜切口特征），保持稳态外边界，后续由单车道独立精准裁到斜停止线
            if d_R[i] > d_R[i - 1] + 0.35:
                d_R[i] = d_R[i - 1]
            if d_L[i] < d_L[i - 1] - 0.35:
                d_L[i] = d_L[i - 1]

        # 低 s 端（出口路口端或进口远端）
        win_lo = W[:min(M, end_span + 4)]
        w_ref_lo = float(np.percentile(win_lo, 80))
        ref_lo = 0
        for i in range(1, min(M, end_span + 4)):
            if W[i] >= 0.88 * w_ref_lo:
                ref_lo = i
                break
        for i in range(ref_lo - 1, -1, -1):
            if d_R[i] > d_R[i + 1] + 0.35:
                d_R[i] = d_R[i + 1]
            if d_L[i] < d_L[i + 1] - 0.35:
                d_L[i] = d_L[i + 1]

    def _has_motor_evidence(i_lo, i_hi, d_lo, d_hi):
        """检查区段 [i_lo..i_hi] 的横向带 [d_lo, d_hi] 内是否有导向箭头或有标线支撑的旧车道。"""
        s_lo, s_hi = ss[i_lo] - 4.0, ss[i_hi] + 4.0
        for sa, da, _, _ in proj_arrows:
            if s_lo <= sa <= s_hi and d_lo <= da <= d_hi:
                return True
        hits = 0
        for j in range(i_lo, i_hi + 1):
            for dc, _, has_mark in old_obs[j]:
                if has_mark and d_lo <= dc <= d_hi:
                    hits += 1
                    if hits >= 2:
                        return True
        return False

    # 1b. 抑制瞬时外鼓（中央掉头口/支路开口/短港湾，跨度 ≤ 12 断面 = 24 m）
    max_span = min(13, M - 1)
    for a in range(M - 2):
        for span in range(max_span, 1, -1):
            b = a + span
            if b >= M:
                continue
            # 左侧（d_L）外鼓
            base_L = max(d_L[a], d_L[b])
            peak_L = float(np.max(d_L[a + 1:b]))
            if peak_L > base_L + 1.0:
                if not _has_motor_evidence(a + 1, b - 1, base_L + 0.35, peak_L + 0.5):
                    chord = np.linspace(d_L[a], d_L[b], span + 1)
                    d_L[a + 1:b] = np.minimum(d_L[a + 1:b], chord[1:-1])
            # 右侧（d_R）外鼓（d_R 更负）
            base_R = min(d_R[a], d_R[b])
            trough_R = float(np.min(d_R[a + 1:b]))
            if trough_R < base_R - 1.0:
                if not _has_motor_evidence(a + 1, b - 1, trough_R - 0.5, base_R - 0.35):
                    chord = np.linspace(d_R[a], d_R[b], span + 1)
                    d_R[a + 1:b] = np.maximum(d_R[a + 1:b], chord[1:-1])

    # 1c. 抑制路口端头（12 m 内）无箭头/无标线支撑的转角路缘倒圆外甩（curb fillet）
    mouth_span = min(6, M // 3)
    if mouth_span >= 2 and M >= mouth_span + 4:
        if role == "approach":
            ref_i = M - 1 - mouth_span
            ref_R = float(np.median(d_R[max(0, ref_i - 3):ref_i + 1]))
            ref_L = float(np.median(d_L[max(0, ref_i - 3):ref_i + 1]))
            if float(np.min(d_R[ref_i + 1:])) < ref_R - 1.0:
                if not _has_motor_evidence(ref_i + 1, M - 1, float(np.min(d_R[ref_i + 1:])) - 0.3, ref_R - 0.4):
                    for k, i in enumerate(range(ref_i + 1, M), 1):
                        d_R[i] = max(d_R[i], ref_R - 0.15 * k)
            if float(np.max(d_L[ref_i + 1:])) > ref_L + 1.0:
                if not _has_motor_evidence(ref_i + 1, M - 1, ref_L + 0.4, float(np.max(d_L[ref_i + 1:])) + 0.3):
                    for k, i in enumerate(range(ref_i + 1, M), 1):
                        d_L[i] = min(d_L[i], ref_L + 0.15 * k)
        elif role == "exit":
            ref_i = mouth_span
            ref_R = float(np.median(d_R[ref_i:min(M, ref_i + 4)]))
            ref_L = float(np.median(d_L[ref_i:min(M, ref_i + 4)]))
            if float(np.min(d_R[:ref_i])) < ref_R - 1.0:
                if not _has_motor_evidence(0, ref_i - 1, float(np.min(d_R[:ref_i])) - 0.3, ref_R - 0.4):
                    for k, i in enumerate(range(ref_i - 1, -1, -1), 1):
                        d_R[i] = max(d_R[i], ref_R - 0.15 * k)
            if float(np.max(d_L[:ref_i])) > ref_L + 1.0:
                if not _has_motor_evidence(0, ref_i - 1, ref_L + 0.4, float(np.max(d_L[:ref_i])) + 0.3):
                    for k, i in enumerate(range(ref_i - 1, -1, -1), 1):
                        d_L[i] = min(d_L[i], ref_L + 0.15 * k)

    return d_R, d_L


def _sample_frenet(cw, cw_old_lanes, B, arrows, bikes, parking, w_global, ds=2.0):
    """沿物理车行道 axis（已按行驶方向定向）按步长 ds 建立 Frenet 断面序列。
    横向坐标 d 沿行驶方向左侧法向 N = [-ty, tx]：
      d_R < d_L，其中 d_R 为右侧外路缘（SUMO lane_index 0 侧），d_L 为左侧内边界（黄线/中分带侧）。"""
    axis = cw["axis"]
    poly = cw["polygon"]
    L = axis.length
    if L < 6.0 or poly is None or poly.is_empty:
        return None
    if MARK_AXIS:
        ma = _ribbon.mark_axis(axis, poly, B)
        if ma is not None:
            axis, L = ma, ma.length
            cw["axis_source"] = "marking"
    M = max(4, int(round(L / ds)) + 1)
    ss_raw = np.linspace(0.0, L, M)
    P_raw = shapely.get_coordinates(shapely.line_interpolate_point(axis, ss_raw))
    # 适度平滑 axis（14-18 m 二次多项式窗口，端头偏移 > 1.2 m 时自动回退更小窗口防拐弯处拉直撞岛）
    if M >= 7:
        for win_cand in (11, 7):
            win_ax = min(win_cand, M if M % 2 else M - 1)
            if win_ax < 5:
                continue
            Px = savgol_filter(P_raw[:, 0], win_ax, 2, mode="interp")
            Py = savgol_filter(P_raw[:, 1], win_ax, 2, mode="interp")
            P_sm = np.stack([Px, Py], axis=1)
            end_shift = max(float(np.linalg.norm(P_sm[0] - P_raw[0])),
                            float(np.linalg.norm(P_sm[-1] - P_raw[-1])))
            if end_shift <= 1.2:
                axis_s = LineString(P_sm)
                if axis_s.length >= 6.0:
                    axis = axis_s
                    L = axis.length
                break
    M = max(4, int(round(L / ds)) + 1)
    ss = np.linspace(0.0, L, M)
    P = shapely.get_coordinates(shapely.line_interpolate_point(axis, ss))
    T = np.array([_tan(axis, s, h=min(2.5, 0.25 * L)) for s in ss])
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)  # 左法向

    # 包围盒对角线覆盖整幅；弯道仅在曲率中心一侧限制法线，避免内侧折叠。
    x0, y0, x1, y1 = poly.bounds
    span = math.hypot(x1 - x0, y1 - y0)
    reach_r = np.full(M, span)
    reach_l = np.full(M, span)
    for i in range(1, M - 1):
        a, b = T[i - 1], T[i + 1]
        ang = math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))
        kappa = ang / (ss[i + 1] - ss[i - 1])
        if abs(kappa) > 1e-3:
            inside = reach_l if kappa > 0 else reach_r
            inside[i] = min(span, max(5.0, 0.85 / abs(kappa)))

    # 1. 提取每个断面的物理车行道边界 [d_R, d_L]
    d_R = np.full(M, np.nan)
    d_L = np.full(M, np.nan)
    for i in range(M):
        cross = LineString([P[i] - N[i] * reach_r[i], P[i] + N[i] * reach_l[i]])
        segs = _parts(cross.intersection(poly), "LineString")
        if not segs:
            continue
        seg = min(segs, key=lambda sg: sg.distance(Point(P[i])))
        if seg.distance(Point(P[i])) > 8.0:
            continue
        c0, c1 = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1])
        v0 = float(np.dot(c0 - P[i], N[i]))
        v1 = float(np.dot(c1 - P[i], N[i]))
        d_R[i], d_L[i] = min(v0, v1), max(v0, v1)

    valid = np.isfinite(d_R) & np.isfinite(d_L) & ((d_L - d_R) >= 1.8)
    if valid.sum() < 3:
        return None
    idx = np.arange(M)
    d_R = np.interp(idx, idx[valid], d_R[valid])
    d_L = np.interp(idx, idx[valid], d_L[valid])
    if M >= 5:
        d_R = median_filter(d_R, size=3, mode="nearest")
        d_L = median_filter(d_L, size=3, mode="nearest")

    # 2. 投影本车行道内的导向箭头（方向一致）与非机动符号
    proj_arrows = []
    for ap, av, code in arrows:
        pt = Point(ap)
        if not poly.buffer(0.8).contains(pt):
            continue
        sa = float(axis.project(pt))
        ta = _tan(axis, sa, 1.5)
        if float(np.dot(av, ta)) < 0.65:
            continue
        na = np.array([-ta[1], ta[0]])
        qa = np.asarray(axis.interpolate(sa).coords[0])
        da = float(np.dot(ap - qa, na))
        proj_arrows.append((sa, da, movement(code), ap))

    proj_bikes = []
    for bp in bikes:
        if poly.buffer(0.5).contains(bp):
            sb = float(axis.project(bp))
            tb = _tan(axis, sb, 1.5)
            nb = np.array([-tb[1], tb[0]])
            qb = np.asarray(axis.interpolate(sb).coords[0])
            db = float(np.dot(np.asarray(bp.coords[0]) - qb, nb))
            proj_bikes.append((sb, db))

    # 3. 投影本车行道对应的 old_lanes（高精度标线中点观测锚点！）
    old_obs = [[] for _ in range(M)]
    if cw_old_lanes:
        for i in range(M):
            cross = LineString([P[i] + N[i] * (d_R[i] - 0.8), P[i] + N[i] * (d_L[i] + 0.8)])
            hits = []
            for ln in cw_old_lanes:
                g = ln["geom"]
                if not cross.intersects(g):
                    continue
                x = cross.intersection(g)
                for q in getattr(x, "geoms", [x]):
                    if q.is_empty:
                        continue
                    pt = q if q.geom_type == "Point" else q.representative_point()
                    sl = g.project(pt)
                    tl = _tan(g, sl, 1.5)
                    if float(np.dot(tl, T[i])) < 0.72:
                        continue
                    dc = float(np.dot(np.asarray(pt.coords[0]) - P[i], N[i]))
                    has_mark = (not ln.get("edgepair", False)) and (ln.get("n_norm", 0) >= 4)
                    hits.append((dc, float(ln["width"]), has_mark))
            # 同断面按 d 排序并去重（1.2 m 内保留有标线支撑的）
            hits.sort(key=lambda z: z[0])
            ded = []
            for dc, wl, hm in hits:
                if ded and dc - ded[-1][0] < 1.2:
                    if hm and not ded[-1][2]:
                        ded[-1] = (dc, wl, hm)
                else:
                    ded.append((dc, wl, hm))
            old_obs[i] = ded

    # 4. 走廊轮廓防夹尖、去瞬时外鼓、抑转角倒圆
    d_R, d_L = _clean_corridor_bounds(d_R, d_L, ss, cw["role"], proj_arrows, old_obs, M)

    # 5. 预筛选落入本车行道附近的纵向标线（用于断面相交）
    poly_buf = poly.buffer(1.4)
    marks = []
    for l, t in B["bounds"]:
        if t not in ("white", "dash", "yellow", "bus", "sep", "barrier"):
            continue
        if l.length < 1.5 or not l.intersects(poly_buf):
            continue
        marks.append((l, t))
    mtree = shapely.STRtree([l for l, _ in marks]) if marks else None

    # 6. 逐断面收集平行纵向标线交点
    obs = [[] for _ in range(M)]
    raw_solid_right = []
    raw_solid_left = []
    for i in range(M):
        cross = LineString([P[i] + N[i] * (d_R[i] - 0.6), P[i] + N[i] * (d_L[i] + 0.6)])
        hits = []
        if mtree is not None:
            for k in mtree.query(cross, predicate="intersects"):
                l, t = marks[k]
                x = cross.intersection(l)
                for q in getattr(x, "geoms", [x]):
                    if q.is_empty:
                        continue
                    pt = q if q.geom_type == "Point" else q.representative_point()
                    s_l = l.project(pt)
                    tl = _tan(l, s_l, 1.0)
                    if abs(float(np.dot(tl, T[i]))) < 0.80:
                        continue
                    dv = float(np.dot(np.asarray(pt.coords[0]) - P[i], N[i]))
                    if d_R[i] - 0.4 <= dv <= d_L[i] + 0.4:
                        hits.append((dv, t))
        prio = {"dash": 4, "bus": 3, "white": 2, "yellow": 2, "sep": 1, "barrier": 1}
        hits.sort(key=lambda z: z[0])
        merged = []
        for dv, t in hits:
            if merged and dv - merged[-1][0] < 0.45:
                if prio.get(t, 0) > prio.get(merged[-1][1], 0):
                    merged[-1] = (0.5 * (merged[-1][0] + dv), t)
                else:
                    merged[-1] = (0.5 * (merged[-1][0] + dv), merged[-1][1])
            else:
                merged.append((dv, t))
        obs[i] = merged
        for dv, t in merged:
            if t in ("white", "barrier", "sep", "yellow"):
                if 0.05 <= dv - d_R[i] <= 6.2:
                    raw_solid_right.append((i, dv, dv - d_R[i]))
                if 0.05 <= d_L[i] - dv <= 1.6:
                    raw_solid_left.append((i, dv, d_L[i] - dv))

    # 7. 物理边界精确收口到油漆路缘线（剔除非机动车道、路内停车带与窄路肩/边沟，消除 0.3–0.6 m 偏移）
    d_R_motor = d_R.copy()
    d_L_motor = d_L.copy()

    # 7a. 显式非机动车道分隔实线裁剪：找紧邻自行车符号内侧（左侧）的第一条实线（最小 dv > db + 0.25）
    bike_trimmed = False
    if proj_bikes and raw_solid_right:
        db_max = max(db for _, db in proj_bikes)
        bike_by_i = {}
        for i, dv, gap in raw_solid_right:
            # 必须严格位于自行车符号左侧（内侧），且距离自行车符号不超过 3.0 m（防越过机动车道切到黄线）
            if dv >= db_max + 0.25 and (dv - db_max) <= 3.0:
                if i not in bike_by_i or dv < bike_by_i[i][0]:
                    bike_by_i[i] = (dv, gap)
        if len(bike_by_i) >= max(2, int(0.15 * M)):
            outer_has_arrow = any(
                da < bike_by_i[min(bike_by_i, key=lambda k: abs(ss[k] - sa))][0] - 0.3
                for sa, da, _, _ in proj_arrows
            )
            if not outer_has_arrow:
                ks = sorted(bike_by_i.keys())
                vals = [bike_by_i[k][0] for k in ks]
                interp_r = np.interp(idx, ks, vals)
                for i in range(max(0, ks[0] - 3), min(M, ks[-1] + 4)):  # 只在实线实际出现的区段内收口，不外推
                    if d_L_motor[i] - interp_r[i] >= 2.35:
                        d_R_motor[i] = max(d_R_motor[i], interp_r[i])
                bike_trimmed = True

    if (not bike_trimmed) and raw_solid_right:
        by_i = {}
        for i, dv, gap in raw_solid_right:
            if gap > 3.2:
                continue
            if i not in by_i or gap < by_i[i][1]:
                by_i[i] = (dv, gap)
        if by_i:
            gaps = [g for _, g in by_i.values()]
            med_gap = float(np.median(gaps))
            outer_has_arrow = any(
                da < by_i[min(by_i, key=lambda k: abs(ss[k] - sa))][0] - 0.3
                for sa, da, _, _ in proj_arrows
            )
            outer_has_marked_old = sum(
                1 for i, (dv, _) in by_i.items()
                if any(hm and dc < dv - 0.5 for dc, _, hm in old_obs[i])
            ) >= max(2, int(0.15 * len(by_i)))
            outer_has_park = False
            if parking is not None and not parking.is_empty and len(by_i) >= 2:
                test_pts = [Point(P[i] + N[i] * (0.5 * (d_R[i] + dv))) for i, (dv, _) in by_i.items()]
                outer_has_park = sum(1 for pt in test_pts if parking.buffer(0.4).contains(pt)) >= max(1, int(0.25 * len(test_pts)))

            trim_right = (not outer_has_arrow) and (not outer_has_marked_old) and (
                (med_gap <= 1.35 and len(by_i) >= max(2, int(0.15 * M)))
                or (outer_has_park and len(by_i) >= 2)
                or (med_gap <= 2.35 and len(by_i) >= max(3, int(0.32 * M)))
            )
            if trim_right:
                ks = sorted(by_i.keys())
                vals = [by_i[k][0] for k in ks]
                interp_r = np.interp(idx, ks, vals)
                for i in range(max(0, ks[0] - 3), min(M, ks[-1] + 4)):
                    if d_L_motor[i] - interp_r[i] >= 2.35:
                        d_R_motor[i] = max(d_R_motor[i], interp_r[i])

    if raw_solid_left:
        by_il = {}
        for i, dv, gap in raw_solid_left:
            if i not in by_il or gap < by_il[i][1]:
                by_il[i] = (dv, gap)
        outer_has_arrow_l = any(
            da > by_il[min(by_il, key=lambda k: abs(ss[k] - sa))][0] + 0.3
            for sa, da, _, _ in proj_arrows
        )
        outer_has_marked_old_l = sum(
            1 for i, (dv, _) in by_il.items()
            if any(hm and dc > dv + 0.5 for dc, _, hm in old_obs[i])
        ) >= max(2, int(0.15 * len(by_il)))
        if (not outer_has_arrow_l) and (not outer_has_marked_old_l) and len(by_il) >= max(2, int(0.18 * M)):
            ks = sorted(by_il.keys())
            vals = [by_il[k][0] for k in ks]
            interp_l = np.interp(idx, ks, vals)
            for i in range(max(0, ks[0] - 3), min(M, ks[-1] + 4)):
                if interp_l[i] - d_R_motor[i] >= 2.35:
                    d_L_motor[i] = min(d_L_motor[i], interp_l[i])

    # 8. 过滤内部标线观测 & 估计本车行道典型车道宽 w_cw
    obs_in = []
    cw_gaps = []
    for i in range(M):
        lo, hi = d_R_motor[i] + 1.1, d_L_motor[i] - 1.1
        inner = [(dv, t) for dv, t in obs[i] if lo <= dv <= hi]
        obs_in.append(inner)
        pts_d = [d_R_motor[i]] + [dv for dv, _ in inner] + [d_L_motor[i]]
        for a, b in zip(pts_d[:-1], pts_d[1:]):
            if 2.4 <= b - a <= 4.3:
                cw_gaps.append(b - a)
        for _, wl, hm in old_obs[i]:
            if hm and 2.4 <= wl <= 4.3:
                cw_gaps.append(wl)

    w_cw = float(np.clip(np.median(cw_gaps), 2.75, 3.9)) if len(cw_gaps) >= 6 else float(np.clip(w_global, 2.85, 3.8))
    return dict(
        M=M, ds=ds, ss=ss, P=P, T=T, N=N,
        d_R=d_R_motor, d_L=d_L_motor,
        d_R_raw=d_R, d_L_raw=d_L,
        obs=obs_in, obs_all=obs, old_obs=old_obs, arrows=proj_arrows, bikes=proj_bikes,
        w_cw=w_cw, axis=axis,
    )


# ---------------------------------------------------------------- 2. 单断面 1D 分界匹配 + 纵向 Viterbi DP

# Frenet 参考轴改用车行道内最长纵向标线（v20a：压线 -12%，跳变/退回均减少）。LANEGEN_MARK_AXIS=0 关闭。
MARK_AXIS = os.environ.get("LANEGEN_MARK_AXIS", "1") != "0"
# Phase 3 切片：有标线车行道走标线链车道带剖分。实验，默认关（v20b：压线再 -23%，但跳变口 9→19、
# center_err_p90 0.04→0.09、平移稳定 P10 0.92→0.87）。LANEGEN_RIBBON=1 打开。
RIBBON = os.environ.get("LANEGEN_RIBBON", "0") == "1"
_MIN_AVG_WIDTH = 2.25  # 多车道断面可行的最小平均宽度（米）
_SNAP_K = 0.8          # 分界吸附标线的横向搜索半径 / 车道宽（0.55 时斜偏走廊吸附不到真实标线）
_TR = (11.0, 24.0, 45.0)
_MARK_TOPO = True   # 车道数由标线拓扑决定（v17）
_MARK_W = 6.0      # 有分界线断面上 |K - (分界线数+1)| 的每车道罚分
_TR_LOCK = 600.0   # 前后都有标线、却没有分界线起止/箭头组数变化时改变车道数的代价（实际等于禁止）  # 车道数转移代价：宽度+标线双证据 / 单证据 / 无证据（实测 14/40/80 对增减数几乎无影响）


def _solve_section_dividers(dR, dL, obs_i, old_i, K, w_cw, near_arrows):
    """给定断面 [dR, dL]、内部标线观测 obs_i、有标线旧车道中点 old_i、候选车道数 K，
    用精确 1D DP 匹配 K-1 条内部分界线，返回 (综合代价, bounds[K+1], snapped_mask[K+1])。"""
    W = dL - dR
    marked_old = [dc for dc, _, hm in old_i if hm and dR - 0.3 <= dc <= dL + 0.3]

    if K == 1:
        b = np.array([dR, dL])
        sn = np.array([True, True])
        if W < 2.1 or W > 5.4:
            return np.inf, b, sn
        cost = 1.6 * ((W - w_cw) / 0.65) ** 2
        for dv, t in obs_i:
            if dR + 0.3 * W < dv < dL - 0.3 * W:
                cost += 5.5 if t == "dash" else 2.8
        c0 = 0.5 * (dR + dL)
        for da in near_arrows:
            if abs(da - c0) < 0.32 * W:
                cost -= 2.0
        if len(marked_old) == 1 and abs(marked_old[0] - c0) < 0.38 * W:
            cost -= 2.8
        elif len(marked_old) >= 2:
            cost += 5.0 * (len(marked_old) - 1)
        return cost, b, sn

    avg_w = W / K
    if avg_w < _MIN_AVG_WIDTH or avg_w > 4.75:
        return np.inf, np.linspace(dR, dL, K + 1), np.zeros(K + 1, bool)

    nd = K - 1
    nm = len(obs_i)
    u = np.array([dR + k * avg_w for k in range(1, K)])
    mark_reward = {"dash": 3.4, "bus": 2.8, "white": 2.2, "yellow": 2.0, "sep": 2.0, "barrier": 1.8}

    INF = 1e9
    dp = np.full((nd + 1, nm + 1), INF)
    choice = np.full((nd + 1, nm + 1), -2, dtype=int)
    prev_st = np.full((nd + 1, nm + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(nd):
        for j in range(nm + 1):
            base = dp[k, j]
            if base >= INF * 0.5:
                continue
            if base < dp[k + 1, j]:
                dp[k + 1, j] = base
                choice[k + 1, j] = -1
                prev_st[k + 1, j] = j
            for m in range(j, nm):
                dv, t = obs_i[m]
                if abs(dv - u[k]) > _SNAP_K * w_cw:
                    continue
                c_snap = -mark_reward.get(t, 2.0) + 0.5 * ((dv - u[k]) / w_cw) ** 2
                val = base + c_snap
                if val < dp[k + 1, m + 1]:
                    dp[k + 1, m + 1] = val
                    choice[k + 1, m + 1] = m
                    prev_st[k + 1, m + 1] = j

    best_j = int(np.argmin(dp[nd]))
    b = np.full(K + 1, np.nan)
    sn = np.zeros(K + 1, bool)
    b[0], b[K] = dR, dL
    sn[0], sn[K] = True, True
    cur_j = best_j
    used_m = set()
    for k in range(nd, 0, -1):
        ch = int(choice[k, cur_j])
        if ch >= 0:
            b[k] = obs_i[ch][0]
            sn[k] = True
            used_m.add(ch)
        cur_j = int(prev_st[k, cur_j])

    anc = np.nonzero(sn)[0]
    for a_i, b_i in zip(anc[:-1], anc[1:]):
        if b_i - a_i > 1:
            b[a_i:b_i + 1] = np.linspace(b[a_i], b[b_i], b_i - a_i + 1)

    widths = np.diff(b)
    if np.any(widths < 2.05) or np.any(widths > 5.1):
        return np.inf, b, sn

    cost = float(dp[nd, best_j])
    for wk in widths:
        if wk < 2.3 or wk > 4.5:
            cost += 6.0 + 2.0 * ((wk - w_cw) / 0.5) ** 2
        else:
            cost += 1.4 * ((wk - w_cw) / 0.58) ** 2

    # 未被吸附的内部虚线落进某车道中间 → 重罚
    for m_idx, (dv, t) in enumerate(obs_i):
        if m_idx in used_m:
            continue
        k_in = int(np.searchsorted(b, dv)) - 1
        if 0 <= k_in < K:
            wk = widths[k_in]
            if min(dv - b[k_in], b[k_in + 1] - dv) > 0.28 * wk:
                cost += 5.5 if t == "dash" else 2.2

    # 箭头居中与防切箭头代价
    for da in near_arrows:
        if not (dR - 0.3 <= da <= dL + 0.3):
            continue
        k_in = int(np.clip(np.searchsorted(b, da) - 1, 0, K - 1))
        wk = widths[k_in]
        center = 0.5 * (b[k_in] + b[k_in + 1])
        off = abs(da - center)
        if off < 0.28 * wk:
            cost -= 2.5
        elif min(abs(da - b[k_in]), abs(da - b[k_in + 1])) < 0.20 * w_cw:
            cost += 5.0

    # 有标线支撑的 old_lanes 中点对齐奖励与防切车道惩罚（锁定 0.02 m 标线中点车道数与位置）
    if marked_old:
        matched_lanes = set()
        for dc in marked_old:
            k_in = int(np.clip(np.searchsorted(b, dc) - 1, 0, K - 1))
            wk = widths[k_in]
            center = 0.5 * (b[k_in] + b[k_in + 1])
            off = abs(dc - center)
            if off < 0.35 * wk and k_in not in matched_lanes:
                matched_lanes.add(k_in)
                cost += -2.8 + 1.2 * (off / max(0.45 * w_cw, 1.0)) ** 2
            elif min(abs(dc - b[k_in]), abs(dc - b[k_in + 1])) < 0.22 * w_cw:
                cost += 5.5
        if K < len(marked_old):
            cost += 4.5 * (len(marked_old) - K)

    return cost, b, sn


def _viterbi_partition(F, role):
    """在 Frenet 断面序列 F 上运行纵向 Viterbi DP，输出最优车道数序列 K* 及每个断面的分界轨迹。"""
    M = F["M"]
    d_R, d_L, w_cw = F["d_R"], F["d_L"], F["w_cw"]
    W = d_L - d_R
    K_max = max(1, math.floor(float(np.max(W)) / _MIN_AVG_WIDTH))

    ss = F["ss"]
    L = ss[-1]
    near_arr = [[] for _ in range(M)]
    min_k_arr = np.ones(M, int)
    for i in range(M):
        s_i = ss[i]
        win = 16.0 if (role == "approach" and L - s_i <= 32.0) else 12.0
        das = [da for sa, da, _, _ in F["arrows"] if abs(sa - s_i) <= win and d_R[i] - 0.4 <= da <= d_L[i] + 0.4]
        near_arr[i] = das  # 居中/防切约束：窗口宽一些（箭头横向位置沿平行车道有效）
        # 车道数下限：不往上游倒灌（箭头上游只认 4 m，防展宽段箭头组数灌进上游窄路段）；
        # 进口道往下游认到 30 m —— 有箭头的车道一直通到停止线，箭头与停止线之间不会少车道
        lo_s, hi_s = (s_i - 30.0, s_i + 4.0) if role == "approach" else (s_i - 4.0, s_i + 4.0)
        das_k = sorted(da for sa, da, _, _ in F["arrows"] if lo_s <= sa <= hi_s and d_R[i] - 0.4 <= da <= d_L[i] + 0.4)
        if das_k:
            min_k_arr[i] = 1 + sum(1 for k in range(1, len(das_k)) if das_k[k] - das_k[k - 1] > 0.52 * w_cw)
    n_inner = np.array([len(o) for o in F["obs"]], float)

    E = np.full((M, K_max + 1), 1e5)
    B_cand = [[None] * (K_max + 1) for _ in range(M)]
    SN_cand = [[None] * (K_max + 1) for _ in range(M)]
    for i in range(M):
        for K in range(1, K_max + 1):
            c, b, sn = _solve_section_dividers(d_R[i], d_L[i], F["obs"][i], F["old_obs"][i], K, w_cw, near_arr[i])
            if K < min_k_arr[i]:
                c += 14.0 * (min_k_arr[i] - K)
            if _MARK_TOPO and n_inner[i] > 0:
                c += _MARK_W * abs(K - (n_inner[i] + 1))  # 有分界线的断面：车道数 = 分界线数 + 1（软约束）
            E[i, K] = c
            B_cand[i][K] = b
            SN_cand[i][K] = sn

        # 宽度越界的状态仍允许跨过局部坏断面。保留车道数之间的相对差异，
        # 即使该行没有可行状态，也由宽度、箭头和标线决定排序。
        costs = E[i, 1:]
        invalid = ~np.isfinite(costs)
        feasible = costs[~invalid]
        counts = np.arange(1, K_max + 1)[invalid]
        relaxed = np.abs(counts - W[i] / w_cw) + 14.0 * np.maximum(min_k_arr[i] - counts, 0)
        if _MARK_TOPO and n_inner[i] > 0:
            relaxed += _MARK_W * np.abs(counts - (n_inner[i] + 1))
        costs[invalid] = (feasible.min() if feasible.size else 0.0) + 300.0 + relaxed
    dp = np.full((M, K_max + 1), 1e9)
    prev = np.zeros((M, K_max + 1), dtype=int)
    dp[0, 1:] = E[0, 1:]

    for i in range(1, M):
        w_bef = float(np.median(W[max(0, i - 8):i]))
        w_aft = float(np.median(W[i:min(M, i + 8)]))
        # 内部纵向标线条数与箭头组数的前后变化：真实增减一条车道会多/少一条分界线（或多/少一组箭头）
        m_bef = float(np.median(n_inner[max(0, i - 8):i]))
        m_aft = float(np.median(n_inner[i:min(M, i + 8)]))
        a_bef = int(np.max(min_k_arr[max(0, i - 8):i]))
        a_aft = int(np.max(min_k_arr[i:min(M, i + 8)]))
        # 前后都是有标线路段：车道数只能在某条分界线起止（或箭头组数变化）处改变，宽度起伏不算
        marked_both = (float(np.mean(n_inner[max(0, i - 8):i] > 0)) >= 0.5
                       and float(np.mean(n_inner[i:min(M, i + 8)] > 0)) >= 0.5)
        for K in range(1, K_max + 1):
            best_v, best_p = 1e9, K
            for Kp in range(max(1, K - 1), min(K_max, K + 1) + 1):
                if dp[i - 1, Kp] >= 1e8:
                    continue
                if Kp == K:
                    tr = 0.0
                else:
                    sgn = 1.0 if K > Kp else -1.0
                    dw = (w_aft - w_bef) * sgn
                    width_ev = dw >= 0.72 * w_cw  # 接近一整条车道的物理宽度变化
                    mark_ev = (m_aft - m_bef) * sgn >= 1.0 or (a_aft - a_bef) * sgn >= 1
                    tr = _TR[0] if (width_ev and mark_ev) else (_TR[1] if (width_ev or mark_ev) else _TR[2])
                    if _MARK_TOPO and marked_both and not mark_ev:
                        tr = _TR_LOCK
                v = dp[i - 1, Kp] + tr + E[i, K]
                if v < best_v:
                    best_v, best_p = v, Kp
            dp[i, K] = best_v
            prev[i, K] = best_p

    K_seq = np.ones(M, dtype=int)
    K_seq[-1] = int(np.argmin(dp[-1, 1:]) + 1)
    if dp[-1, K_seq[-1]] >= 1e8:
        K_seq[:] = max(1, int(round(float(np.median(W)) / w_cw)))
    else:
        for i in range(M - 1, 0, -1):
            K_seq[i - 1] = prev[i, K_seq[i]]

    # 消除短毛刺段（中间段至少 12 m，起止端至少 8 m；并消除中间“先增后减”且 < 24 m 且无箭头的伪港湾）
    changed = True
    while changed:
        changed = False
        runs = []
        s0 = 0
        for i in range(1, M + 1):
            if i == M or K_seq[i] != K_seq[s0]:
                runs.append([s0, i, int(K_seq[s0])])
                s0 = i
        if len(runs) <= 1:
            break
        for r_idx, (a, b, kv) in enumerate(runs):
            is_end = (r_idx == 0 or r_idx == len(runs) - 1)
            # 出口最前端（路口出口导圆段）若无箭头，要求至少 18 m 才允许单独切分，防 12 m 导圆假 lane_drop
            if role == "exit" and r_idx == 0 and max(min_k_arr[a:b]) < kv:
                min_len = 9
            else:
                min_len = 5 if is_end else 7
            # 中间短凸起（K > 左右两侧，跨度 < 12 断面 = 24 m 且无独立新增箭头）
            short_bump = (
                not is_end
                and kv > runs[r_idx - 1][2]
                and kv > runs[r_idx + 1][2]
                and (b - a) < 12
                and max(min_k_arr[a:b]) < kv
            )
            # 中间短凹陷（K < 左右两侧，< 24 m）：多是标线被遮挡/路侧开口把断面拉窄，按两侧车道数补平
            # （补平后代价不能明显变差：E 均值增加 ≤ 25）
            short_dip = (
                not is_end
                and kv < runs[r_idx - 1][2]
                and kv < runs[r_idx + 1][2]
                and (b - a) < 12
                and runs[r_idx - 1][2] == runs[r_idx + 1][2]
                and float(np.mean(E[a:b, kv + 1])) <= float(np.mean(E[a:b, kv])) + 25.0
            )
            if short_dip:
                K_seq[a:b] = kv + 1
                changed = True
                break
            # 单调阶梯（3→4→5）中间的过渡段不能删：删了会在两侧直接跳 2 条车道（jump_3_to_5）
            stair = (not is_end) and abs(runs[r_idx - 1][2] - runs[r_idx + 1][2]) >= 2
            if ((b - a < min_len) or short_bump) and not stair:
                cand_k = None
                if not is_end:
                    cand_k = runs[r_idx - 1][2] if (runs[r_idx - 1][1] - runs[r_idx - 1][0]) >= (runs[r_idx + 1][1] - runs[r_idx + 1][0]) else runs[r_idx + 1][2]
                elif r_idx == 0:
                    cand_k = runs[1][2]
                else:
                    cand_k = runs[-2][2]
                if cand_k is not None and B_cand[a][cand_k] is not None:
                    K_seq[a:b] = cand_k
                    changed = True
                    break

    return K_seq, B_cand, SN_cand


# ---------------------------------------------------------------- 3. 边界平滑、标线中点残差锚定与连续车道重构

def _section_lanes(F, i0, i1, K, B_cand, SN_cand, anchor=True):
    """在恒定车道数区间 [i0, i1] 内：
    1) 平滑各条 DP 分界轨迹；
    2) 用有标线支撑的 old_obs 做偏移残差场（offset-field）锚定，把中心线精度拉回到标线中点 (~0.03–0.08 m)，
       同时在标线缺失段保持沿道路曲率平滑插值、绝不弯折。"""
    n = i1 - i0 + 1
    w_cw = F["w_cw"]
    b_mat = np.zeros((n, K + 1))
    sn_mat = np.zeros((n, K + 1), bool)
    for t_i, i in enumerate(range(i0, i1 + 1)):
        b_i = B_cand[i][K]
        sn_i = SN_cand[i][K]
        if b_i is None or len(b_i) != K + 1:
            b_i = np.linspace(F["d_R"][i], F["d_L"][i], K + 1)
            sn_i = np.zeros(K + 1, bool)
            sn_i[0] = sn_i[-1] = True
        b_mat[t_i] = b_i
        sn_mat[t_i] = sn_i

    idx = np.arange(n)
    for k in range(1, K):
        sn = sn_mat[:, k]
        if sn.sum() >= 2:
            first, last = int(np.argmax(sn)), n - 1 - int(np.argmax(sn[::-1]))
            mid_miss = (~sn) & (idx > first) & (idx < last)
            if mid_miss.any():
                b_mat[mid_miss, k] = np.interp(idx[mid_miss], idx[sn], b_mat[sn, k])
            if last < n - 1 and (n - 1 - last) <= 8:
                frac = (b_mat[last, k] - b_mat[last, 0]) / max(b_mat[last, K] - b_mat[last, 0], 1.0)
                for j in range(last + 1, n):
                    b_mat[j, k] = b_mat[j, 0] + frac * (b_mat[j, K] - b_mat[j, 0])
            if first > 0 and first <= 8:
                frac = (b_mat[first, k] - b_mat[first, 0]) / max(b_mat[first, K] - b_mat[first, 0], 1.0)
                for j in range(0, first):
                    b_mat[j, k] = b_mat[j, 0] + frac * (b_mat[j, K] - b_mat[j, 0])

    b_raw = b_mat.copy()
    for k in range(K + 1):
        b_mat[:, k] = _smooth_1d(b_mat[:, k], win=7)
    # 标线硬锚点：DP 已吸附到真实纵向标线的内部分界，直接用标线观测值（平滑只用于桥接缺线段）
    for k in range(1, K):
        hard = sn_mat[:, k] & (np.abs(b_raw[:, k] - b_mat[:, k]) < 0.6)
        b_mat[hard, k] = b_raw[hard, k]
    for t_i in range(n):
        for k in range(1, K):
            lo = b_mat[t_i, k - 1] + 2.05
            hi = b_mat[t_i, K] - 2.05 * (K - k)
            if lo <= hi:
                b_mat[t_i, k] = float(np.clip(b_mat[t_i, k], lo, hi))

    centers_d = 0.5 * (b_mat[:, :-1] + b_mat[:, 1:])  # (n, K)
    widths = b_mat[:, 1:] - b_mat[:, :-1]             # (n, K)

    # 偏移残差场锚定：将有标线支撑的 old_obs 一对一匹配到 DP 车道，平滑修正系统性路缘/边沟偏移
    delta = np.full((n, K), np.nan)
    anch_w = np.full((n, K), np.nan)
    for t_i, i in enumerate(range(i0, i1 + 1)):
        m_obs = [(dc, wl) for dc, wl, hm in F["old_obs"][i] if hm]
        if not m_obs:
            continue
        # 按距离从小到大贪心一对一匹配
        pairs = []
        for oi, (dc, wl) in enumerate(m_obs):
            for k in range(K):
                err = abs(dc - centers_d[t_i, k])
                if err < 0.33 * w_cw:
                    pairs.append((err, oi, k, dc, wl))
        pairs.sort(key=lambda z: z[0])
        used_o, used_k = set(), set()
        for err, oi, k, dc, wl in pairs:
            if oi in used_o or k in used_k:
                continue
            used_o.add(oi)
            used_k.add(k)
            delta[t_i, k] = dc - centers_d[t_i, k]
            anch_w[t_i, k] = wl

    for k in range(K if anchor else 0):  # 车道带模式下中心已居中于真实标线，不再向 Step 1 旧车道拉（审计 Bug 10）
        ok = np.isfinite(delta[:, k])
        if ok.sum() >= 2:
            # 剔除相对本车道中位残差偏离 > 0.75 m 的跨车道错配毛刺
            med_d = float(np.median(delta[ok, k]))
            ok = ok & (np.abs(delta[:, k] - med_d) <= 0.75)
        if ok.sum() >= 2:
            d_k = delta[ok, k]
            if len(d_k) >= 3:
                d_k = median_filter(d_k, size=3, mode="nearest")
            d_interp = np.interp(idx, idx[ok], d_k)
            dist_to_ok = np.min(np.abs(idx[:, None] - idx[ok][None, :]), axis=1)
            weight = np.clip(1.0 - np.maximum(0.0, dist_to_ok - 6.0) / 14.0, 0.0, 1.0)
            centers_d[:, k] = centers_d[:, k] + np.clip(d_interp, -0.38 * w_cw, 0.38 * w_cw) * weight
            centers_d[:, k] = _smooth_1d(centers_d[:, k], win=7)
            # 更新宽度估计（优先采用有标线支撑的宽度中位数）
            wok = np.isfinite(anch_w[:, k])
            if wok.sum() >= 2:
                w_med_k = float(np.clip(np.median(anch_w[wok, k]), 2.5, 4.3))
                widths[wok, k] = anch_w[wok, k]
                widths[~wok, k] = 0.5 * (widths[~wok, k] + w_med_k)

    # 最终单调非交叉保护：相邻车道中心间距 ≥ 2.15 m
    for t_i in range(n):
        for k in range(1, K):
            if centers_d[t_i, k] < centers_d[t_i, k - 1] + 2.15:
                mid = 0.5 * (centers_d[t_i, k - 1] + centers_d[t_i, k])
                centers_d[t_i, k - 1] = mid - 1.08
                centers_d[t_i, k] = mid + 1.08

    return centers_d, widths, b_mat


def _snap_approach_end(line, core_stops, core_poly, nondrive, w):
    """进口车道末端精确对齐本路口停止线：
    - 若已穿过某条横向核心停止线，直接精准裁齐在该停止线处（绝不越线冲进斑马线）；
    - 若止于停止线前方/略超停止线，沿稳定切向直线延伸或回裁到斜停止线，严禁穿入核心或压岛。"""
    if line.length < 4.0 or not core_stops:
        return line
    L = line.length
    inner_core = core_poly.buffer(-0.4)

    # Step A：检测是否已直接与某条本路口停止线相交（在车道末尾 28 m 内，且与停止线呈横交而非平行）
    best_cut = None
    for s_ in core_stops:
        if s_.length < 1.5 or not line.intersects(s_):
            continue
        x = line.intersection(s_)
        sd = _unit(np.asarray(s_.coords[-1]) - np.asarray(s_.coords[0]))
        for q in getattr(x, "geoms", [x]):
            if q.is_empty:
                continue
            pt = q if q.geom_type == "Point" else q.representative_point()
            s_hit = float(line.project(pt))
            if s_hit < max(4.0, L - 28.0):
                continue
            t_hit = _tan(line, s_hit, min(3.0, 0.3 * L))
            if abs(float(np.dot(t_hit, sd))) > 0.78:
                continue  # 与停止线近平行，不是横跨停止线
            # 取沿行驶方向最先碰到的本路口停止线（或离末端最近的）
            if best_cut is None or s_hit < best_cut:
                best_cut = s_hit
    if best_cut is not None and best_cut >= 4.0:
        return substring(line, 0.0, best_cut)

    # Step B：未直接相交时，用稳定末段切向（取倒数 2–10 m，避开端头微小摆动）射线求交停止线
    p = np.asarray(line.coords[-1])
    p_ref0 = np.asarray(line.interpolate(max(0.0, L - 10.0)).coords[0])
    p_ref1 = np.asarray(line.interpolate(max(0.5, L - 1.5)).coords[0])
    t = _unit(p_ref1 - p_ref0)
    if np.linalg.norm(t) < 1e-6:
        t = _tan(line, L, min(3.0, 0.4 * L))

    best = None
    for s_ in core_stops:
        if s_.length < 1.5:
            continue
        a, b = np.asarray(s_.coords[0]), np.asarray(s_.coords[-1])
        d_ab = b - a
        sd = _unit(d_ab)
        if abs(float(np.dot(t, sd))) > 0.80:
            continue
        den = t[0] * d_ab[1] - t[1] * d_ab[0]
        if abs(den) < 1e-9:
            continue
        w_ = a - p
        dist = (w_[0] * d_ab[1] - w_[1] * d_ab[0]) / den
        u = (w_[0] * t[1] - w_[1] * t[0]) / den
        Ls = np.linalg.norm(d_ab)
        tol = 0.48 * w / max(Ls, 1e-6)
        if -10.0 <= dist <= 16.0 and -tol <= u <= 1.0 + tol:
            q = p + t * dist
            if dist > 0.1:
                seg = LineString([p, q])
                if not inner_core.is_empty and seg.intersection(inner_core).length > 0.5:
                    continue
                if nondrive is not None and not nondrive.is_empty and seg.intersection(nondrive).length > 0.3:
                    continue
            if best is None or abs(dist) < abs(best[0]):
                best = (dist, q)

    if best is not None:
        dist, q = best
        if dist > 0.1:
            line = LineString(list(line.coords) + [tuple(q)])
        elif dist < -0.2 and L + dist >= 4.0:
            line = substring(line, 0.0, L + dist)
    return line


def _on_bus_marks(geom, B, w):
    """车道中线 0.35 W 内的公交专用车道标记（字样锚线）累计 ≥ 8 m 或 ≥ 20% 车道长 → 公交专用道。"""
    bm = B.get("bus_marks") or []
    if not bm or geom.length < 4.0:
        return False
    buf = geom.buffer(0.35 * w)
    ln = sum(buf.intersection(m).length for m in bm if m.intersects(buf))
    return ln >= min(8.0, 0.2 * geom.length)


def _clip_safe(geom, core_poly, nondrive, drivable, role, nd_pad=-0.3):
    """硬门禁保护：确保任何生成的车道绝不侵入核心内部、安全岛/导流区或离开路面。"""
    g = geom
    inner_core = core_poly.buffer(-0.5)
    if role != "slip" and not inner_core.is_empty and g.intersects(inner_core):
        rest = g.difference(inner_core)
        parts = [p for p in _parts(rest, "LineString") if p.length >= 2.0]
        if not parts:
            return None
        e0, e1 = Point(g.coords[0]), Point(g.coords[-1])
        far = e0 if e0.distance(core_poly) > e1.distance(core_poly) else e1
        g = min(parts, key=lambda p: p.distance(far))
    if nondrive is not None and not nondrive.is_empty:
        nd_in = nondrive.buffer(nd_pad)
        if not nd_in.is_empty and g.intersects(nd_in):
            rest = g.difference(nd_in)
            parts = [p for p in _parts(rest, "LineString") if p.length >= 3.0]
            if not parts:
                return None
            g = max(parts, key=lambda p: p.length)
    if drivable is not None and not drivable.is_empty:
        dv = drivable.buffer(0.4)
        if g.difference(dv).length > 0.5:
            rest = g.intersection(dv)
            parts = [p for p in _parts(rest, "LineString") if p.length >= 3.0]
            if not parts:
                return None
            g = max(parts, key=lambda p: p.length)
    return g if g.length >= 4.0 else None



def _align_lanes(prev_c, cur_c):
    """两组右→左排列的车道中心横向位置做保序配对，配对数 = min(两侧车道数)，横移绝对值之和最小。
    多出来的一侧在任意位置留空（新增 / 消失车道）。车道数差 > 3 时退回贪心。"""
    from itertools import combinations
    a, b = [float(x) for x in prev_c], [float(x) for x in cur_c]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return []
    swap = na > nb
    if swap:
        a, b, na, nb = b, a, nb, na
    best = None
    if nb - na <= 3:
        for skip in combinations(range(nb), nb - na):
            keep = [j for j in range(nb) if j not in skip]
            c = sum(abs(a[i] - b[j]) for i, j in enumerate(keep))
            if best is None or c < best[0]:
                best = (c, list(zip(range(na), keep)))
        pairs = best[1]
    else:  # 保序贪心：每条依次取右侧尚未用过的最近车道
        pairs, lo = [], 0
        for i in range(na):
            cand = range(lo, nb - (na - 1 - i))
            j = min(cand, key=lambda j: abs(a[i] - b[j]))
            pairs.append((i, j))
            lo = j + 1
    return [(j, i) for i, j in pairs] if swap else pairs


def partition_carriageway(cw, cw_old_lanes, B, core_poly, core_stops, arrows, bikes, parking, w_global):
    """对单条物理车行道运行 Step 3 Frenet DP 剖分，返回 (lanes_list, 分段切点 s 列表)。"""
    role = cw["role"]
    F = _sample_frenet(cw, cw_old_lanes, B, arrows, bikes, parking, w_global)
    if F is None:
        return None, []
    cw["frenet_axis"] = F["axis"]
    if float(np.median(F["d_L"] - F["d_R"])) < 2.2:
        return None, []  # 机动车道宽都不够的窄缝（如 0.2–1.1 m 的缝隙），不剖分

    rib = _ribbon.ribbon_partition(F, role) if RIBBON else None
    if rib is not None:
        K_seq, B_cand, SN_cand, _ = rib
        cw["partition"] = "ribbon"
    else:
        K_seq, B_cand, SN_cand = _viterbi_partition(F, role)
        cw["partition"] = "viterbi"
    M = F["M"]
    P, N, ss = F["P"], F["N"], F["ss"]

    raw_secs = []
    s0 = 0
    for i in range(1, M + 1):
        if i == M or K_seq[i] != K_seq[s0]:
            raw_secs.append((s0, i - 1, int(K_seq[s0])))
            s0 = i

    sec_data = []
    for i0, i1, K in raw_secs:
        cd, wd, bm = _section_lanes(F, i0, i1, K, B_cand, SN_cand, anchor=rib is None)
        sec_data.append(dict(i0=i0, i1=i1, K=K, cd=cd, wd=wd, bm=bm))

    # 车道追踪：每条 track 记录自己经过的 (断面 i, 本段内右→左序号 k, 本段车道数 K)。
    # 车道在各分段里的序号会随右侧增减而变化，不能只存一个 lane_index；
    # 最终的分段 / 序号 / 连接映射统一由 build_sections() 按几何重建（DP 车道与兜底车道同一套规则）。
    active = {}
    all_tracks = []
    cuts = []

    def _new_track(add_event):
        tr = dict(pts=[], widths=[], st=[], add_event=add_event, drop_event=False)
        all_tracks.append(tr)
        return tr

    def _extend(tr, sd, sec_pts, k):
        tr["pts"].append(sec_pts[k])
        tr["widths"].extend(sd["wd"][:, k])
        tr["st"].extend((i, k, sd["K"], t_i) for t_i, i in enumerate(range(sd["i0"], sd["i1"] + 1)))

    for s_idx, sd in enumerate(sec_data):
        i0, i1, K = sd["i0"], sd["i1"], sd["K"]
        cd = sd["cd"]
        sec_pts = [P[i0:i1 + 1] + N[i0:i1 + 1] * cd[:, k:k + 1] for k in range(K)]
        new_active = {}
        if s_idx == 0:
            for k in range(K):
                tr = _new_track(False)
                _extend(tr, sd, sec_pts, k)
                new_active[k] = tr
        else:
            cuts.append(float(ss[i0]))
            prev_sd = sec_data[s_idx - 1]
            Kp = prev_sd["K"]
            # 段间续接：按横向位置做保序最优对齐（不再假设“只在最左/最右增减”）。
            # 内部 Y 形分叉、渠化分流时新增 / 消失车道可以在任意位置；旧做法按两侧边界噪声猜 off，
            # 猜错就把全部存续车道嫁接到隔壁车道。（实测：配对后再按横移阈值断开会增加跳变与增减事件，不做。）
            prev_cd = prev_sd["cd"][-1]
            matches = _align_lanes(prev_cd, cd[0])
            used_p, used_c = set(), set()
            for kp, kc in matches:
                used_p.add(kp); used_c.add(kc)
                _extend(active[kp], sd, sec_pts, kc)
                new_active[kc] = active[kp]
            for kp in range(Kp):
                if kp not in used_p:
                    active[kp]["drop_event"] = True
            for kc in range(K):
                if kc not in used_c:
                    tr = _new_track(True)
                    _extend(tr, sd, sec_pts, kc)
                    new_active[kc] = tr
        active = new_active

    sec_by_station = {}
    for sd in sec_data:
        for t_i, i in enumerate(range(sd["i0"], sd["i1"] + 1)):
            sec_by_station[i] = sd
    obs_all, dRr, dLr = F["obs_all"], F["d_R_raw"], F["d_L_raw"]

    def _side_stats(tr):
        """真实标线采样：每个断面看本车道左右分界附近 0.45 m 内有没有虚线 / 路缘(物理边界或护栏) / 公交线。"""
        n = dash = curb = bus_l = 0
        for i, k, K, t_i in tr["st"]:
            bm = sec_by_station[i]["bm"]
            br, bl = float(bm[t_i, k]), float(bm[t_i, k + 1])
            near_r = [t for dv, t in obs_all[i] if abs(dv - br) < 0.45]
            near_l = [t for dv, t in obs_all[i] if abs(dv - bl) < 0.45]
            n += 1
            dash += ("dash" in near_r) or ("dash" in near_l)
            edge_r = (k == 0 and abs(br - dRr[i]) < 0.35 and not near_r) or ("barrier" in near_r)
            edge_l = (k == K - 1 and abs(bl - dLr[i]) < 0.35 and not near_l) or ("barrier" in near_l)
            curb += edge_r or edge_l
            bus_l += ("bus" in near_l) or ("bus" in near_r)
        n = max(n, 1)
        return dash / n, curb / n, bus_l / n

    # 车行道轴线远端是否到了 Step 2 建模范围尽头（离核心 ≥ 100 m）：只有这种远端才算 ROI/建模截断（cut）
    ax = F["axis"]
    far_xy = np.asarray(ax.coords[0] if role == "approach" else ax.coords[-1])
    axis_capped = Point(far_xy).distance(core_poly) >= 100.0

    out_lanes = []
    for tr in all_tracks:
        xy = np.vstack(tr["pts"])
        if len(xy) < 2:
            continue
        keep = np.ones(len(xy), bool)
        keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.1
        xy = xy[keep]
        if len(xy) < 2:
            continue
        if len(xy) >= 5:
            geom = LineString(np.stack([_smooth_1d(xy[:, 0], 5), _smooth_1d(xy[:, 1], 5)], axis=1))
        else:
            geom = LineString(xy)
        # 先精准对齐/裁齐停止线，再做硬门禁保护（DP 车道对安全岛/导流区采取 0.1 m 严格外扩零侵入裁剪）
        if role == "approach" and not tr["drop_event"]:
            geom = _snap_approach_end(geom, core_stops, core_poly, B["nondrive"], F["w_cw"])
        geom = _clip_safe(geom, core_poly, B["nondrive"], B["drivable"], role, nd_pad=0.1)
        if geom is None or geom.length < 4.0:
            continue
        w_raw = float(np.median(tr["widths"]))
        if w_raw < 2.2:
            continue  # 物理上容不下机动车的窄缝，不生成车道
        w_med = float(np.clip(w_raw, 2.4, 4.4))
        min_k = min(k for _, k, _, _ in tr["st"])
        # 外侧车道若直接压在自行车符号上且无机动导向箭头 → 剔除非机动车道
        if min_k <= 1 and bikes:
            buf_b = geom.buffer(0.55 * w_med)
            if any(buf_b.contains(bp) for bp in bikes) and not any(buf_b.contains(Point(ap)) for ap, _, _ in arrows):
                continue
        f_dash, f_curb, f_bus = _side_stats(tr)
        far_pt = Point(geom.coords[0] if role == "approach" else geom.coords[-1])
        far_cut = bool(axis_capped and far_pt.distance(Point(far_xy)) <= 0.5 * float(np.max(F["d_L"] - F["d_R"])) + 8.0)
        out_lanes.append(dict(
            geom=geom,
            width=w_med,
            direction="known",
            score=4.0,
            evidence=["frenet_dp"],
            role=role,
            arm=cw["arm"],
            carriageway=cw["id"],
            dp_add=tr["add_event"],
            dp_drop=tr["drop_event"],
            far_cut=far_cut,
            n_norm=int(geom.length),
            n_edge=0,
            n_anch=0,
            yellow_pts=np.zeros((0, 2)),
            has_yellow=False,
            frac_dash=round(f_dash, 3),
            frac_curb=round(f_curb, 3),
            frac_island=0.0,
            allow="bus" if ((min_k == 0 and f_bus >= 0.5) or _on_bus_marks(geom, B, w_med)) else "all",
            slip=False,
            nonmotor_suspect=False,
            source="frenet_dp",
        ))
    return out_lanes, cuts


# ---------------------------------------------------------------- 4. 渠化右转走廊车道生成与全路口总装

def _slip_lanes(slips, B, core_poly, w_global):
    """由 Step 2 绕岛右转走廊生成平滑右转车道。返回 [(slip 对象, 车道)]，与 slip 严格成对（失败的 slip 不产出）。"""
    out = []
    for s in slips:
      for axis in [s.get("axis")] + list(s.get("extra_axes", [])):
        if axis is None or axis.length < 6.0:
            continue
        g = _clip_safe(axis, core_poly, B["nondrive"], B["drivable"], role="slip", nd_pad=0.1)
        if g is None or g.length < 6.0:
            continue
        w_s = float(np.clip(s.get("width_m") or w_global, 2.8, 4.5))
        out.append((s, dict(
            geom=g,
            width=w_s,
            direction="known",
            score=3.0,
            evidence=["bypass_slip"],
            role="slip",
            arm=None,
            carriageway=None,
            slip_from_arm=s.get("from_arm"),
            slip_to_arm=s.get("to_arm"),
            n_norm=0,
            n_edge=int(g.length),
            n_anch=0,
            yellow_pts=np.zeros((0, 2)),
            has_yellow=False,
            frac_dash=0.0,
            frac_curb=1.0,
            frac_island=1.0,
            slip=True,
            nonmotor_suspect=False,
            dp_add=False,
            dp_drop=False,
            allow="all",
            source="bypass_slip",
        )))
    return out


def _cw_matching_old_lanes(cw, old_lanes):
    """筛选空间上落入本车行道且方向一致的 old_lanes，供 Frenet 锚定使用。"""
    poly_buf = cw["polygon"].buffer(0.8)
    axis = cw["axis"]
    out = []
    for ln in old_lanes:
        if ln.get("role") == "slip":
            continue
        g = ln["geom"]
        if not g.intersects(poly_buf):
            continue
        ov = g.intersection(poly_buf).length
        if ov < min(6.0, 0.35 * g.length):
            continue
        mid = g.interpolate(0.5 * g.length)
        s_a = axis.project(mid)
        ta = _tan(axis, s_a, 2.0)
        tg = _tan(g, 0.5 * g.length, 2.0)
        if float(np.dot(ta, tg)) > 0.65:
            out.append(ln)
    return out


def _cw_eval(lanes_cand, cw_arrows, cw_stops, w_global):
    """用本车行道内部可观测真值（箭头覆盖数、停止线车道数精确匹配数）评估候选车道集合。"""
    if not lanes_cand:
        return 0, 0
    arr_cov = 0
    for ap, av in cw_arrows:
        pt = Point(ap)
        for ln in lanes_cand:
            g = ln["geom"]
            if g.distance(pt) <= 0.5 * w_global + 0.3:
                if float(np.dot(_tan(g, g.project(pt), 1.5), av)) > 0.7:
                    arr_cov += 1
                    break
    stop_ok = 0
    for s_, groups in cw_stops:
        nl = sum(1 for ln in lanes_cand if Point(ln["geom"].coords[-1]).distance(s_) < 0.5 * w_global + 1.0)
        if nl == groups:
            stop_ok += 1
    return arr_cov, stop_ok


def _dp_deviation_from_old(cw_lanes, cw_old, w_global):
    """计算 DP 车道与有标线支撑的旧车道之间的 P90 横向偏离距离（m），衡量贴标线保真度。"""
    marked = [ln["geom"] for ln in cw_old if (not ln.get("edgepair", False)) and ln.get("n_norm", 0) >= 6]
    if not marked or not cw_lanes:
        return 0.0
    dp_u = unary_union([ln["geom"] for ln in cw_lanes])
    dists = []
    for g in marked:
        if g.length < 8.0:
            continue
        ss = np.arange(3.0, g.length - 3.0, 3.0)
        for s in ss:
            pt = g.interpolate(float(s))
            d = float(dp_u.distance(pt))
            if d < 1.8 * w_global:
                dists.append(d)
    return float(np.percentile(dists, 90)) if len(dists) >= 5 else 0.0


def _select_cw_lanes(cw, cw_lanes, cw_old, core_stops, arrows, B, core_poly, w_global):
    """多假设几何模型选择（H_DP vs H_ray）：
    默认采用全局 Frenet DP（补齐无标线出口、无缝跨接标线断裂、输出精确增减分段）；
    仅当某条复杂异形车行道（如多停止线分岔/高架导流分裂）上单轴 DP 出现箭头覆盖下降、
    停止线车道数失配、或相对完整无断标线旧车道偏移 > 0.35 m 时，安全回退保留该车行道的标线中点车道。"""
    if not cw_lanes:
        fb = []
        for ln in cw_old:
            g = _clip_safe(ln["geom"], core_poly, B["nondrive"], B["drivable"], cw["role"])
            if g is not None:
                fb.append(_as_fallback(ln, g, cw, "ray_fallback"))
        return fb, False
    if not cw_old or cw.get("unmarked"):
        return cw_lanes, True

    poly_buf = cw["polygon"].buffer(0.8)
    axis = cw["axis"]
    # 1. 收集落入本车行道且同向的导向箭头
    cw_arrows = []
    for ap, av, _ in arrows:
        pt = Point(ap)
        if not poly_buf.contains(pt):
            continue
        sa = float(axis.project(pt))
        if float(np.dot(av, _tan(axis, sa, 1.5))) > 0.65:
            cw_arrows.append((ap, av))

    # 2. 收集本车行道进口对应的核心停止线及其箭头组数
    cw_stops = []
    if cw["role"] == "approach":
        for s_ in core_stops:
            if s_.length < 2.0 or s_.distance(poly_buf) > 3.5:
                continue
            # 找前向射线打到该停止线的本车行道箭头
            hits = []
            for ap, av in cw_arrows:
                ray = LineString([ap, ap + av * 80.0])
                if ray.intersects(s_):
                    x = ray.intersection(s_)
                    q = x if x.geom_type == "Point" else x.representative_point()
                    hits.append(np.asarray(q.coords[0]))
            if not hits:
                continue
            dv = _unit(np.asarray(s_.coords[-1]) - np.asarray(s_.coords[0]))
            lat = sorted(float(np.dot(h - np.asarray(s_.coords[0]), dv)) for h in hits)
            groups = 1 + sum(1 for k in range(1, len(lat)) if lat[k] - lat[k - 1] > 0.5 * w_global)
            cw_stops.append((s_, groups))

    arr_dp, st_dp = _cw_eval(cw_lanes, cw_arrows, cw_stops, w_global)
    arr_old, st_old = _cw_eval(cw_old, cw_arrows, cw_stops, w_global)

    old_span = max((ln["geom"].length for ln in cw_old), default=0.0)
    # 出口道若旧车道仅为短残段（< 65% 车行道轴长），优先保留全长 DP 剖分以补全出口道
    if cw["role"] == "exit" and old_span < 0.65 * axis.length and max(ln["geom"].length for ln in cw_lanes) >= 0.80 * axis.length:
        return cw_lanes, True

    # DP 丢失箭头或丢失停止线精确车道数 → 回退 H_ray
    if arr_dp < arr_old or st_dp < st_old:
        fb = []
        for ln in cw_old:
            g = _clip_safe(ln["geom"], core_poly, B["nondrive"], B["drivable"], cw["role"])
            if g is not None:
                fb.append(_as_fallback(ln, g, cw, "ray_fallback"))
        return (fb if fb else cw_lanes), False

    # 若旧车道已几乎完整无断覆盖本车行道（无 open 断裂、覆盖 ≥80% 轴长），且车道数不少于 DP（防把无标线多车道出口误退回单车道），
    # 但 DP 偏离真实标线中点 > 0.35 m → 回退 H_ray 保真
    old_has_open = any(ln.get("start") == "open" or ln.get("end") == "open" for ln in cw_old)
    if (not old_has_open) and old_span >= 0.78 * axis.length and len(cw_old) >= len(cw_lanes):
        dev = _dp_deviation_from_old(cw_lanes, cw_old, w_global)
        if dev > 0.35:
            fb = []
            for ln in cw_old:
                g = _clip_safe(ln["geom"], core_poly, B["nondrive"], B["drivable"], cw["role"])
                if g is not None:
                    fb.append(_as_fallback(ln, g, cw, "ray_fallback"))
            if fb:
                return fb, False

    return cw_lanes, True


def _cut_at_foreign(g, foreign, core_poly, min_len=5.0):
    """车道穿过邻口停止线：只留靠本路口核心的一段。"""
    for f in foreign:
        if not g.intersects(f):
            continue
        rest = g.difference(f.buffer(0.3))
        parts = [p for p in _parts(rest, "LineString") if p.length >= min_len]
        if not parts:
            return None
        g = min(parts, key=lambda p: p.distance(core_poly))
    return g


_STALE_KEYS = ("dp_add", "dp_drop", "far_cut", "dp_far_start", "dp_far_end", "lane_index", "sec_idx", "_slip_id")


def _as_fallback(ln, geom, cw, source):
    """把 Step 1 射线车道复制成本车行道的兜底车道：清掉上一阶段遗留的拓扑字段，重新挂到 cw。"""
    q = {k: v for k, v in ln.items() if k not in _STALE_KEYS}
    q["geom"] = geom
    q["source"] = source
    q["direction"] = "known"
    q.setdefault("allow", "all")
    if cw is not None:
        q["arm"] = cw["arm"]
        q["carriageway"] = cw["id"]
        q["role"] = cw["role"]
    else:
        q["carriageway"] = None
    return q


def _ax(cw):
    a = cw.get("frenet_axis")
    return a if a is not None else cw["axis"]


def _lane_s_range(axis, g):
    sv = [axis.project(Point(c)) for c in g.coords]
    return min(sv), max(sv)


def _lateral(axis, g, s_ref):
    """车道 g 在基线 axis 弧长 s_ref 处的带符号横向偏移（左正右负）。"""
    q = np.asarray(axis.interpolate(s_ref).coords[0])
    t = _tan(axis, s_ref, 1.5)
    n = np.array([-t[1], t[0]])
    lp = np.asarray(g.interpolate(g.project(Point(q))).coords[0])
    return float(np.dot(lp - q, n))


def build_sections(cw, lanes, idxs, cuts=()):
    """Step 3 → Step 4 契约：按几何重建车行道的分段表（DP 车道与兜底车道同一套规则）。
    每段：section_id、s_start/s_end（沿 frenet_axis 的弧长，行驶方向）、n_lanes、
    lane_ids（全局车道下标，严格右→左 = SUMO lane index 0..K-1）、cut_xy（段起点切点，局部米坐标）、
    transition（与上一段的关系）、links（[上一段序号, 本段序号] 的延续映射）。
    同时给每条车道写 sec_idx {section_id: 序号} 与 lane_index（靠路口一端所在分段的序号）。"""
    axis = _ax(cw)
    L = axis.length
    rng = {gi: _lane_s_range(axis, lanes[gi]["geom"]) for gi in idxs}
    # 切点 = DP 车道数变化点 ∪ 车道几何真实起止点（被岛鼻/停止线/硬门禁裁短的车道在哪儿结束，就在哪儿切一段，
    # 两条车道在不同位置结束时不会被并成“一次少两条”的跳变）；离轴线两端 8 m 内的起止是远端毛边，不切
    cand = sorted([(c, 0) for c in cuts if 0.5 < c < L - 0.5] +
                  [(v, 1) for lo, hi in rng.values() for v in (lo, hi) if 8.0 < v < L - 8.0])
    pts = []
    for v, kind in cand:
        if pts and v - pts[-1][0] < 1.5:
            if kind == 0 and pts[-1][1] == 1:
                pts[-1] = (v, 0)  # 同一处优先用 DP 切点
            continue
        pts.append((v, kind))
    bps = [0.0] + [v for v, _ in pts] + [L]
    raw = []
    for a, b in zip(bps[:-1], bps[1:]):
        pres = []
        for gi, (lo, hi) in rng.items():
            ov = min(b, hi) - max(a, lo)
            if ov >= min(3.0, 0.5 * (b - a)):
                s_ref = 0.5 * (max(a, lo) + min(b, hi))
                pres.append((_lateral(axis, lanes[gi]["geom"], s_ref), gi))
        if pres:
            raw.append([a, b, [gi for _, gi in sorted(pres)]])
    # 相邻两段车道集合与顺序完全相同 → 合并（被删车道留下的多余切点）
    merged = []
    for a, b, ids in raw:
        if merged and merged[-1][2] == ids:
            merged[-1][1] = b
        else:
            merged.append([a, b, ids])
    # 车道 s 范围没覆盖到的车道（极短或横向支路）也要有归属：并入与之重叠最多的段
    seen = {gi for _, _, ids in merged for gi in ids}
    for gi in idxs:
        if gi in seen or not merged:
            continue
        lo, hi = rng[gi]
        j = max(range(len(merged)), key=lambda k: min(merged[k][1], hi) - max(merged[k][0], lo))
        a, b, ids = merged[j]
        s_ref = float(np.clip(0.5 * (lo + hi), a, b))
        cand = [(_lateral(axis, lanes[x]["geom"], s_ref), x) for x in ids + [gi]]
        merged[j][2] = [x for _, x in sorted(cand)]
    secs = []
    for j, (a, b, ids) in enumerate(merged):
        sec = dict(section_id=j, s_start=round(float(a), 1), s_end=round(float(b), 1), n_lanes=len(ids),
                   lane_ids=list(ids), cut_xy=tuple(np.round(np.asarray(axis.interpolate(a).coords[0]), 2)),
                   transition=None, links=[])
        if j > 0:
            prev = merged[j - 1][2]
            cont = [x for x in ids if x in prev]
            added = [x for x in ids if x not in prev]
            dropped = [x for x in prev if x not in ids]
            sec["links"] = [[prev.index(x), ids.index(x)] for x in cont]
            if len(added) == 1 and not dropped:
                p = ids.index(added[0])
                sec["transition"] = "lane_add_right" if p == 0 else ("lane_add_left" if p == len(ids) - 1 else "lane_add_mid")
            elif len(dropped) == 1 and not added:
                p = prev.index(dropped[0])
                sec["transition"] = "lane_drop_right" if p == 0 else ("lane_drop_left" if p == len(prev) - 1 else "lane_drop_mid")
            elif abs(len(ids) - len(prev)) > 1:
                sec["transition"] = f"jump_{len(prev)}_to_{len(ids)}"
            else:
                sec["transition"] = "reconfig"
        secs.append(sec)
    for gi in idxs:
        si = {sec["section_id"]: sec["lane_ids"].index(gi) for sec in secs if gi in sec["lane_ids"]}
        lanes[gi]["sec_idx"] = si
        if si:
            key = max(si) if cw["role"] == "approach" else min(si)
            lanes[gi]["lane_index"] = si[key]
        else:
            lanes[gi]["lane_index"] = None
    # cw["lanes"] 按靠路口一端的序号右→左排序
    cw["lanes"] = sorted(idxs, key=lambda gi: (lanes[gi]["lane_index"] is None, lanes[gi]["lane_index"] or 0))
    cw["sections"] = secs
    return secs


def validate_model_contract(jm, lanes):
    """Step 4 强契约自检：车行道 / 渠化右转 / 附加支路三类容器与车道 carriageway 字段双向一致、无越界、
    无重复归属、无孤儿；分段 n_lanes 与 lane_ids 一致、分段车道并集 = cw.lanes、links 下标在范围内。"""
    errs = []
    n = len(lanes)
    owner = {}
    cw_ids = {c["id"] for c in jm["carriageways"]}

    def own(i, who):
        if not isinstance(i, (int, np.integer)) or not 0 <= i < n:
            errs.append(f"{who}: 车道下标越界 {i}")
            return False
        if i in owner:
            errs.append(f"{who}: 车道 {i} 已归属 {owner[i]}")
            return False
        owner[i] = who
        return True

    for c in jm["carriageways"]:
        who = f"cw{c['id']}"
        for i in c.get("lanes", []):
            if own(i, who) and lanes[i].get("carriageway") != c["id"]:
                errs.append(f"{who}: 车道 {i} 的 carriageway={lanes[i].get('carriageway')}")
        secs = c.get("sections", [])
        if c.get("lanes") and not secs:
            errs.append(f"{who}: 有车道但无分段")
        u = set()
        for k, sec in enumerate(secs):
            ids = sec.get("lane_ids", [])
            if sec.get("n_lanes") != len(ids):
                errs.append(f"{who} 段{k}: n_lanes={sec.get('n_lanes')} 与 lane_ids 数 {len(ids)} 不符")
            bad = [i for i in ids if i not in c.get("lanes", [])]
            if bad:
                errs.append(f"{who} 段{k}: lane_ids {bad} 不在 cw.lanes")
            u |= set(ids)
            if k > 0:
                np_ = secs[k - 1]["n_lanes"]
                for a, b in sec.get("links", []):
                    if not (0 <= a < np_ and 0 <= b < len(ids)):
                        errs.append(f"{who} 段{k}: links 越界 {a}->{b}")
        if secs and u != set(c.get("lanes", [])):
            errs.append(f"{who}: 分段车道并集与 cw.lanes 不一致")
    for s in jm["slips"]:
        for i in s.get("lanes", []):
            if own(i, f"slip{s['id']}") and lanes[i].get("carriageway") is not None:
                errs.append(f"slip{s['id']}: 车道 {i} 带 carriageway={lanes[i].get('carriageway')}")
    for i in jm.get("extra_lanes", []):
        if own(i, "extra") and lanes[i].get("carriageway") is not None:
            errs.append(f"extra: 车道 {i} 带 carriageway={lanes[i].get('carriageway')}")
    for i, ln in enumerate(lanes):
        if i not in owner:
            errs.append(f"孤儿车道 {i}")
        c = ln.get("carriageway")
        if c is not None and c not in cw_ids:
            errs.append(f"车道 {i}: 悬挂 carriageway={c}")
    return errs


def partition_junction(jm, old_lanes, B, core_stops, arrows, bikes, parking, P):
    """Step 3 主入口：对 JunctionModel 的每条物理车行道与绕岛右转走廊执行全局车道剖分，
    采用单车行道多假设几何自检（H_DP vs H_ray），保留多分支支路，并做去重与防交叉清理；
    最后按几何统一重建分段表与三类容器（carriageways / slips / extra_lanes），并做契约自检。"""
    core_poly = jm["core_poly"]
    w_global = P["w_mode"]
    new_lanes = []

    for cw in jm["carriageways"]:
        cw_old = _cw_matching_old_lanes(cw, old_lanes)
        cw_lanes, cuts = partition_carriageway(cw, cw_old, B, core_poly, core_stops, arrows, bikes, parking, w_global)
        chosen, used_dp = _select_cw_lanes(cw, cw_lanes, cw_old, core_stops, arrows, B, core_poly, w_global)
        cw["_cuts"] = cuts if used_dp else []
        cw["used_frenet_dp"] = bool(used_dp and chosen)
        cw["marked_input"] = not cw.get("unmarked", False)
        new_lanes.extend(chosen or [])

    # 绕岛右转车道（与 slip 严格成对，失败的 slip 不再错位占用后面 slip 的车道）
    for s_obj in jm["slips"]:
        s_obj["lanes"], s_obj["lane"] = [], None
    for s_obj, ln in _slip_lanes(jm["slips"], B, core_poly, w_global):
        ln["_slip_id"] = s_obj["id"]
        new_lanes.append(ln)

    # 多分支/分叉/匝道兜底：有效（接停止线/含箭头/有真实标线支撑且长度足够）但未被新车道覆盖的旧车道
    if new_lanes and old_lanes:
        cov_u = unary_union([l["geom"].buffer(0.55 * w_global) for l in new_lanes])
        cov_tight = unary_union([l["geom"].buffer(0.35 * w_global) for l in new_lanes])
        for ln in old_lanes:
            g = ln["geom"]
            if g.length < 6.0:
                continue
            ov_frac = g.intersection(cov_u).length / max(g.length, 1e-6)
            if ov_frac > 0.35:
                continue
            has_stop = any(Point(g.coords[-1]).distance(s) < 0.5 * w_global + 1.2 for s in core_stops)
            has_arrow = len(ln.get("arrows", [])) > 0
            is_marked_main = ((not ln.get("edgepair", False)) and (ln.get("n_norm", 0) >= 8)
                              and (g.length >= 14.0) and (ln.get("role") in ("approach", "exit")))
            is_slip = ln.get("role") == "slip" or ln.get("slip")
            is_uncovered_slip = is_slip and ov_frac < 0.15 and g.length >= 12.0
            if not (has_stop or has_arrow or is_marked_main or is_uncovered_slip):
                continue
            g2 = g
            if g2.intersection(cov_tight).length > 5.0:
                rest = g2.difference(cov_tight)
                parts = [p for p in _parts(rest, "LineString") if p.length >= 6.0]
                if not parts:
                    continue
                g2 = max(parts, key=lambda p: p.length)
            g2 = _cut_at_foreign(g2, jm.get("foreign_stops", []), core_poly)
            if g2 is None:
                continue
            if ln.get("role") == "approach":
                g2 = _snap_approach_end(g2, core_stops, core_poly, B["nondrive"], w_global)
            g2 = _clip_safe(g2, core_poly, B["nondrive"], B["drivable"], ln.get("role", "approach"))
            if g2 is None or g2.length < 5.0:
                continue
            # 注册：同向且 ≥80% 落在某条车行道内 → 挂到该车行道；落在某条 slip 走廊内 → 挂到该 slip；否则附加支路
            host = None
            if ln.get("role") in ("approach", "exit") and not is_slip:
                tg = _tan(g2, 0.5 * g2.length, 2.0)
                for cw in jm["carriageways"]:
                    if cw["role"] != ln["role"]:
                        continue
                    if g2.intersection(cw["polygon"].buffer(1.0)).length < 0.8 * g2.length:
                        continue
                    ax = _ax(cw)
                    if float(np.dot(tg, _tan(ax, ax.project(g2.interpolate(0.5, normalized=True)), 2.0))) > 0.7:
                        host = cw
                        break
            q = _as_fallback(ln, g2, host, "fallback_branch")
            if host is None:
                q["role"] = ln.get("role")
                if is_slip:
                    for s_obj in jm["slips"]:
                        cp = s_obj.get("corridor_poly")
                        if cp is not None and g2.intersection(cp.buffer(1.5)).length >= 0.6 * g2.length:
                            q["_slip_id"] = s_obj["id"]
                            break
            new_lanes.append(q)
            cov_u = unary_union([cov_u, g2.buffer(0.55 * w_global)])
            cov_tight = unary_union([cov_tight, g2.buffer(0.35 * w_global)])

    # 清除极少数跨车行道拼接/兜底分支处产生的重叠与中段交叉
    kept_lanes = new_lanes
    if len(new_lanes) > 1:
        keep = [True] * len(new_lanes)
        for i in range(len(new_lanes)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(new_lanes)):
                if not keep[j]:
                    continue
                ga, gb = new_lanes[i]["geom"], new_lanes[j]["geom"]
                if ga.distance(gb) > 0.45 * w_global:
                    continue
                ov = ga.intersection(gb.buffer(0.4 * w_global)).length
                if ov > min(8.0, 0.5 * min(ga.length, gb.length)):
                    sa = (1 if new_lanes[i].get("source") == "frenet_dp" else 0, ga.length)
                    sb = (1 if new_lanes[j].get("source") == "frenet_dp" else 0, gb.length)
                    if sa >= sb:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        kept_lanes = [l for l, k in zip(new_lanes, keep) if k]
        for _ in range(3):
            any_fix = False
            drop_set = set()
            for i in range(len(kept_lanes)):
                if i in drop_set:
                    continue
                for j in range(i + 1, len(kept_lanes)):
                    if j in drop_set:
                        continue
                    a, b = kept_lanes[i]["geom"], kept_lanes[j]["geom"]
                    if a.distance(b) > 0.1:
                        continue
                    x = a.intersection(b)
                    pts = [q for q in getattr(x, "geoms", [x]) if q.geom_type == "Point"]
                    ends = [Point(a.coords[0]), Point(a.coords[-1]), Point(b.coords[0]), Point(b.coords[-1])]
                    bad_pts = [q for q in pts if min(q.distance(e) for e in ends) > 3.0]
                    if bad_pts:
                        any_fix = True
                        pa = (1 if kept_lanes[i].get("source") == "frenet_dp" else 0, a.length)
                        pb = (1 if kept_lanes[j].get("source") == "frenet_dp" else 0, b.length)
                        wi = j if pa >= pb else i
                        other = a if wi == j else b
                        rest = kept_lanes[wi]["geom"].difference(other.buffer(0.45 * w_global))
                        parts = [p for p in _parts(rest, "LineString") if p.length >= 4.0]
                        if parts:
                            kept_lanes[wi]["geom"] = max(parts, key=lambda p: p.length)
                        else:
                            drop_set.add(wi)
            if drop_set:
                kept_lanes = [l for idx, l in enumerate(kept_lanes) if idx not in drop_set]
            if not any_fix:
                break

    # 统一重建容器与分段表（删道/裁道之后做，sections 与 cw.lanes 天然同步）
    for i, ln in enumerate(kept_lanes):
        if ln.get("role") == "slip" or ln.get("_slip_id") is not None:
            ln["carriageway"] = None
    for cw in jm["carriageways"]:
        idxs = [i for i, ln in enumerate(kept_lanes) if ln.get("carriageway") == cw["id"]]
        cw["sections"] = []
        cw["lanes"] = idxs
        if idxs:
            build_sections(cw, kept_lanes, idxs, cw.get("_cuts", []))
        cw["n_lanes"] = len(cw["lanes"])
        cw["unmarked"] = not cw["lanes"]
        cw["used_frenet_dp"] = bool(cw["lanes"]) and all(kept_lanes[i].get("source") == "frenet_dp" for i in cw["lanes"])
    for s_obj in jm["slips"]:
        ids = [i for i, ln in enumerate(kept_lanes) if ln.get("_slip_id") == s_obj["id"]]
        ax = s_obj.get("axis")
        if len(ids) > 1 and ax is not None:
            ids.sort(key=lambda i: _lateral(ax, kept_lanes[i]["geom"], 0.5 * ax.length))
        for k, i in enumerate(ids):
            kept_lanes[i]["lane_index"] = k
            kept_lanes[i]["slip_from_arm"] = s_obj.get("from_arm")
            kept_lanes[i]["slip_to_arm"] = s_obj.get("to_arm")
        s_obj["lanes"] = ids
        s_obj["lane"] = max(ids, key=lambda i: kept_lanes[i]["geom"].length) if ids else None
    owned = {i for c in jm["carriageways"] for i in c["lanes"]} | {i for s in jm["slips"] for i in s["lanes"]}
    jm["extra_lanes"] = [i for i in range(len(kept_lanes)) if i not in owned]
    for i in jm["extra_lanes"]:
        kept_lanes[i]["carriageway"] = None
        kept_lanes[i].setdefault("lane_index", None)
    for ln in kept_lanes:
        ln.setdefault("allow", "all")
        ln.setdefault("source", "ray")
    jm["contract_errors"] = validate_model_contract(jm, kept_lanes)
    return kept_lanes
