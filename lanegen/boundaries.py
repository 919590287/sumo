"""边界构建：纵向标线缝合、虚线补段、可行驶包络边缘、物理隔离。

每条边界是 (LineString, 类型)。类型：
  white  白实线/组合线（含路缘线）
  dash   白虚线（已补段成连续链）
  yellow 黄线（对向分隔）
  bus    公交专用道线
  barrier 物理隔离
  edge   可行驶包络边缘（路缘、安全岛、导流区）
"""
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from scipy.spatial import cKDTree

WHITE = {502157, 504881, 502156, 502154, 502155}
DASH = {503969}
# 路口内部导向线 / 待转区线：全市 97% 位于停止线 30 m 内、98% 靠近人行横道（路段虚线 503969 为 0%）。
# 不作车道边界、不算路段标线；单独保存，供生成路口内部转弯轨迹使用。
GUIDE = {504948, 505039}
YELLOW = {503970, 504874, 503313, 504878}
BUS = {504634, 504658, 504866, 504867}
REV = {504636, 504660, 504635, 504659}
STOP = {501775}
MARKING_TYPES = {"white", "dash", "yellow", "bus", "sep"}
SEGMENT_MARKINGS = {"dash", "yellow", "bus"}  # 只出现在路段上、不出现在路口内部的纵向标线


def explode_lines(geoms):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            out.append(g)
        elif g.geom_type in ("MultiLineString", "GeometryCollection"):
            out.extend([p for p in g.geoms if p.geom_type == "LineString"])
    return [l for l in out if l.length > 0.05]


def stitch(lines, grid=0.03):
    """瓦片接缝缝合：坐标对齐到网格后 linemerge。"""
    if not lines:
        return []
    snapped = [shapely.set_precision(l, grid) for l in lines]
    u = unary_union([s for s in snapped if not s.is_empty])
    if u.is_empty:
        return []
    if u.geom_type == "LineString":
        return explode_lines([u])
    return explode_lines([linemerge(u)])


def _end_dirs(line):
    c = np.asarray(line.coords)
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    # 端点向外的切向
    p0, p1 = c[0], c[-1]
    k0 = 1
    while k0 < len(c) - 1 and np.linalg.norm(c[k0] - p0) < 0.5:
        k0 += 1
    k1 = len(c) - 2
    while k1 > 0 and np.linalg.norm(c[k1] - p1) < 0.5:
        k1 -= 1
    return p0, unit(p0 - c[k0]), p1, unit(p1 - c[k1])


def _arc_ok(ta, tb, u, g, max_half=15.0, max_asym=3.0, min_r=20.0):
    """弯道虚线补段：a 端行驶切向 ta、b 端行驶切向 tb、弦方向 u。圆弧上弦与两端切向的夹角相等且同向
    （ta→u 转 θ/2，u→tb 再转 θ/2）；横向错开的两条虚线是反向 S 形（两个转角异号），不接。
    半径不小于 min_r（城市道路弯道下限量级）。"""
    def ang(p, q):
        return np.degrees(np.arctan2(p[0] * q[1] - p[1] * q[0], np.dot(p, q)))
    a1, a2 = ang(ta, u), ang(u, tb)
    if abs(a1) > max_half or abs(a2) > max_half or abs(a1 - a2) > max_asym:
        return False
    half = np.radians(0.5 * abs(a1 + a2))
    return half < 1e-3 or g / (2 * np.sin(half)) >= min_r


def bridge_dashes(dashes, blockers, max_gap=16.0, max_ang=12.0, max_lat=0.45):
    """虚线补段：每段虚线端点与下一段的起点按共线性配对后连接。"""
    if not dashes:
        return []
    ends = []  # (point, outward_dir, dash_idx, which_end)
    for i, d in enumerate(dashes):
        p0, t0, p1, t1 = _end_dirs(d)
        ends.append((p0, t0, i, 0))
        ends.append((p1, t1, i, 1))
    P = np.array([e[0] for e in ends])
    T = np.array([e[1] for e in ends])
    tree = cKDTree(P)
    cand = []
    cos_a = np.cos(np.radians(max_ang))
    for a, b in tree.query_pairs(max_gap):
        if ends[a][2] == ends[b][2]:
            continue
        v = P[b] - P[a]
        g = np.linalg.norm(v)
        if g < 1e-6:
            cand.append((0.0, a, b))
            continue
        u = v / g
        # a 的外向切向指向 b，b 的外向切向指向 a，两段方向一致
        straight = (np.dot(T[a], u) >= cos_a and np.dot(T[b], -u) >= cos_a and np.dot(T[a], -T[b]) >= cos_a
                    and abs(T[a][0] * v[1] - T[a][1] * v[0]) <= max_lat + 0.02 * g)
        if not straight and not _arc_ok(T[a], -T[b], u, g):
            continue
        cand.append((g, a, b))
    cand.sort()
    used = set()
    bridges = []
    for g, a, b in cand:
        if a in used or b in used:
            continue
        seg = LineString([P[a], P[b]])
        if blockers is not None and g > 0.5 and seg.intersects(blockers):
            continue
        used.add(a)
        used.add(b)
        if g > 0.01:
            bridges.append(seg)
    return stitch(list(dashes) + bridges, grid=0.02)


def classify_barriers(barriers, drivable, off=1.5, step=2.0):
    """隔离设施分两类（由位置算，不用固定语义）：
    两侧 off 米处都是可行驶区 → 车行道之间的分隔（sep，等同标线）；否则是路侧护栏（barrier，等同路缘）。"""
    out = []
    for l in barriers:
        if drivable is None or l.length < 1.0:
            out.append((l, "barrier"))
            continue
        d = np.arange(step / 2, l.length, step)
        if len(d) == 0:
            d = np.array([l.length / 2])
        pts = shapely.get_coordinates(shapely.line_interpolate_point(l, d))
        a = shapely.get_coordinates(shapely.line_interpolate_point(l, np.clip(d - 0.5, 0, l.length)))
        b = shapely.get_coordinates(shapely.line_interpolate_point(l, np.clip(d + 0.5, 0, l.length)))
        t = b - a
        n = np.linalg.norm(t, axis=1, keepdims=True)
        t = t / np.where(n < 1e-9, 1, n)
        nv = np.stack([-t[:, 1], t[:, 0]], 1)
        left = shapely.contains_xy(drivable, *(pts + nv * off).T)
        right = shapely.contains_xy(drivable, *(pts - nv * off).T)
        out.append((l, "sep" if np.mean(left & right) >= 0.6 else "barrier"))
    return out


HATCH_CLOSE = 1.5  # 导流区斜纹条间距（实测条宽约 0.3 m、间距 1–2.5 m）的一半多一点


def nondrive_zone(main):
    """不可行驶区：安全岛 + 导流区。
    导流区（channelization_area）在数据里是一根根 0.3 m 宽的斜纹条，不是整块面；直接挖空会在路面上
    留下锯齿状小洞，每个齿边都变成“路缘”。先按斜纹间距闭运算连成整块，再挖。"""
    code = main["分类代码"]
    isl = [g for g, c in zip(main.geometry, code) if c == "traffic_island"]
    ch = [g for g, c in zip(main.geometry, code) if c == "channelization_area"]
    parts = []
    if isl:
        parts.append(unary_union(isl))
    if ch:
        z = unary_union([g.buffer(HATCH_CLOSE) for g in ch]).buffer(-HATCH_CLOSE)
        # 闭运算后仍是零散小块（孤立单根细条 < 1.5 m²）的不算导流区
        parts += [p for p in getattr(z, "geoms", [z]) if p.geom_type == "Polygon" and p.area >= 1.5]
    if not parts:
        return None
    u = unary_union(parts)
    return u if not u.is_empty else None


BUS_STRIP_MAX = 4.8  # 公交专用道（单车道）宽度上限，米


def _is_bus_edge_yellow(line, yellows, bus_marks, step=2.0):
    """黄线是不是公交专用道边线（而非对向分隔中心线）。
    判据：线的任一侧一个车道宽（0.4–4.8 m）内有公交专用标记，且它不是双黄线中的一条
    （0.6 m 内另有平行黄线 → 中心线，保留）。
    百度把公交道黄线和中心线画成同名样式（504878 只有 56% 靠近公交标记，503970 也有 14%），不能按样式号区分；
    公交道也不一定贴路缘（开创大道西出口外侧还有 5 m 路面）。
    当成中心线的后果：被“黄线在左”判反方向、把同向车行道劈成两幅。"""
    if not bus_marks or line.length < 6.0:
        return False
    ss = np.arange(1.0, line.length - 1.0, step)
    if len(ss) < 3:
        return False
    lb = line.buffer(0.3)
    if sum(lb.intersection(bm).length for bm in bus_marks) >= 0.5 * line.length:
        return "center"  # 与公交字样锚线重合：是车道中线上的字样，不是边界
    rays = {1: [], -1: []}
    others = [y for y in yellows if y is not line and y.distance(line) < 0.8]
    dbl = 0
    for s in ss:
        p = np.asarray(line.interpolate(s).coords[0])
        a = np.asarray(line.interpolate(max(0.0, s - 1.0)).coords[0])
        b = np.asarray(line.interpolate(min(line.length, s + 1.0)).coords[0])
        t = b - a
        nt = np.linalg.norm(t)
        if nt < 1e-9:
            continue
        n = np.array([-t[1], t[0]]) / nt
        for sg in (1, -1):
            rays[sg].append(LineString([p + sg * n * 0.4, p + sg * n * BUS_STRIP_MAX]))
        if any(0.1 < Point(p).distance(o) < 0.6 for o in others):  # 重合的重复线不算双黄
            dbl += 1
    if dbl >= 0.3 * len(ss):
        return False
    side = {}
    for sg in (1, -1):
        band = unary_union(rays[sg]).buffer(0.3)
        side[sg] = sum(band.intersection(bm).length for bm in bus_marks)
    return max(side.values()) >= 1.0


def build(roi, P=None):
    """P 为数据推导参数（params.derive），其中 dash_max_gap 需在本函数内先算。"""
    from . import params as params_mod
    lines = roi["lines"]
    main = roi["main"]
    sid = lines["源样式ID"].astype(int)

    def sel(ids):
        return explode_lines(lines[sid.isin(ids)].geometry)

    stops = stitch(sel(STOP))
    crosswalks = [g for g, c in zip(main.geometry, main["分类代码"]) if c == "crosswalk_area"]
    blockers = unary_union(stops + [c.boundary for c in crosswalks]) if (stops or crosswalks) else None

    # 可行驶包络
    code = main["分类代码"]
    road = [g for g, c, cat in zip(main.geometry, code, main["原始类别"]) if c == "road_surface" and cat == 15]
    close_r = 0.3  # physical：<0.6 m 的块间缝隙不构成通行分隔
    road_u = unary_union([g.buffer(close_r) for g in road]).buffer(-close_r) if road else None
    non_u = nondrive_zone(main)
    drivable = road_u.difference(non_u) if (road_u is not None and non_u is not None) else road_u
    if drivable is not None and not drivable.is_empty:
        drivable = drivable.buffer(0)

    out = []
    for l in stitch(sel(WHITE | REV)):
        out.append((l, "white"))
    bus_marks = stitch(sel(BUS - YELLOW))
    yl = stitch(sel(YELLOW))
    kind = [_is_bus_edge_yellow(l, yl, bus_marks) for l in yl]
    for j, l in enumerate(yl):
        if not kind[j] and l.length < 6.0:  # 瓦片切碎的短段随相接的公交道黄线
            k2 = [kind[i] for i, q in enumerate(yl) if kind[i] and l.distance(q) < 0.5]
            if k2:
                kind[j] = "center" if "center" in k2 else True
    for l, k in zip(yl, kind):
        if k != "center":
            out.append((l, "bus" if k else "yellow"))
    # 公交专用车道标记（504634/504658 等）是“公交车道”字样/图案的锚线，画在车道中线上，不是车道边界：
    # 当边界会把公交道从中间劈开（开创大道西出口）。单独保存，供车道打 allow=bus
    dashes = sel(DASH)
    g = params_mod.dash_gap(dashes)
    if g is not None and len(g) >= 8:
        gap_p75 = float(np.percentile(g, 75))
        max_gap = float(np.clip(params_mod.F["dash_k"] * gap_p75, 3.0, 20.0))
        gap_src = "data"
    else:
        gap_p75, max_gap, gap_src = None, 16.0, "default"
    for l in bridge_dashes(dashes, blockers, max_gap=max_gap):
        out.append((l, "dash"))
    out += classify_barriers(explode_lines(roi["barrier"].geometry), drivable)
    if drivable is not None and not drivable.is_empty:
        edge = drivable.boundary.simplify(0.05)
        for l in explode_lines([edge]):
            out.append((l, "edge"))
    return {
        "bounds": out,
        "stops": stops,
        "crosswalks": crosswalks,
        "drivable": drivable,
        "nondrive": non_u,
        "dashes_raw": dashes,
        "guides": stitch(sel(GUIDE)),
        "bus_marks": bus_marks,
        # 潮汐/可变车道线：几何上仍按白线参与分界（上面并进了 white），这里单独留一份供车道打 reversible 标记
        "rev": stitch(sel(REV)),
        "dash_max_gap": (max_gap, gap_src),
        "dash_gap_p75": gap_p75,
    }
