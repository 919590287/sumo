"""Step 2：物理车行道先行（Carriageway-First）的路口结构模型 JunctionModel。

把路口从可行驶面 drivable 物理剖分为：
  core_poly      紧致非凸核心面（含错位臂保护，绝不跨过安全岛吞并绕岛右转道）
  obstacles      障碍分类：corner_island（紧贴核心、被路面环绕的转角岛/导流区/空洞）、
                 island（其他岛/中分带）、block（外围建筑街区）
  arms[]         按方位角排序的路臂（同向分幅口门合并，过滤无标线窄路侧支路）
  slips[]        基于 corner_island 外环扫描检出的绕岛渠化右转走廊
                 (from_arm, to_arm, corridor_poly, axis, diverge_pt, merge_pt, obstacle, lanes)
  carriageways[] 从 drivable - core_poly - slips 物理切分出的单向车行道多边形
                 （硬分隔线/中分带鼻端自动封口至核心，无黄线双向路按车道中线二分；
                   即使无标线/无车道也保留物理走廊与抗支路抖动的平滑 Frenet 基线 axis）
"""
import math

import numpy as np
import shapely
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import linemerge, unary_union


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _tan(g, s, h=1.0):
    a = np.asarray(g.interpolate(max(0.0, s - h)).coords[0])
    b = np.asarray(g.interpolate(min(g.length, s + h)).coords[0])
    return _unit(b - a)


def _parts(g, kind="Polygon"):
    if g is None or g.is_empty:
        return []
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == kind]


def _smooth_pts(pts, win=7):
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 5:
        return pts
    w = min(win, n if n % 2 else n - 1)
    if w < 5:
        return pts
    x = savgol_filter(pts[:, 0], w, 2, mode="interp")
    y = savgol_filter(pts[:, 1], w, 2, mode="interp")
    x[0], y[0] = pts[0, 0], pts[0, 1]
    x[-1], y[-1] = pts[-1, 0], pts[-1, 1]
    return np.stack([x, y], 1)


def _ang_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0))))


def _signed_turn_deg(line):
    c = np.asarray(line.coords)
    if len(c) < 3 or line.length < 4.0:
        return 0.0
    v0 = _tan(line, min(2.0, 0.2 * line.length))
    v1 = _tan(line, max(line.length - 2.0, 0.8 * line.length))
    return float(np.degrees(np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], np.dot(v0, v1))))


# ---------------------------------------------------------------- 1. 紧致核心面

def _strip(stop, d, depth=60.0, sign=-1.0):
    """沿停止线法向 d * sign 拉出的条带（sign=-1 上游，sign=+1 下游）。"""
    a, b = np.asarray(stop.coords[0]), np.asarray(stop.coords[-1])
    t = _unit(b - a)
    a, b = a - t * 0.3, b + t * 0.3
    return Polygon([a, b, b + d * (sign * depth), a + d * (sign * depth)])


def build_core_poly(core_hull, quads, drivable, core_stops, w, stop_dirs=None, crosswalks=None,
                    nondrive=None, seg_marks=None):
    """紧致非凸核心面：
    1. 停止线 + 人行横道凸包 ∪ T 形主路四边形，扣掉所有停止线的上游条带（保证核心绝不侵占正常进口道）；
    2. 与 drivable 求交，开运算去细颈，保留与主路口相连的主体（排除被安全岛隔开的微小右转停止线块）；
    3. 仅在不含安全岛的大幅中分带处桥接主路半幅；
    4. 若存在因错位 T 口被垂直停止线上游条带切脱的臂（如荔湾 7913 西臂，停止线距核心 > 2 m），
       单独把该停止线及其人行横道沿下游方向接回主核心，不破坏其他停止线的口门边界。"""
    base0 = unary_union([core_hull] + [q for q, _ in quads]) if quads else core_hull
    base = base0
    if stop_dirs and core_stops:
        ups = [_strip(s, d, depth=60.0, sign=-1.0) for s, d in zip(core_stops, stop_dirs) if d is not None]
        if ups:
            base = base.difference(unary_union(ups))
            if base.geom_type == "MultiPolygon":
                keep_b = [p for p in base.geoms if p.area > 5.0]
                base = unary_union(keep_b) if keep_b else base

    if drivable is None or drivable.is_empty:
        return base if base.geom_type == "Polygon" else max(_parts(base), key=lambda p: p.area)
    dv = drivable.buffer(0.5).buffer(-0.5)
    x = base.intersection(dv)
    if seg_marks is not None and not seg_marks.is_empty:
        x = x.difference(seg_marks.buffer(1.8))
    r = 0.4 * w
    opened = x.buffer(-r).buffer(r)
    parts = _parts(opened)
    if not parts:
        return base if base.geom_type == "Polygon" else max(_parts(base), key=lambda p: p.area)

    anchors = [s.buffer(1.2) for s in core_stops] or [base.centroid.buffer(5.0)]
    keep = [p for p in parts if any(p.intersects(a) for a in anchors)]
    if not keep:
        keep = [max(parts, key=lambda p: p.area)]
    main_p = max(keep, key=lambda p: p.area)
    major_stops = [s.buffer(1.2) for s in core_stops if s.length >= 6.5]
    if len(keep) > 1 and nondrive is not None and not nondrive.is_empty:
        nd_clean = nondrive.buffer(-0.3)
        filtered = [main_p]
        for p in keep:
            if p is main_p:
                continue
            is_major = p.area >= 120.0 or any(p.intersects(ms) for ms in major_stops)
            if not is_major:
                a_pt, b_pt = shapely.ops.nearest_points(p, main_p)
                if a_pt.distance(b_pt) > 0.5:
                    seg = LineString([a_pt, b_pt])
                    if not nd_clean.is_empty and seg.intersection(nd_clean).length > 1.0:
                        continue
            filtered.append(p)
        keep = filtered

    poly = unary_union(keep).buffer(0.3).buffer(-0.3).buffer(0)
    if poly.geom_type == "MultiPolygon":
        bridged = poly.buffer(3.0).buffer(-3.0).intersection(base)
        poly = unary_union([poly, bridged]).buffer(0)
    if poly.geom_type == "MultiPolygon":
        major_parts = [p for p in poly.geoms if p.area >= 120.0 or any(p.intersects(ms) for ms in major_stops)]
        if len(major_parts) >= 2:
            bridged = unary_union(major_parts).buffer(6.5).buffer(-6.5).intersection(base.buffer(1.0)).buffer(0)
            poly = unary_union(major_parts + [bridged]).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def grow_free(core_poly, lanes, drivable, w, reach=25.0):
    """停止线 ≤ 2 条时，把与核心相连、没有车道经过的空路面并入核心。"""
    if drivable is None or not lanes:
        return core_poly
    lane_u = unary_union([ln["geom"].buffer(0.5 * ln["width"] + 0.5, cap_style="flat") for ln in lanes])
    free = drivable.intersection(core_poly.buffer(reach)).difference(lane_u)
    free = free.buffer(-1.0).buffer(1.0)
    comp = [p for p in _parts(free) if p.distance(core_poly) < 1.0]
    if not comp:
        return core_poly
    g = unary_union([core_poly] + comp).buffer(0.5).buffer(-0.5).buffer(0)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return g if g.geom_type == "Polygon" else core_poly


# ---------------------------------------------------------------- 2. 障碍分类（转角岛 vs 中分带 vs 街区）

def obstacles(drivable, nondrive, core_poly, max_hole_area=4000.0, near=45.0):
    """区分转角渠化岛（corner_island）、其他岛/中分带（island）与建筑街区（block）。"""
    raw = []
    zone = core_poly.buffer(near)
    dv_clean = drivable.buffer(0.3).buffer(-0.3) if (drivable is not None and not drivable.is_empty) else None
    if dv_clean is not None:
        for p in _parts(dv_clean):
            for ring in p.interiors:
                h = Polygon(ring)
                if 5.0 <= h.area <= max_hole_area and h.intersects(zone):
                    raw.append((h, "hole"))
    for g in _parts(nondrive):
        if g.area >= 5.0 and g.intersects(zone):
            raw.append((g, "island"))
    if not raw:
        return []
    merged_u = unary_union([g.buffer(0.25) for g, _ in raw]).buffer(-0.25)
    dv_buf = dv_clean.buffer(1.5) if dv_clean is not None else None
    out = []
    for p in _parts(merged_u):
        if p.area < 5.0:
            continue
        dist_core = float(p.distance(core_poly))
        ring = p.exterior
        L = ring.length
        if dv_buf is not None and L > 1.0:
            s = np.arange(0.0, L, 2.0)
            pts = shapely.get_coordinates(shapely.line_interpolate_point(ring, s))
            wrap = float(np.mean(shapely.contains_xy(dv_buf, pts[:, 0], pts[:, 1])))
        else:
            wrap = 0.0
        c_pts = np.asarray(ring.coords)[:, :2]
        max_ext = float(max(Point(q).distance(core_poly) for q in c_pts[:: max(1, len(c_pts) // 24)]))
        has_island_src = any(k == "island" and p.intersects(g) for g, k in raw)
        # 真转角岛：紧贴核心（≤12 m）、面积 ≤ 2600 m²、四面几乎全被路面环绕、最远端不超出 75 m
        if dist_core <= 12.0 and 5.0 <= p.area <= 2600.0 and wrap >= 0.70 and max_ext <= 75.0:
            kind = "corner_island"
        elif has_island_src or (p.area <= 1500.0 and max_ext <= 70.0):
            kind = "island"
        else:
            kind = "block"
        out.append(dict(geom=p, kind=kind, area=round(float(p.area), 1), dist_core=round(dist_core, 1), wrap=round(wrap, 2)))
    return out


# ---------------------------------------------------------------- 3. 路臂发现、同向合并与假空臂过滤

def find_mouths(core_poly, drivable, min_len=2.5):
    """核心边界上与外侧可行驶面相接的段 = 口门。返回 [(LineString, 外向单位向量)]。"""
    if drivable is None:
        return []
    outside = drivable.difference(core_poly.buffer(1.0)).buffer(-0.5).buffer(0.5)
    far = core_poly.buffer(10.0)
    ring = core_poly.exterior
    lines = []
    # 吸附容差会让相邻口门在路口转角处重叠。沿每块已分离的外侧路面
    # 分别求交，仅在同一块内缝合，避免把多个臂拼成一条闭环口门。
    for part in _parts(outside):
        if part.difference(far).is_empty:
            continue
        x = ring.intersection(part.buffer(1.6))
        segments = _parts(x, "LineString") + [s for g in _parts(x, "MultiLineString") for s in g.geoms]
        if len(segments) == 1:
            lines.extend(segments)
        elif segments:
            lines.extend(_parts(linemerge(segments), "LineString"))
    out = []
    c = np.asarray(core_poly.centroid.coords[0])
    for l in lines:
        if l.length < min_len:
            continue
        m = np.asarray(l.interpolate(0.5, normalized=True).coords[0])
        a, b = np.asarray(l.coords[0]), np.asarray(l.coords[-1])
        if l.is_ring:
            continue  # 包围整个核心的一块路面不能提供有方向的口门
        n = _unit(np.array([-(b - a)[1], (b - a)[0]]))
        p1, p2 = m + n * 2.0, m - n * 2.0
        if core_poly.contains(Point(p1)) and not core_poly.contains(Point(p2)):
            n = -n
        elif not core_poly.contains(Point(p1)) and not core_poly.contains(Point(p2)):
            n = n if np.dot(n, m - c) > 0 else -n
        out.append((l, n))
    return out


def group_arms(mouths, max_ang=35.0, max_gap=20.0):
    """口门按外向方位角相近、空间相邻聚成初始路臂。"""
    groups = []
    for l, n in mouths:
        for g in groups:
            if _ang_deg(g["n"], n) < max_ang and min(l.distance(o) for o, _ in g["mouths"]) < max_gap:
                g["mouths"].append((l, n))
                g["n"] = _unit(sum(o.length * nn for o, nn in g["mouths"]))
                break
        else:
            groups.append(dict(mouths=[(l, n)], n=n))
    arms = []
    for g in groups:
        L = sum(o.length for o, _ in g["mouths"])
        if L < 2.5:
            continue
        mid = np.mean([np.asarray(o.interpolate(0.5, normalized=True).coords[0]) for o, _ in g["mouths"]], axis=0)
        bearing = (math.degrees(math.atan2(g["n"][0], g["n"][1])) + 360) % 360
        arms.append(dict(mouths=[o for o, _ in g["mouths"]], n=g["n"], mid=mid, bearing=bearing, width=L))
    arms.sort(key=lambda a: a["bearing"])
    for i, a in enumerate(arms):
        a["id"] = i
    return arms


def lane_arm(ln, arms, core_poly, reach=30.0):
    """车道靠近核心的一端离哪个路臂口门最近（且切向大致沿该臂方向）。"""
    g = ln["geom"]
    best = None
    for which in (0, 1):
        e = Point(g.coords[0] if which == 0 else g.coords[-1])
        t_out = _tan(g, min(5.0, 0.5 * g.length)) if which == 0 else -_tan(g, max(0.0, g.length - min(5.0, 0.5 * g.length)))
        for a in arms:
            d = min(m.distance(e) for m in a["mouths"])
            if d >= reach:
                continue
            cos_a = float(np.dot(t_out, a["n"]))
            if d > 8.0 and cos_a < 0.25:
                continue
            score = d + max(0.0, 1.0 - max(0.0, cos_a)) * 8.0
            if best is None or score < best[0]:
                best = (score, a["id"], which)
    return (best[1], best[2]) if best else (None, None)


def _road_dir(arm, lanes, s_out=25.0):
    """路臂的外向道路方向：各车道离口门 s_out 米处的外向切向按长度加权平均；无车道时用口门法向。"""
    v = np.zeros(2)
    for ln in lanes:
        if ln.get("arm") != arm["id"] or ln.get("role") not in ("approach", "exit"):
            continue
        g = ln["geom"]
        s = min(s_out, g.length)
        t = _tan(g, s) if ln["role"] == "exit" else -_tan(g, g.length - s)
        v += t * min(g.length, 50.0)
    return _unit(v) if np.linalg.norm(v) > 1e-6 else arm["n"]


def _has_road_evidence(arm, B, refs, drivable, probe_len=40.0):
    """无车道口门是否属于真实道路臂（排除窄无标线小巷/小区出入口）。"""
    u = arm["n"]
    lat = np.array([-u[1], u[0]])
    half_w = max(0.5 * arm["width"] + 2.0, 5.0)
    m = arm["mid"]
    probe = Polygon([
        m - lat * half_w,
        m + lat * half_w,
        m + u * probe_len + lat * half_w,
        m + u * probe_len - lat * half_w,
    ])
    # 证据 1：≥12 m 与口门方向平行的纵向标线（最可靠：无标线小巷一定没有）
    mark_len = 0.0
    for l, t in B["bounds"]:
        if t not in ("white", "dash", "yellow", "bus") or not l.intersects(probe):
            continue
        x = l.intersection(probe)
        for seg in _parts(x, "LineString"):
            if seg.length < 2.0:
                continue
            tv = _unit(np.asarray(seg.coords[-1]) - np.asarray(seg.coords[0]))
            if abs(float(np.dot(tv, u))) > 0.8:
                mark_len += seg.length
    if mark_len >= 12.0:
        return True
    # 证据 2：宽路面（≥8 m 且 20 m 外仍宽 ≥7 m）+ 参考线或 ≥10 m 宽主干道
    if drivable is not None and arm["width"] >= 8.0:
        cross = LineString([m + u * 20.0 - lat * 15.0, m + u * 20.0 + lat * 15.0])
        w20 = cross.intersection(drivable).length
        if w20 >= 8.0 and arm["width"] >= 10.0:
            return True
        if w20 >= 7.0:
            for r in refs:
                if r.intersects(probe) and r.intersection(probe).length >= 12.0:
                    return True
    return False


def _arm_stop_ids(arm, core_stops, stop_dirs, tol=4.0):
    """贴在本臂口门上、方向指向核心（= 本臂进口）的本路口停止线序号。"""
    out = set()
    for k, (s_, d_) in enumerate(zip(core_stops, stop_dirs)):
        if d_ is None or s_.length < 2.0:
            continue
        if min(m.distance(s_) for m in arm["mouths"]) > tol:
            continue
        if float(np.dot(d_, arm["n"])) < -0.5:
            out.add(k)
    return out


_MOVE_SECTOR = {  # 转向 → 出口臂外向方位相对进口行驶方位的顺时针夹角范围（度），斜交路口也接得住
    "S": ((0.0, 35.0), (325.0, 360.0)),
    "R": ((20.0, 160.0),),
    "L": ((200.0, 340.0),),
    "r": ((10.0, 90.0),),
    "l": ((270.0, 350.0),),
}


def _arrow_moves(arrows, core_stops, reach=70.0):
    """本路口进口导向箭头（前向射线 reach 内碰到本路口停止线）的 (行驶方位角, 转向) 列表；掉头 T 不指向别的臂。"""
    from .junction import movement
    if not core_stops:
        return []
    su = unary_union(list(core_stops))
    out = []
    for p, v, code in arrows:
        if not LineString([p, p + v * reach]).intersects(su):
            continue
        hb = (math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0
        out += [(hb, ch) for ch in movement(code) if ch in _MOVE_SECTOR]
    return out


def _in_sector(arm_bearing, hb, ch):
    d = (arm_bearing - hb + 360.0) % 360.0
    return any(lo <= d <= hi for lo, hi in _MOVE_SECTOR[ch])


def _minor_exits(arms, kept, moves, road, core, refs):
    """无标线口门须有接续路面。共享箭头只提供候选资格，不能单独证明各支路。"""
    weak = []
    strong_ids = {a["id"] for a in kept}
    for a in arms:
        labels = {(hb, ch) for hb, ch in moves if _in_sector(a["bearing"], hb, ch)}
        if a["id"] in strong_ids or not labels or road is None or a["width"] < 3.0:
            continue
        mid = Point(a["mid"])
        mouth = unary_union(a["mouths"]).buffer(2.0)
        outside = road.intersection(mid.buffer(60.0)).difference(core.buffer(1.0))
        branches = [p for p in _parts(outside) if p.intersects(mouth) and mid.hausdorff_distance(p) >= 25.0]
        if not branches:
            continue
        # 同一有向参考线穿过口门并沿其接续路面向外延伸，可独立支持这个出口。
        supported = False
        for ref in refs:
            if not ref.intersects(mouth):
                continue
            for branch in branches:
                for seg in _parts(ref.intersection(branch), "LineString"):
                    if seg.intersects(mouth):
                        displacement = np.asarray(seg.coords[-1]) - np.asarray(seg.coords[0])
                        if np.dot(displacement[:2], a["n"]) >= a["width"]:
                            supported = True
        weak.append((a, labels, supported))
    confirmed, uncertain = [], []
    for a, labels, supported in weak:
        unique = any(not any(_in_sector(k["bearing"], hb, ch) for k in kept)
                     and sum((hb, ch) in others for _, others, _ in weak) == 1 for hb, ch in labels)
        if supported or unique:
            a["minor_branch"] = True
            confirmed.append(a)
        else:
            uncertain.append(dict(bearing=a["bearing"], mouths=a["mouths"], reason="shared_arrow_without_branch_evidence"))
    return confirmed, uncertain


def pair_arms(arms, lanes, blocks_u=None, max_ang=28.0, max_lat=65.0, max_lon=60.0, core_stops=(), stop_dirs=(),
              indep_ang=22.0):
    """合并同一条道路被岛/宽中分带/斜导流区拆开的口门。
    两个口门各自有独立的进口停止线、且道路方向差 > indep_ang 时是两条路（斜交多岔口），不合并。"""
    for a in arms:
        a["u"] = _road_dir(a, lanes)
        a["_stops"] = _arm_stop_ids(a, core_stops, stop_dirs)
    merged = True
    while merged:
        merged = False
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                ang = _ang_deg(a["u"], b["u"])
                # 若两个口门紧挨着（< 18 m），允许放宽到 38°（如斜切角口门导致法向略偏）
                mdist = min(ma.distance(mb) for ma in a["mouths"] for mb in b["mouths"])
                both_stops = bool(a["_stops"]) and bool(b["_stops"]) and not (a["_stops"] & b["_stops"])
                if both_stops and ang > indep_ang:
                    continue
                if ang > (38.0 if (mdist < 18.0 and not both_stops) else max_ang):
                    continue
                u = _unit(a["u"] + b["u"])
                d = b["mid"] - a["mid"]
                lat_, lon_ = abs(u[0] * d[1] - u[1] * d[0]), abs(float(np.dot(u, d)))
                if lat_ > max_lat or lon_ > max_lon:
                    continue
                if lat_ > 32.0 and blocks_u is not None and not blocks_u.is_empty:
                    if LineString([a["mid"] + u * 45.0, b["mid"] + u * 45.0]).intersection(blocks_u).length > 15.0:
                        continue
                a["mouths"] = a["mouths"] + b["mouths"]
                a["width"] += b["width"]
                a["_stops"] = a["_stops"] | b["_stops"]
                a["minor_branch"] = bool(a.get("minor_branch")) and bool(b.get("minor_branch"))
                a["mid"] = np.mean([np.asarray(m.interpolate(0.5, normalized=True).coords[0]) for m in a["mouths"]], axis=0)
                for ln in lanes:
                    if ln.get("arm") == b["id"]:
                        ln["arm"] = a["id"]
                a["u"] = _road_dir(a, lanes)
                arms.pop(j)
                merged = True
                break
            if merged:
                break
    for a in arms:
        a["n"] = a["u"]
        a["bearing"] = (math.degrees(math.atan2(a["u"][0], a["u"][1])) + 360) % 360
    arms.sort(key=lambda a: a["bearing"])
    remap = {a["id"]: i for i, a in enumerate(arms)}
    for i, a in enumerate(arms):
        a["id"] = i
    for ln in lanes:
        if ln.get("arm") is not None:
            ln["arm"] = remap.get(ln["arm"])
    return arms


# ---------------------------------------------------------------- 4. 绕岛渠化右转拓扑检出

def _match_slip_arm(pt, tan_vec, arms, is_from, max_d=65.0):
    """匹配渠化右转端点对应的路臂。"""
    best = None
    p = Point(pt)
    for a in arms:
        d = min(m.distance(p) for m in a["mouths"])
        ray = LineString([a["mid"], a["mid"] + a["n"] * 80.0])
        dr = ray.distance(p)
        if min(d, dr) > max_d:
            continue
        align = float(np.dot(tan_vec, -a["n"] if is_from else a["n"]))
        if align < 0.2:
            continue
        score = min(d, dr + 4.0) + (1.0 - align) * 20.0
        if best is None or score < best[0]:
            best = (score, a["id"])
    return best[1] if best else None


import os as _os
_DBG_SLIP = bool(_os.environ.get("LANEGEN_DEBUG_SLIP"))


def _arm_strip(arm, core_poly, depth=45.0):
    """路臂口门沿外向方向拉出的条带（去掉核心）：判断走廊是否接到该臂。"""
    n = arm["n"]
    polys = []
    for m in arm["mouths"]:
        a, b = np.asarray(m.coords[0]), np.asarray(m.coords[-1])
        polys.append(Polygon([a, b, b + n * depth, a + n * depth]).buffer(3.0))
    return unary_union(polys).difference(core_poly.buffer(1.0))


def _corridor_slips(obst, dv_clean, core_poly, arms, w, existing, zone_r=30.0, cell=1.0):
    """带障碍约束的绕岛通道检出：对外环扫描没给出 slip 的转角岛，取岛周 zone_r 内、去掉核心后的路面；
    若同一连通块同时接到 from_arm 与右转目标 to_arm 的条带，就在该块里按“离路缘越远越便宜”求最短中心通道，
    保留贴着岛走（≤ 2 倍车道宽 + 6 m）的那一段作为绕行道轴线。两端必须分别落进两臂条带（排除死胡同）。"""
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    out = []
    # 只有两条臂时，绕岛连通走廊就是主路本身（弯道 / 分幅），不是渠化右转
    if dv_clean is None or dv_clean.is_empty or len(arms) < 3:
        return out
    strips = {a["id"]: _arm_strip(a, core_poly) for a in arms}
    for ob in obst:
        if ob["kind"] != "corner_island" or ob["area"] < 30.0:
            continue
        O = ob["geom"]
        if any(sl.get("obstacle_geom") is not None and sl["obstacle_geom"].equals(O) for sl in existing):
            continue
        if any(sl["corridor_poly"].distance(O) < 1.5 for sl in existing + out):
            continue
        R = dv_clean.intersection(O.buffer(zone_r)).difference(core_poly.buffer(1.0))
        R = R.buffer(-0.8).buffer(0.8)
        for comp in _parts(R):
            if comp.distance(O) > 2.0 or comp.area < 40.0:
                continue
            touch = [a for a in arms if comp.intersection(strips[a["id"]]).area >= 15.0]
            pairs = []
            for fa in touch:
                for ta in touch:
                    if fa["id"] == ta["id"]:
                        continue
                    db = (ta["bearing"] - fa["bearing"] + 360) % 360
                    if 195.0 <= db <= 345.0:
                        pairs.append((fa, ta))
            if not pairs:
                continue
            x0, y0, x1, y1 = comp.bounds
            xs = np.arange(x0 + cell / 2, x1, cell)
            ys = np.arange(y0 + cell / 2, y1, cell)
            gx, gy = np.meshgrid(xs, ys)
            mask = shapely.contains_xy(comp, gx, gy)
            if mask.sum() < 30:
                continue
            dt = distance_transform_edt(mask) * cell
            idx = -np.ones(mask.shape, int)
            idx[mask] = np.arange(int(mask.sum()))
            rr, cc = np.nonzero(mask)
            rows, cols, wts = [], [], []
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                r2, c2 = rr + dr, cc + dc
                ok = (r2 >= 0) & (r2 < mask.shape[0]) & (c2 >= 0) & (c2 < mask.shape[1])
                ok[ok] = mask[r2[ok], c2[ok]]
                a_i, b_i = idx[rr[ok], cc[ok]], idx[r2[ok], c2[ok]]
                step = cell * (math.sqrt(2.0) if dr and dc else 1.0)
                cst = step / (np.minimum(dt[rr[ok], cc[ok]], dt[r2[ok], c2[ok]]) + 0.3)
                rows += [a_i, b_i]; cols += [b_i, a_i]; wts += [cst, cst]
            nn = int(mask.sum())
            G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(nn, nn)).tocsr()
            pts = np.stack([gx[mask], gy[mask]], 1)
            for fa, ta in pairs:
                src = np.nonzero(shapely.contains_xy(strips[fa["id"]], pts[:, 0], pts[:, 1]))[0]
                dst = np.nonzero(shapely.contains_xy(strips[ta["id"]], pts[:, 0], pts[:, 1]))[0]
                if len(src) == 0 or len(dst) == 0:
                    continue
                dist, pred, srcs = dijkstra(G, indices=src, min_only=True, return_predecessors=True)
                j = int(dst[np.argmin(dist[dst])])
                if not np.isfinite(dist[j]):
                    continue
                path = [j]
                while pred[path[-1]] >= 0:
                    path.append(int(pred[path[-1]]))
                path = path[::-1]
                P = pts[path]
                if len(P) < 8:
                    continue
                # 只保留贴着岛走的一段（远离岛的部分是两臂自己的车行道）
                near = np.array([Point(q).distance(O) for q in P]) <= 2.0 * w + 6.0
                if near.sum() < 6:
                    continue
                k0, k1 = int(np.argmax(near)), len(near) - 1 - int(np.argmax(near[::-1]))
                P = _smooth_pts(P[k0:k1 + 1], win=7)
                axis = LineString(P)
                if not (12.0 <= axis.length <= 80.0):
                    continue
                wd = 2.0 * np.array([dt[rr[i], cc[i]] for i in path[k0:k1 + 1]])
                if float(np.median(wd)) > 9.5:
                    continue  # 比两车道绕行道还宽得多：是路口转角的开阔路面，不是渠化右转道
                w_s = float(np.clip(np.median(wd), 3.0, 9.5))
                poly = axis.buffer(0.5 * w_s, cap_style="flat").intersection(comp).buffer(0)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
                if poly.is_empty or poly.area < 25.0:
                    continue
                if any(sl["corridor_poly"].intersection(poly).area > 0.3 * min(poly.area, sl["corridor_poly"].area)
                       for sl in existing + out):
                    continue
                if _DBG_SLIP:
                    print("  [corridor-slip]", fa["id"], "->", ta["id"], "len", round(axis.length, 1), "w", round(w_s, 1))
                out.append(dict(id=len(existing) + len(out), lane=None, lanes=[], from_arm=fa["id"], to_arm=ta["id"],
                                diverge_pt=tuple(P[0]), merge_pt=tuple(P[-1]), axis=axis, corridor_poly=poly,
                                width_m=round(w_s, 2), obstacle=ob["kind"], obstacle_geom=O, source="island_corridor"))
                break
    return out


def _merge_parallel_slips(slips, w):
    out = []
    for s in slips:
        host = next((o for o in out if o["from_arm"] == s["from_arm"] and o["to_arm"] == s["to_arm"]
                     and o["corridor_poly"].distance(s["corridor_poly"]) <= 1.5
                     and o["axis"].hausdorff_distance(s["axis"]) <= 2.5 * w), None)
        if host is None:
            out.append(s)
            continue
        host["corridor_poly"] = unary_union([host["corridor_poly"], s["corridor_poly"]]).buffer(0.3).buffer(-0.3)
        if host["corridor_poly"].geom_type == "MultiPolygon":
            host["corridor_poly"] = max(host["corridor_poly"].geoms, key=lambda g: g.area)
        host["n_lanes_hint"] = host.get("n_lanes_hint", 1) + 1
        host["extra_axes"] = host.get("extra_axes", []) + [s["axis"]]
    for k, s in enumerate(out):
        s["id"] = k
    return out


def slip_connectivity(slips, cws, core_poly, w):
    """绕行道两端是否真的接上：分流端贴 from_arm 的进口车行道（或核心），汇入端贴 to_arm 的出口车行道（或核心）。
    返回 (保留的 slips)；两端都接不上的删掉，只一端接不上的标 dead_end。"""
    tol = max(12.0, 3.0 * w)
    keep = []
    for s in slips:
        p0, p1 = Point(s["diverge_pt"]), Point(s["merge_pt"])
        src = [c["polygon"] for c in cws if c["arm"] == s["from_arm"] and c["role"] == "approach"]
        dst = [c["polygon"] for c in cws if c["arm"] == s["to_arm"] and c["role"] == "exit"]
        ok0 = p0.distance(core_poly) <= tol or any(p0.distance(g) <= tol for g in src)
        ok1 = p1.distance(core_poly) <= tol or any(p1.distance(g) <= tol for g in dst)
        s["dead_end"] = not (ok0 and ok1)
        if _DBG_SLIP:
            print("  [slip-conn]", s["from_arm"], s["to_arm"], "d0core", round(p0.distance(core_poly), 1),
                  "d0src", [round(p0.distance(g), 1) for g in src], "d1core", round(p1.distance(core_poly), 1),
                  "d1dst", [round(p1.distance(g), 1) for g in dst])
        if ok0 or ok1:
            keep.append(s)
    for k, s in enumerate(keep):
        s["id"] = k
    return keep


def detect_bypass_slips(obst, drivable, core_poly, arms, lanes, w):
    """基于转角岛（corner_island）外环法向扫描检出绕岛渠化右转走廊。
    CCW 遍历岛外轮廓时，岛体恒在左侧——正是右行规则下右转车辆绕岛行驶的方向。"""
    slips = []
    if drivable is None or drivable.is_empty or len(arms) < 2:
        return slips
    dv_clean = drivable.buffer(0.2).buffer(-0.2)
    core_buf = core_poly.buffer(1.0)
    step = 1.5
    max_w = max(12.0, 3.3 * w)

    for ob in obst:
        if ob["kind"] != "corner_island":
            continue
        O = ob["geom"]
        ring = shapely.geometry.polygon.orient(O, sign=1.0).exterior  # CCW：前进时岛在左
        L = ring.length
        if L < 12.0:
            continue
        n_samp = max(12, int(round(L / step)))
        ss = np.linspace(0.0, L, n_samp, endpoint=False)
        pts = shapely.get_coordinates(shapely.line_interpolate_point(ring, ss))
        prev_p = np.roll(pts, 1, axis=0)
        next_p = np.roll(pts, -1, axis=0)
        tv = next_p - prev_p
        tn = np.linalg.norm(tv, axis=1, keepdims=True)
        tv = tv / np.where(tn < 1e-9, 1.0, tn)
        nout = np.stack([tv[:, 1], -tv[:, 0]], axis=1)

        valid = np.zeros(n_samp, bool)
        mids = np.zeros((n_samp, 2))
        outers = np.zeros((n_samp, 2))
        widths = np.zeros(n_samp)

        for i in range(n_samp):
            p = pts[i]
            if core_buf.contains(Point(p)):
                continue
            ray = LineString([p + nout[i] * 0.15, p + nout[i] * (max_w + 1.0)])
            x = ray.intersection(dv_clean)
            if x.is_empty:
                continue
            segs = _parts(x, "LineString")
            first = next((sg for sg in segs if sg.distance(Point(p)) < 0.6), None)
            if first is None:
                continue
            d = first.length
            if not (2.2 <= d <= max_w):
                continue
            q = p + nout[i] * d
            mid = p + nout[i] * (0.5 * d)
            if core_buf.contains(Point(mid)) or core_buf.contains(Point(q)):
                continue
            valid[i] = True
            mids[i] = mid
            outers[i] = q
            widths[i] = d

        if valid.sum() < 5:
            continue
        v2 = valid.copy()
        for i in range(n_samp):
            if not v2[i] and valid[(i - 1) % n_samp] and valid[(i + 1) % n_samp]:
                v2[i] = True
                mids[i] = 0.5 * (mids[(i - 1) % n_samp] + mids[(i + 1) % n_samp])
                outers[i] = 0.5 * (outers[(i - 1) % n_samp] + outers[(i + 1) % n_samp])
                widths[i] = 0.5 * (widths[(i - 1) % n_samp] + widths[(i + 1) % n_samp])
        if v2.all():
            continue
        runs = []
        start_zero = int(np.argmin(v2))
        cur = []
        for k in range(n_samp):
            idx = (start_zero + k) % n_samp
            if v2[idx]:
                cur.append(idx)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        # 在岛轮廓的尖角处把连续段切开：三角岛的“面向核心 / 面向路臂 / 面向绕行道”三条边各自判，
        # 否则一条段跨两条边，首尾切向和起止臂都会错（随机 4 东北角岛实测：整段被配到同一臂）
        corner = np.array([float(np.dot(tv[(i - 2) % n_samp], tv[(i + 2) % n_samp])) < math.cos(math.radians(45.0))
                           for i in range(n_samp)])
        pieces = []  # 先整段判（保持原有检出），再判尖角切开后的各条边（补漏检）
        for run in runs:
            cur = []
            for idx in run:
                if corner[idx]:
                    if cur:
                        pieces.append(cur)
                    cur = []
                else:
                    cur.append(idx)
            if cur:
                pieces.append(cur)
        runs = runs + [pc for pc in pieces if not any(pc == r for r in runs)]

        for run in runs:
            if len(run) < 5:
                continue
            c_pts = _smooth_pts(mids[run], win=5)
            axis = LineString(c_pts)
            if not (8.0 <= axis.length <= 150.0):
                continue
            p0, p1 = np.asarray(axis.coords[0]), np.asarray(axis.coords[-1])
            t0 = _tan(axis, min(2.0, 0.25 * axis.length))
            t1 = _tan(axis, max(axis.length - 2.0, 0.75 * axis.length))
            fa = _match_slip_arm(p0, t0, arms, is_from=True)
            ta = _match_slip_arm(p1, t1, arms, is_from=False)
            if _DBG_SLIP:
                print("  [slip] run", len(run), "len", round(axis.length, 1), "fa", fa, "ta", ta, "turn", round(_signed_turn_deg(axis)),
                      np.round(p0, 1), np.round(p1, 1), "t0", np.round(t0, 2), "t1", np.round(t1, 2),
                      [round(float(np.dot(t0, -a["n"])), 2) for a in arms], [round(float(np.dot(t1, a["n"])), 2) for a in arms])
            if fa is None or ta is None or fa == ta:
                continue
            turn = _signed_turn_deg(axis)
            db = (arms[ta]["bearing"] - arms[fa]["bearing"] + 360) % 360
            # 真绕岛右转必须同时满足：右转曲率（turn < -10°）+ 右转目标臂方位（195°–345°）+ 主体不在路臂主口门正中
            if turn >= -10.0 or not (195.0 <= db <= 345.0):
                continue
            mid_pt = axis.interpolate(0.5, normalized=True)
            if any(m.distance(mid_pt) < 4.0 for a in arms for m in a["mouths"]):
                continue
            in_pts = pts[run]
            out_pts = outers[run]
            poly = Polygon(np.vstack([in_pts, out_pts[::-1]])).buffer(0)
            if not poly.is_valid or poly.is_empty:
                poly = axis.buffer(0.5 * float(np.median(widths[run])), cap_style="flat")
            poly = poly.intersection(dv_clean).buffer(0)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.is_empty or poly.area < 15.0:
                continue
            # 去重：与已有走廊重叠 > 30%（同一对臂的重合岛段，或整段/切开段重复判出）→ 同一对臂留更长的，否则跳过
            dup = [k for k, s in enumerate(slips)
                   if s["corridor_poly"].intersection(poly).area > 0.3 * min(poly.area, s["corridor_poly"].area)]
            if dup:
                k = dup[0]
                if not (len(dup) == 1 and slips[k]["from_arm"] == fa and slips[k]["to_arm"] == ta
                        and axis.length > slips[k]["axis"].length):
                    continue
                slips.pop(k)
            slips.append(dict(
                id=len(slips),
                lane=None,
                lanes=[],
                from_arm=fa,
                to_arm=ta,
                diverge_pt=tuple(p0),
                merge_pt=tuple(p1),
                axis=axis,
                corridor_poly=poly,
                width_m=round(float(np.median(widths[run])), 2),
                obstacle=ob["kind"],
                obstacle_geom=O,
                source="bypass_island",
            ))

    # 外环扫描没检出的转角岛：用“去掉核心后的路面连通走廊 + 最短中心通道”再验一次
    slips += _corridor_slips(obst, dv_clean, core_poly, arms, w, slips)

    # 同一对 (from_arm, to_arm)、走廊相邻（并排的双车道绕行道被两个岛/两侧分别扫出）→ 合并成一条多车道 slip
    slips = _merge_parallel_slips(slips, w)

    # 吸收落入物理绕岛走廊内的零散车道
    for s in slips:
        sp = s["corridor_poly"].buffer(1.2)
        for i, ln in enumerate(lanes):
            g = ln["geom"]
            if g.intersection(sp).length < 0.45 * g.length:
                continue
            if any(bool(set(m) - {"R", "r"}) for _, m in ln.get("arrows", [])):
                continue
            ln["role"] = "slip"
            ln["slip"] = True
            ln["arm"] = None
            ln["slip_from_arm"] = s["from_arm"]
            ln["slip_to_arm"] = s["to_arm"]
            s["lanes"].append(i)
        if s["lanes"]:
            s["lane"] = max(s["lanes"], key=lambda k: lanes[k]["geom"].length)

    # 兜底：保留已由散线识别出、但绕过超大空洞（>1800 m²）的远距右转匝道（如样例 1 西南匝道）
    for i, ln in enumerate(lanes):
        if ln.get("role") != "slip":
            continue
        if any(i in s["lanes"] for s in slips):
            continue
        g = ln["geom"]
        p0, p1 = np.asarray(g.coords[0]), np.asarray(g.coords[-1])
        t0 = _tan(g, min(3.0, 0.25 * g.length))
        t1 = _tan(g, max(g.length - 3.0, 0.75 * g.length))
        fa = _match_slip_arm(p0, t0, arms, is_from=True, max_d=70.0)
        ta = _match_slip_arm(p1, t1, arms, is_from=False, max_d=70.0)
        # 已有同一对臂的绕岛走廊紧挨着这条车道 → 它是那条走廊的另一条车道，并入而不是另建一条 slip
        host = next((s for s in slips if s["from_arm"] == fa and s["to_arm"] == ta and s.get("corridor_poly") is not None
                     and g.distance(s["corridor_poly"]) <= 1.6 * w
                     and g.intersection(s["corridor_poly"].buffer(1.6 * w)).length >= 0.5 * g.length), None)
        if host is not None:
            ln["slip_from_arm"], ln["slip_to_arm"] = fa, ta
            host["lanes"].append(i)
            host["n_lanes_hint"] = host.get("n_lanes_hint", 1) + 1
            host["corridor_poly"] = unary_union([host["corridor_poly"], g.buffer(0.5 * ln["width"], cap_style="flat")]).buffer(0)
            continue
        mid = g.interpolate(0.5, normalized=True)
        cand = [o for o in obst if o["geom"].distance(mid) < 15.0]
        isl = min(cand, key=lambda o: o["geom"].distance(mid)) if cand else None
        poly = g.buffer(0.5 * ln["width"], cap_style="flat")
        if dv_clean is not None:
            poly = poly.intersection(dv_clean).buffer(0)
        s_obj = dict(
            id=len(slips),
            lane=i,
            lanes=[i],
            from_arm=fa,
            to_arm=ta,
            diverge_pt=tuple(p0),
            merge_pt=tuple(p1),
            axis=g,
            corridor_poly=poly,
            width_m=round(float(ln["width"]), 2),
            obstacle=None if isl is None else isl["kind"],
            obstacle_geom=None if isl is None else isl["geom"],
            source="lane_fallback",
        )
        ln["slip_from_arm"], ln["slip_to_arm"] = fa, ta
        slips.append(s_obj)
    return slips


# ---------------------------------------------------------------- 5. 物理单向车行道切分与 Frenet 基线提取

def _seal_hard_dividers(B, core_poly, obst, max_ext=12.0):
    """硬分隔两端封口：
    1. 近核心端（≤12 m）：黄线、物理隔离与岛鼻延长封口到 core_poly，防止在口门处粘连；
    2. 远端（背离核心端）：黄线、物理隔离沿端点切向向外延长 45 m，防止标线在路段中间（如 70 m 处或掉头口）
       提前终止导致左右两幅车行道在远端绕成 U 形整块（越秀 7919 北臂实测）。"""
    cuts = []
    core_ext = core_poly.buffer(0.3)
    cc = np.asarray(core_poly.centroid.coords[0])
    for l, t in B["bounds"]:
        if t not in ("yellow", "sep"):
            continue
        r = 0.18 if t == "yellow" else 0.25
        cuts.append(l.buffer(r))
        if l.length < 3.0:
            continue
        for which in (0, 1):
            c = np.asarray(l.coords)
            p = c[0] if which == 0 else c[-1]
            q = np.asarray(l.interpolate(min(4.0, l.length) if which == 0 else max(0.0, l.length - 4.0)).coords[0])
            tv = _unit(p - q)
            dc = Point(p).distance(core_poly)
            if dc <= max_ext:
                ray = LineString([p, p + tv * (max_ext + 2.0)])
                if ray.intersects(core_ext):
                    x = ray.intersection(core_ext)
                    d = min(Point(p).distance(pt) for pt in getattr(x, "geoms", [x]) if not pt.is_empty)
                    if 0.05 < d <= max_ext + 1.0:
                        cuts.append(LineString([p, p + tv * (d + 0.5)]).buffer(r))
                        continue
            # 远端（切向背离核心中心）：向外延 45 m 切断远端 U 形粘连（只延离核心 ≤110 m 内的中分线）
            if dc <= 110.0 and np.dot(tv, p - cc) > 0.3:
                cuts.append(LineString([p, p + tv * 45.0]).buffer(r))
    # 靠近核心 ≤ 6 m 的岛鼻端封口到核心
    for ob in obst:
        if ob["kind"] == "block" or ob["dist_core"] > 10.0 or ob["dist_core"] < 0.2:
            continue
        O = ob["geom"]
        a_pt, b_pt = shapely.ops.nearest_points(O, core_poly)
        if a_pt.distance(b_pt) <= 6.0:
            cuts.append(LineString([a_pt, b_pt]).buffer(0.25))
    return unary_union(cuts) if cuts else None


def _trace_poly_axis(poly, mouth, u_out, role, lanes_in=(), max_len=120.0, step=2.0):
    """抗路侧支路干扰的物理车行道 Frenet 中心轴线与中位宽度提取。
    每步限制左右边界变化幅度（单步 ≤ 0.7 m），路侧支路开口不再把中心线拉弯。"""
    if mouth is not None and not mouth.is_empty:
        p = np.asarray(mouth.interpolate(0.5, normalized=True).coords[0])
    else:
        p = np.asarray(poly.representative_point().coords[0])
    t = _unit(u_out)
    for off in (0.5, 1.5, 3.0, 5.0):
        q = p + t * off
        if poly.contains(Point(q)):
            p = q
            break
    else:
        _, q = shapely.ops.nearest_points(Point(p), poly)
        p = np.asarray(q.coords[0])

    lane_geoms = [ln["geom"] for ln in lanes_in]

    mids = []
    widths = []
    cur = p.copy()
    cur_t = t.copy()
    prev_dl = prev_dr = None  # 相对 cur 的左右边界距离（dl >= 0 在左，dr >= 0 在右）
    x0, y0, x1, y1 = poly.bounds
    reach = math.hypot(x1 - x0, y1 - y0)

    for _ in range(int(max_len / step)):
        n = np.array([-cur_t[1], cur_t[0]])  # 左法向
        cross = LineString([cur - n * reach, cur + n * reach])
        x = cross.intersection(poly)
        segs = _parts(x, "LineString")
        if not segs:
            break
        seg = min(segs, key=lambda sg: sg.distance(Point(cur)))
        if seg.distance(Point(cur)) > 4.0:
            break
        c0, c1 = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1])
        d0 = float(np.dot(c0 - cur, n))
        d1 = float(np.dot(c1 - cur, n))
        dl = max(d0, d1)
        dr = -min(d0, d1)
        if dl < -0.5 or dr < -0.5:
            dl, dr = 0.5 * seg.length, 0.5 * seg.length
            mid = 0.5 * (c0 + c1)
        else:
            # 抗支路开口：若单侧突然外扩 > 1.2 m（遇到路侧支路），钳制该侧边界，避免中点跳进支路
            if prev_dl is not None:
                if dl > prev_dl + 1.2:
                    dl = prev_dl + 0.4
                if dr > prev_dr + 1.2:
                    dr = prev_dr + 0.4
            mid = cur + n * (0.5 * (dl - dr))
        w_cur = dl + dr
        if mids and np.linalg.norm(mid - mids[-1]) > 6.0:
            break
        mids.append(mid)
        widths.append(w_cur)
        prev_dl = 0.5 * w_cur
        prev_dr = 0.5 * w_cur

        # 平滑更新局部外向切向
        new_t = cur_t.copy()
        if lane_geoms:
            vt = np.zeros(2)
            pt = Point(mid)
            for lg in lane_geoms:
                if lg.distance(pt) < max(6.0, 0.6 * w_cur):
                    s_ = lg.project(pt)
                    lt = _tan(lg, s_, 2.5)
                    if np.dot(lt, cur_t) < 0:
                        lt = -lt
                    vt += lt
            if np.linalg.norm(vt) > 1e-6:
                new_t = _unit(0.45 * cur_t + 0.55 * _unit(vt))
        elif len(mids) >= 4:
            dt = _unit(mids[-1] - mids[-4])
            if np.dot(dt, cur_t) > 0.8:
                new_t = _unit(0.7 * cur_t + 0.3 * dt)
        cur_t = new_t
        cur = mid + cur_t * step

    if len(mids) < 2:
        if lane_geoms:
            best = max(lane_geoms, key=lambda g: g.length)
            return best, float(np.median([ln["width"] for ln in lanes_in]) * len(lanes_in))
        return None, 0.0

    w_med = float(np.median(widths))
    pts = _smooth_pts(mids, win=9)
    if role == "approach":
        pts = pts[::-1]
    return LineString(pts), w_med


def _split_undivided_poly(poly, in_lanes, out_lanes, u_out):
    """无黄线/无物理中分带的双向车行道多边形：沿进/出口车道之间的分界中线一分为二（兼容正常右行与立交交织）。"""
    lat = np.array([-u_out[1], u_out[0]])
    ref_pt = np.asarray(poly.representative_point().coords[0])

    def sample_sd(geoms):
        S, D = [], []
        for g in geoms:
            ss = np.arange(0.0, g.length + 1e-6, 3.0)
            pts = shapely.get_coordinates(shapely.line_interpolate_point(g, ss))
            rel = pts - ref_pt
            S.append(rel @ u_out)
            D.append(rel @ lat)
        return np.concatenate(S), np.concatenate(D)

    si, di = sample_sd([ln["geom"] for ln in in_lanes])
    so, do = sample_sd([ln["geom"] for ln in out_lanes])
    in_on_left = float(np.median(di)) >= float(np.median(do))
    s_lo = min(si.min(), so.min()) - 20.0
    s_hi = max(si.max(), so.max()) + 20.0
    bins = np.arange(s_lo, s_hi + 1.0, 5.0)
    cut_pts = []
    for sb in bins:
        mi = np.abs(si - sb) < 8.0
        mo = np.abs(so - sb) < 8.0
        if mi.any() and mo.any():
            if in_on_left:
                d_mid = 0.5 * (float(np.min(di[mi])) + float(np.max(do[mo])))
            else:
                d_mid = 0.5 * (float(np.max(di[mi])) + float(np.min(do[mo])))
            cut_pts.append(ref_pt + u_out * sb + lat * d_mid)
    if len(cut_pts) < 2:
        d_mid = 0.5 * (float(np.min(di)) + float(np.max(do))) if in_on_left else 0.5 * (float(np.max(di)) + float(np.min(do)))
        cut_pts = [ref_pt + u_out * (s_lo - 20.0) + lat * d_mid, ref_pt + u_out * (s_hi + 20.0) + lat * d_mid]
    else:
        t0 = _unit(cut_pts[1] - cut_pts[0])
        t1 = _unit(cut_pts[-1] - cut_pts[-2])
        cut_pts = [cut_pts[0] - t0 * 25.0] + cut_pts + [cut_pts[-1] + t1 * 25.0]
    parts = _parts(poly.difference(LineString(cut_pts).buffer(0.15)))
    return [p for p in parts if p.area >= 12.0]


def _split_unmarked_twoway(poly, arm_mid, u_out):
    """完全无车道、宽度 ≥ 5.5 m 的双向未分隔路臂：沿外向中心线二分左右半幅。"""
    lat = np.array([-u_out[1], u_out[0]])
    coords = np.asarray(poly.exterior.coords)[:, :2]
    s_vals = (coords - arm_mid) @ u_out
    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    mids = []
    for sb in np.arange(s_min + 2.0, s_max, 5.0):
        base = arm_mid + u_out * sb
        cross = LineString([base - lat * 25.0, base + lat * 25.0])
        segs = _parts(cross.intersection(poly), "LineString")
        if segs:
            sg = max(segs, key=lambda g: g.length)
            mids.append(np.asarray(sg.interpolate(0.5, normalized=True).coords[0]))
    if len(mids) < 2:
        return [poly]
    t0 = _unit(mids[1] - mids[0])
    t1 = _unit(mids[-1] - mids[-2])
    cut_pts = [mids[0] - t0 * 20.0] + mids + [mids[-1] + t1 * 20.0]
    parts = _parts(poly.difference(LineString(cut_pts).buffer(0.15)))
    return [p for p in parts if p.area >= 12.0]


STRONG = ("arrow+", "arrow-", "stop", "arrow_inferred", "yellow")


def _arrow_host_lane(pt, v, lanes, w):
    """箭头所在的 Step 1 车道：0.5 W + 0.5 m 内、走向与箭头一致（cos > 0.7）的最近车道下标；没有则 None。"""
    P_ = Point(pt)
    best = None
    for i, ln in enumerate(lanes):
        if ln.get("role") not in ("approach", "exit"):
            continue
        g = ln["geom"]
        d = g.distance(P_)
        if d > 0.5 * w + 0.5 or (best is not None and d >= best[0]):
            continue
        t = _tan(g, g.project(P_))
        if abs(float(np.dot(t, v))) > 0.7:
            best = (d, i)
    return best[1] if best else None


def _strength(ln):
    ev = ln.get("evidence", [])
    return abs(float(ln.get("score", 0.0))) if any(e in STRONG for e in ev) else 0.0


def build_physical_carriageways(lanes, arms, core_poly, drivable, B, obst, slips, arrows, refs, w, reach=125.0,
                                core_stops=(), stop_dirs=(), foreign_stops=()):
    """从物理路面 drivable - core_poly - slips 出发，用封口后的硬分隔切出物理单向车行道多边形与平滑 Frenet 轴线。"""
    cws = []
    if drivable is None or drivable.is_empty or not arms:
        return cws

    ext = drivable.intersection(core_poly.buffer(reach + 10.0)).difference(core_poly.buffer(0.15))
    slip_polys = [s["corridor_poly"].buffer(-0.2) for s in slips if s.get("corridor_poly") is not None and not s["corridor_poly"].is_empty]
    slip_polys = [p for p in slip_polys if not p.is_empty]
    if slip_polys:
        ext = ext.difference(unary_union(slip_polys))

    hard_u = _seal_hard_dividers(B, core_poly, obst)
    ext_cut = ext.difference(hard_u) if hard_u is not None else ext
    fcut = foreign_cut(foreign_stops)
    if fcut is not None:
        ext_cut = ext_cut.difference(fcut)

    for ln in lanes:
        if ln.get("role") != "slip":
            ln["carriageway"] = None

    for a in arms:
        aid = a["id"]
        u = a["n"]
        lat = np.array([-u[1], u[0]])
        mouth_u = unary_union(a["mouths"])
        m_pts = np.vstack([np.asarray(m.coords)[:, :2] for m in a["mouths"]])
        m_rel = m_pts - a["mid"]
        s_min = float(np.min(m_rel @ u)) - 5.0
        d_min = float(np.min(m_rel @ lat)) - 10.0
        d_max = float(np.max(m_rel @ lat)) + 10.0

        arm_lanes_init = [i for i, ln in enumerate(lanes) if ln.get("arm") == aid and ln.get("role") in ("approach", "exit")]
        for i in arm_lanes_init:
            pts = np.asarray(lanes[i]["geom"].coords)[:, :2]
            rel = pts - a["mid"]
            ok = (rel @ u) <= reach
            if ok.any():
                d_min = min(d_min, float(np.min((rel @ lat)[ok])) - 5.0)
                d_max = max(d_max, float(np.max((rel @ lat)[ok])) + 5.0)
        d_min = max(d_min, float(np.min(m_rel @ lat)) - 28.0)
        d_max = min(d_max, float(np.max(m_rel @ lat)) + 28.0)

        arm_env = Polygon([
            a["mid"] + u * s_min + lat * d_min,
            a["mid"] + u * s_min + lat * d_max,
            a["mid"] + u * reach + lat * d_max,
            a["mid"] + u * reach + lat * d_min,
        ])
        other_m = [unary_union(o["mouths"]).buffer(2.5) for o in arms if o["id"] != aid]
        if other_m:
            arm_env = arm_env.difference(unary_union(other_m))

        arm_road = ext_cut.intersection(arm_env)
        # 若本路臂内部存在中分带岛（严格夹在口门横向跨度内、且向远端延伸不横穿任何已有车道），
        # 将岛远端沿臂方向 u 向外封口，防止两幅路面绕过岛远端连成 U 形整块（同时绝不误切转角岛或斜向车行道）
        med_cuts = []
        m_dlo, m_dhi = float(np.min(m_rel @ lat)), float(np.max(m_rel @ lat))
        arm_lane_geoms = [lanes[i]["geom"] for i in arm_lanes_init]
        for ob in obst:
            if ob["kind"] == "block":
                continue
            O = ob["geom"]
            if not O.intersects(arm_env):
                continue
            oc = np.asarray(O.centroid.coords[0])
            od = float(np.dot(oc - a["mid"], lat))
            if not (m_dlo + 1.0 < od < m_dhi - 1.0):
                continue
            pts_o = np.asarray(O.exterior.coords)[:, :2]
            so = (pts_o - a["mid"]) @ u
            far_pt = pts_o[int(np.argmax(so))]
            near_pt = pts_o[int(np.argmin(so))]
            far_ray = LineString([far_pt, far_pt + u * (reach + 5.0)])
            if not any(far_ray.intersects(lg) for lg in arm_lane_geoms):
                med_cuts.append(far_ray.buffer(0.2))
            if Point(near_pt).distance(core_poly) <= 8.0:
                near_ray = LineString([near_pt, near_pt - u * 10.0])
                if not any(near_ray.intersects(lg) for lg in arm_lane_geoms):
                    med_cuts.append(near_ray.buffer(0.2))
        if med_cuts:
            arm_road = arm_road.difference(unary_union(med_cuts))

        raw_polys = []
        for p in _parts(arm_road):
            if p.area < 18.0:
                continue
            touches_mouth = p.distance(mouth_u) <= 4.5
            has_lane = any(lanes[i]["geom"].intersection(p.buffer(1.0)).length > 4.0 for i in arm_lanes_init)
            if not (touches_mouth or has_lane):
                continue
            # 被邻口停止线隔开、在其外侧的路面属于下一个路口
            if not touches_mouth and any(f.intersects(arm_env) and p.distance(core_poly) > f.distance(core_poly) - 1.0
                                         for f in foreign_stops):
                continue
            coords = np.asarray(p.exterior.coords)[:, :2]
            ext_s = float(np.max((coords - a["mid"]) @ u) - np.min((coords - a["mid"]) @ u))
            if ext_s < 8.0 and not has_lane:
                continue
            raw_polys.append(p)

        final_polys = []  # [(polygon, 是否无标线双向路对称二分的半幅)]
        for p in raw_polys:
            pb = p.buffer(0.1)  # 严禁外扩跨过 0.36 m 黄线切缝偷取对向车道
            inside = []
            for i, ln in enumerate(lanes):
                if ln.get("role") not in ("approach", "exit"):
                    continue
                g = ln["geom"]
                ov = g.intersection(pb).length
                if ov < min(6.0, 0.45 * g.length):
                    continue
                mid_t = _tan(g, 0.5 * g.length)
                if abs(float(np.dot(mid_t, u))) < 0.75:
                    continue
                inside.append(i)
            in_l = [lanes[i] for i in inside if lanes[i]["role"] == "approach" and _strength(lanes[i]) > 0]
            out_l = [lanes[i] for i in inside if lanes[i]["role"] == "exit" and _strength(lanes[i]) > 0]
            if not in_l:
                in_l = [lanes[i] for i in inside if lanes[i]["role"] == "approach"]
            if not out_l:
                out_l = [lanes[i] for i in inside if lanes[i]["role"] == "exit"]
            if in_l and out_l:
                final_polys.extend((q, False) for q in _split_undivided_poly(p, in_l, out_l, u))
            elif not inside:
                c_rel = np.asarray(p.exterior.coords)[:, :2] - a["mid"]
                d_span = float(np.max(c_rel @ lat) - np.min(c_rel @ lat))
                d_lo, d_hi = float(np.min(c_rel @ lat)), float(np.max(c_rel @ lat))
                w_est = 2.0 * p.area / max(p.exterior.length, 1e-6)  # 细长多边形的平均宽度
                if a.get("minor_branch") and w_est < 5.5:
                    final_polys.append((p, "shared"))  # 窄的无标线双向支路：不二分，整幅作一条双向共用车行道
                elif d_span >= 5.5 and d_lo < -0.5 and d_hi > 0.5:
                    halves = _split_unmarked_twoway(p, a["mid"], u)
                    final_polys.extend((q, len(halves) >= 2) for q in halves)
                else:
                    final_polys.append((p, False))
            else:
                final_polys.append((p, False))

        arm_cws_tmp = []
        for p, twoway_half in final_polys:
            pb = p.buffer(0.1)  # 严格落在单幅物理车行道内，不跨黄线/分隔带串道
            cand_idx = []
            for i, ln in enumerate(lanes):
                if ln.get("role") not in ("approach", "exit"):
                    continue
                g = ln["geom"]
                ov = g.intersection(pb).length
                if ov < min(6.0, 0.45 * g.length):
                    continue
                mid_t = _tan(g, 0.5 * g.length)
                if abs(float(np.dot(mid_t, u))) < 0.75:
                    continue
                cand_idx.append((ov, i))
            # 方向证据分三级：① 本幅内有向导向箭头（每枚 5）+ 贴在本幅上的本路口停止线（每条 6）——最可靠，一票定；
            # ② Step 1 车道角色按强度加权；③ 参考线走向；都没有才用右行默认（d_lat）
            pb4 = p.buffer(0.4)
            av_vote = av_weak = 0.0
            own = {i for _, i in cand_idx}
            for pt_, v_, code in arrows:
                if pb4.contains(Point(pt_)):
                    c_ = float(np.dot(v_, u))
                    # 箭头只给它所在车道的车行道投票：最近的同向 Step 1 车道若属于别的幅，说明多边形
                    # 在中分带开口/渠化处切偏了（开创大道×荔红二路：对向掉头箭头把出口判成进口）
                    # 没有宿主车道的纯掉头箭头只作弱证据（见下）：它画在贴中分带的最左车道，正是中分带开口处
                    # 两个方向的多边形相接、最容易切错的位置。其他无宿主箭头照常一票定（南沙 2035000001 北幅
                    # 3 枚无宿主箭头是对的、Step 1 车道方向是错的）
                    host = _arrow_host_lane(pt_, v_, lanes, w)
                    if host is not None and host not in own:
                        continue
                    if abs(c_) > 0.7:
                        if host is None and code == "arrow_uturn":
                            av_weak += -5.0 if c_ > 0 else 5.0
                        else:
                            av_vote += -5.0 if c_ > 0 else 5.0
            arrow_sum = av_vote
            pb6 = p.buffer(0.6)
            for s_, d_ in zip(core_stops, stop_dirs):
                if d_ is None or s_.length < 1.5:
                    continue
                if s_.intersection(pb6).length >= max(1.5, 0.3 * s_.length):
                    c_ = float(np.dot(d_, u))
                    if abs(c_) > 0.5:
                        av_vote += -6.0 if c_ > 0 else 6.0
            vote = 0.0
            for ov, i in cand_idx:
                sgn = 1.0 if lanes[i]["role"] == "approach" else -1.0
                st = _strength(lanes[i])
                vote += sgn * (st if st > 0 else 0.3) * min(ov, 40.0)
            evidence = "lanes" if abs(vote) > 1e-6 else None
            if abs(av_vote) < 1e-6 and abs(av_weak) > 1e-6 and not (abs(vote) > 1e-6 and vote * av_weak < 0):
                av_vote = av_weak  # 无宿主箭头：本幅车道没有相反的方向证据时才采信
            stop_sum = av_vote - arrow_sum
            if arrow_sum * stop_sum < 0 and abs(vote) > 1e-6:
                av_vote = 0.0  # 箭头与停止线互相矛盾（多边形蹭到对向停止线）：交给车道投票（海珠 6875000001 北出口）
            if abs(av_vote) > 1e-6:
                vote, evidence = av_vote, "arrow_stop"
            if abs(vote) < 1e-6 and refs:
                for r in refs:
                    rx = r.intersection(p)
                    segs = _parts(rx, "LineString")
                    if segs and sum(sg.length for sg in segs) >= 8.0:
                        lg = max(segs, key=lambda g: g.length)
                        rt = _tan(lg, 0.5 * lg.length)
                        dt = float(np.dot(rt, u))
                        if abs(dt) > 0.75:
                            vote += -1.0 if dt > 0 else 1.0
                if abs(vote) > 1e-6:
                    evidence = "refs"
            cp = np.asarray(p.representative_point().coords[0])
            d_lat = float(np.dot(cp - a["mid"], lat))
            if twoway_half == "shared" and abs(av_vote) < 1e-6:
                vote, evidence = -1.0, "shared_minor_branch"  # 主进口箭头指向它：作为出口建模（双向共用）
            if abs(vote) < 1e-6:
                role = "approach" if d_lat >= 0 else "exit"
                evidence = "right_hand_default"
            else:
                role = "approach" if vote > 0 else "exit"

            member_idx = [i for _, i in cand_idx if lanes[i]["role"] == role]
            conflict_idx = [i for _, i in cand_idx if lanes[i]["role"] != role]

            cut = mouth_u.intersection(p.buffer(2.0))
            cl = _parts(cut, "LineString")
            mouth = max(cl, key=lambda g: g.length) if cl else None
            if mouth is None and member_idx:
                ends = [np.asarray(lanes[i]["geom"].coords[-1] if role == "approach" else lanes[i]["geom"].coords[0]) for i in member_idx]
                if len(ends) >= 2:
                    mouth = LineString(sorted(ends, key=lambda q: float(np.dot(q, lat))))
                else:
                    hw = 0.5 * lanes[member_idx[0]]["width"]
                    mouth = LineString([ends[0] - lat * hw, ends[0] + lat * hw])

            axis, w_med = _trace_poly_axis(p, mouth, u, role, [lanes[i] for i in member_idx], max_len=reach)
            if axis is None or axis.length < 6.0:
                continue
            # 物理宽度门槛：< 2.2 m 的缝隙容不下机动车，无论是否有 Step 1 车道都不建车行道
            if w_med < 2.2:
                continue
            # 无标线、无同向箭头、无参考线、又不是无标线双向路对称二分出来的半幅，且宽 < 3.5 m → 小巷/出入口，不建机动车行道
            if not member_idx and not twoway_half and w_med < 3.5 and not a.get("minor_branch"):
                has_dir_arrow = any(pb4.contains(Point(pt_)) and abs(float(np.dot(v_, u))) > 0.7 for pt_, v_, _ in arrows)
                has_ref = any(sum(sg.length for sg in _parts(r.intersection(p), "LineString")) >= 8.0 for r in refs)
                if not (has_dir_arrow or has_ref):
                    continue
            # 裁剪掉伸向路侧支路的细长侧枝，让物理车行道多边形贴合主走廊
            corridor_clip = axis.buffer(max(0.75 * w_med + 1.5, 4.5), cap_style="flat")
            p_clean = p.intersection(corridor_clip).buffer(0)
            if p_clean.geom_type == "MultiPolygon":
                p_clean = max(p_clean.geoms, key=lambda g: g.area)
            if not p_clean.is_empty and p_clean.area >= 12.0:
                p = p_clean

            # 只挂与本幅行驶方向一致的箭头；反向箭头单独计数（方向证据矛盾，进评测门禁）
            want = -u if role == "approach" else u
            arr, n_opp = [], 0
            pb4c = p.buffer(0.4)
            for pt_, v_, code in arrows:
                if not pb4c.contains(Point(pt_)):
                    continue
                c_ = float(np.dot(v_, want))
                if c_ > 0.5:
                    arr.append(code)
                elif c_ < -0.5:
                    n_opp += 1
            arm_cws_tmp.append(dict(
                arm=aid,
                role=role,
                lanes=member_idx,
                conflicts=conflict_idx,
                polygon=p,
                axis=axis,
                mouth=mouth,
                n_lanes=len(member_idx),
                width_m=round(float(w_med), 2),
                est_lanes=max(len(member_idx), max(1, int(round(w_med / w)))),
                unmarked=(len(member_idx) == 0),
                arrows=arr,
                arrow_dir_conflicts=n_opp,
                role_evidence=evidence,
                bidirectional=(twoway_half == "shared"),
                d_lat=d_lat,
            ))

        arm_cws_tmp.sort(key=lambda c: -c["d_lat"])
        for cw in arm_cws_tmp:
            cid = len(cws)
            cw["id"] = cid
            for i in cw["lanes"]:
                lanes[i]["arm"] = aid
                lanes[i]["carriageway"] = cid
            for i in cw["conflicts"]:
                lanes[i]["dir_conflict"] = True
            cws.append(cw)

    # 兜底：吸收贴近物理车行道但因微小路面缺口未纳入的同向车道
    for i, ln in enumerate(lanes):
        if ln.get("role") not in ("approach", "exit") or ln.get("carriageway") is not None:
            continue
        g = ln["geom"]
        best = None
        for cw in cws:
            if cw["role"] != ln["role"]:
                continue
            t_ln = _tan(g, 0.5 * g.length)
            t_ax = _tan(cw["axis"], cw["axis"].project(g.interpolate(0.5, normalized=True)))
            if float(np.dot(t_ln, t_ax)) < 0.7:
                continue
            d = cw["polygon"].distance(g)
            if d < 1.5 * w and (best is None or d < best[0]):
                best = (d, cw)
        if best is not None:
            cw = best[1]
            ln["arm"] = cw["arm"]
            ln["carriageway"] = cw["id"]
            cw["lanes"].append(i)
            cw["n_lanes"] = len(cw["lanes"])
            cw["unmarked"] = False

    return cws


# ---------------------------------------------------------------- 6. 总装与导出

def foreign_cut(foreign_stops, ext=1.5):
    """邻口停止线两端各延 ext 米后的细带：车行道多边形在这里断开，不伸进下一个路口。"""
    cuts = []
    for s_ in foreign_stops:
        a, b = np.asarray(s_.coords[0]), np.asarray(s_.coords[-1])
        t = _unit(b - a)
        cuts.append(LineString([a - t * ext, b + t * ext]).buffer(0.25))
    return unary_union(cuts) if cuts else None


def build(lanes, core_poly, B, arrows, w, opposing=None, n_stops=99, refs=(), core_stops=(), stop_dirs=(), grow=True,
          foreign_stops=()):
    if grow and n_stops <= 2:
        core_poly = grow_free(core_poly, lanes, B["drivable"], w)
    obst = obstacles(B["drivable"], B["nondrive"], core_poly)
    blocks_u = unary_union([o["geom"] for o in obst if o["kind"] == "block" and o.get("area", 0) >= 2500.0]) if any(o["kind"] == "block" and o.get("area", 0) >= 2500.0 for o in obst) else None

    mouths = find_mouths(core_poly, B["drivable"])
    # 错位臂口门补齐（如荔湾 7913 西臂）：若某条核心停止线因错位未贴在 core_poly 上、且与核心之间不隔转角安全岛，
    # 直接将其作为口门纳入路臂发现（既不侵占其他进口道，也不丢错位臂）
    nd_clean = B["nondrive"].buffer(-0.3) if (B.get("nondrive") is not None and not B["nondrive"].is_empty) else None
    for s, d in zip(core_stops, stop_dirs):
        if d is None or s.length < 4.0 or s.distance(core_poly) <= 3.0:
            continue
        a_pt, b_pt = shapely.ops.nearest_points(s, core_poly)
        if nd_clean is not None and not nd_clean.is_empty and LineString([a_pt, b_pt]).intersection(nd_clean).length > 1.5:
            continue
        if any(m.distance(s) < 6.0 for m, _ in mouths):
            continue
        # 若该停止线两侧紧邻同向/对向车行道（有中分带隔开的出口），扩大口门锚点范围使 lane_arm 同时收到该臂进出口
        mouths.append((s, -d))

    arms = group_arms(mouths)
    for ln in lanes:
        if ln.get("role") in ("approach", "exit"):
            aid, _ = lane_arm(ln, arms, core_poly)
            ln["arm"] = aid
        else:
            ln["arm"] = None

    used = {ln["arm"] for ln in lanes if ln.get("arm") is not None}
    kept = [a for a in arms if a["id"] in used or _has_road_evidence(a, B, refs, B["drivable"])]
    moves = _arrow_moves(arrows, core_stops)
    confirmed, uncertain = _minor_exits(arms, kept, moves, B["drivable"], core_poly, refs)
    kept.extend(confirmed)
    kept.sort(key=lambda a: a["bearing"])
    arms = kept
    remap = {a["id"]: i for i, a in enumerate(arms)}
    for i, a in enumerate(arms):
        a["id"] = i
    for ln in lanes:
        if ln.get("arm") is not None:
            ln["arm"] = remap.get(ln["arm"])

    arms = pair_arms(arms, lanes, blocks_u=blocks_u, core_stops=core_stops, stop_dirs=stop_dirs)
    slips = detect_bypass_slips(obst, B["drivable"], core_poly, arms, lanes, w)
    cws = build_physical_carriageways(lanes, arms, core_poly, B["drivable"], B, obst, slips, arrows, refs, w,
                                      core_stops=core_stops, stop_dirs=stop_dirs, foreign_stops=foreign_stops)

    slips = slip_connectivity(slips, cws, core_poly, w)
    for s_ in slips:
        cp = s_.get("corridor_poly")
        s_["controlled"] = bool(cp is not None and (any(st.intersects(cp.buffer(0.5)) for st in B["stops"])
                                                    or any(c.intersects(cp) for c in B["crosswalks"])))
    for a in arms:
        a["in_carriageways"] = [c["id"] for c in cws if c["arm"] == a["id"] and c["role"] == "approach"]
        a["out_carriageways"] = [c["id"] for c in cws if c["arm"] == a["id"] and c["role"] == "exit"]

    dir_conflicts = sum(1 for ln in lanes if ln.get("dir_conflict"))
    return dict(
        foreign_stops=list(foreign_stops),
        core_poly=core_poly,
        obstacles=obst,
        arms=arms,
        candidate_mouths=uncertain,
        carriageways=cws,
        slips=slips,
        dir_conflicts=dir_conflicts,
    )


def to_json(model, to_wgs):
    """导出（所有几何统一转 WGS84 GeoJSON；分段切点同时给局部米坐标 cut_xy_local 与 WGS84 点 cut_pt）。
    车道下标（cw.lanes / sections[].lane_ids / slips[].lanes / extra_lanes）= lanes.geojson 的 lane_id。"""
    def gj(g):
        return None if g is None else mapping(to_wgs(g))

    def sec_out(sec):
        d = {k: v for k, v in sec.items() if k != "cut_xy"}
        cxy = sec.get("cut_xy")
        d["cut_xy_local"] = None if cxy is None else [float(cxy[0]), float(cxy[1])]
        d["cut_pt"] = None if cxy is None else gj(Point(cxy))
        return d

    return dict(
        status=model.get("status"), status_reason=model.get("status_reason"),
        crs_note="几何为 WGS84 经纬度；*_local 字段为本路口局部横轴墨卡托米坐标（原点=种子点）",
        core_poly=None if model.get("status") == "no_data" else gj(model["core_poly"]),
        candidate_mouths=[dict(bearing=c["bearing"], mouths=[gj(m) for m in c["mouths"]], reason=c["reason"])
                          for c in model.get("candidate_mouths", [])],
        arms=[dict(id=a["id"], bearing=round(a["bearing"], 1), mouths=[gj(m) for m in a["mouths"]],
                   minor_branch=bool(a.get("minor_branch", False)),
                   in_carriageways=a["in_carriageways"], out_carriageways=a["out_carriageways"]) for a in model["arms"]],
        carriageways=[dict(id=c["id"], arm=c["arm"], role=c["role"], lanes=c["lanes"], n_lanes=c["n_lanes"],
                           width_m=c.get("width_m"), est_lanes=c.get("est_lanes"), unmarked=bool(c.get("unmarked", False)),
                           used_frenet_dp=bool(c.get("used_frenet_dp", False)), bidirectional=bool(c.get("bidirectional", False)),
                           sections=[sec_out(sec) for sec in c.get("sections", [])],
                           arrows=c["arrows"], arrow_dir_conflicts=int(c.get("arrow_dir_conflicts", 0)),
                           role_evidence=c.get("role_evidence"),
                           polygon=gj(c["polygon"]), axis=gj(c["axis"]), frenet_axis=gj(c.get("frenet_axis")),
                           mouth=gj(c["mouth"]))
                      for c in model["carriageways"]],
        slips=[dict(id=s["id"], lane=s["lane"], lanes=s.get("lanes", []), from_arm=s["from_arm"], to_arm=s["to_arm"],
                    width_m=s.get("width_m"), source=s.get("source"), obstacle=s["obstacle"],
                    controlled=bool(s.get("controlled", False)), dead_end=bool(s.get("dead_end", False)),
                    diverge_pt=gj(Point(s["diverge_pt"])), merge_pt=gj(Point(s["merge_pt"])),
                    axis=gj(s.get("axis")), corridor_poly=gj(s.get("corridor_poly"))) for s in model["slips"]],
        extra_lanes=list(model.get("extra_lanes", [])),
        guides=[gj(g) for g in model.get("guides", [])],
        obstacles=[dict(kind=o["kind"], area=o.get("area"), geom=gj(o["geom"])) for o in model["obstacles"]],
        dir_conflicts=int(model.get("dir_conflicts", 0)),
        contract_errors=list(model.get("contract_errors", [])),
    )
