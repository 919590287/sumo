"""路口归属、方向投票、车道增减与渠化右转标注。"""
import numpy as np
import shapely
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import unary_union, substring

from .lanes import TYPES

ARROW_PREFIX = "arrow_"


def _end(line, which, h=4.0):
    """端点坐标和向外的单位切向。"""
    c = np.asarray(line.coords)
    if which == 0:
        p = c[0]
        q = np.asarray(line.interpolate(min(h, line.length)).coords[0])
    else:
        p = c[-1]
        q = np.asarray(line.interpolate(max(0.0, line.length - h)).coords[0])
    v = p - q
    n = np.linalg.norm(v)
    return p, (v / n if n > 1e-9 else v)


def stitch_lanes(lanes, blockers, P=None, space=None, max_gap=60.0, min_cos=0.985, oriented=False, marks=None):
    """车道断开处接回：出入口、标线缺损造成的断开。
    判据：两端共线（横向偏差 ≤ 0.25 W）、宽度相近（≤ 0.25 W）、接缝在路面内、不穿任何标线和其他车道、
    不跨停止线/人行横道/路口核心。max_gap 只是搜索范围。"""
    max_lat = P["stitch_lat"] if P else 0.8
    max_dw = P["stitch_dw"] if P else 0.9
    from scipy.spatial import cKDTree
    changed = True
    while changed:
        changed = False
        best = None
        E = [(i, w) + _end(l["geom"], w) for i, l in enumerate(lanes) for w in (0, 1)]
        if len(E) < 2:
            break
        tree = cKDTree(np.array([e[2] for e in E]))
        for x, y in tree.query_pairs(max_gap):
            i, wa, pa, ta = E[x]
            j, wb, pb, tb = E[y]
            if i == j:
                continue
            if oriented and wa == wb:
                continue  # 已定向的车道只能尾接头
            v = pb - pa
            g = np.linalg.norm(v)
            if g < 1e-6:
                continue
            u = v / g
            if np.dot(ta, u) < min_cos or np.dot(tb, -u) < min_cos:
                continue
            lat = abs(ta[0] * v[1] - ta[1] * v[0])
            # 横向容差按这两条车道自身的宽度算（全局众数会被窄路拉低），并随间隙略放宽：
            # 无标线段两侧车道常有 1° 左右的错位，间隙 24 m 时横向差 0.8–1 m；
            # 上限 0.45 W，远小于相邻车道的间距（≥ 0.75 W），不会接到隔壁车道上
            wp = 0.5 * (lanes[i]["width"] + lanes[j]["width"])
            lat_tol = min(max(max_lat, 0.25 * wp) + 0.03 * g, 0.45 * wp)
            if lat > lat_tol:
                continue
            if abs(lanes[i]["width"] - lanes[j]["width"]) > max(max_dw, 0.25 * wp):
                continue
            seg = LineString([pa, pb])
            if blockers is not None and seg.intersects(blockers):
                continue
            if g > 1.0:
                inner = LineString([pa + u * 0.5, pb - u * 0.5])
                if space is not None and space.drivable is not None and inner.intersection(space.drivable).length < 0.95 * inner.length:
                    continue
                if marks is not None and inner.intersects(marks):
                    continue
                if any(inner.intersects(lanes[k]["geom"]) for k in range(len(lanes)) if k not in (i, j)):
                    continue
            cost = g + 10.0 * lat
            if best is None or cost < best[0]:
                best = (cost, i, wa, j, wb) if i < j else (cost, j, wb, i, wa)
        if best:
            _, i, wa, j, wb = best
            ca = list(lanes[i]["geom"].coords)
            cb = list(lanes[j]["geom"].coords)
            if oriented:
                # 已定向：只会是 i 尾接 j 头（wa=1, wb=0）或 j 尾接 i 头（wa=0, wb=1），两段都保持原方向
                coords = ca + cb if wa == 1 else cb + ca
            else:
                if wa == 0:
                    ca = ca[::-1]
                if wb == 1:
                    cb = cb[::-1]
                coords = ca + cb
            merged = dict(lanes[i])
            merged["geom"] = LineString(coords)
            for k in ("n_norm", "n_edge", "n_anch"):
                merged[k] = lanes[i][k] + lanes[j][k]
            li, lj = lanes[i]["geom"].length, lanes[j]["geom"].length
            for k in ("frac_dash", "frac_curb", "frac_island"):
                merged[k] = (lanes[i].get(k, 0) * li + lanes[j].get(k, 0) * lj) / max(li + lj, 1e-9)
            merged["yellow_pts"] = np.vstack([lanes[i]["yellow_pts"], lanes[j]["yellow_pts"]])
            merged["has_yellow"] = lanes[i]["has_yellow"] or lanes[j]["has_yellow"]
            merged["width"] = (lanes[i]["width"] * lanes[i]["geom"].length + lanes[j]["width"] * lanes[j]["geom"].length) / (
                lanes[i]["geom"].length + lanes[j]["geom"].length)
            merged["stitched"] = lanes[i].get("stitched", 0) + lanes[j].get("stitched", 0) + 1
            lanes = [l for k, l in enumerate(lanes) if k not in (i, j)] + [merged]
            changed = True
    return lanes


def _ray_line_hit(p, t, a, b):
    """射线 p + s t 与直线 ab 的交点参数 (s, u)；u 是在 ab 上的位置（0..1 为线段内）。"""
    d = b - a
    den = t[0] * d[1] - t[1] * d[0]
    if abs(den) < 1e-9:
        return None
    w_ = a - p
    s_ = (w_[0] * d[1] - w_[1] * d[0]) / den
    u = (w_[0] * t[1] - w_[1] * t[0]) / den
    return s_, u


def extend_to_junction(lanes, stops, core, space=None, w=3.3, search=30.0, crosswalks=(), nondrive=None):
    """进口车道终点延伸到本路口停止线，出口车道起点回延到路口核心。
    与停止线所在直线求交（斜停止线也能接上），交点可落在停止线两端外延 0.5 W 内；
    不设距离阈值，改由结构判断：延伸段须是不穿标线、不穿其他车道的空路面（search 只是搜索范围）。"""
    others = [l["geom"] for l in lanes]
    for idx, ln in enumerate(lanes):
        g = ln["geom"]
        cs = list(g.coords)
        # 终点 → 停止线
        p, t = _end(g, 1)
        best = None
        for s_ in stops:
            a = np.asarray(s_.coords[0])
            b = np.asarray(s_.coords[-1])
            hit = _ray_line_hit(p, t, a, b)
            if hit is None:
                continue
            dist, u = hit
            L = np.linalg.norm(b - a)
            tol = 0.5 * w / max(L, 1e-6)
            if not (0.0 <= dist <= search and -tol <= u <= 1 + tol):
                continue
            q = p + t * dist
            if best is None or dist < best[0]:
                best = (dist, q)
        if best is not None and best[0] > 0.05:
            ok = True
            if space is not None and best[0] > 1.0:
                ok = space.clear(p + t * 0.3, best[1], allow_stops=set(range(len(space.stops))))
            # 延伸段不能横穿路口核心（穿过去碰到的是对面进口的停止线）
            if ok and LineString([p, best[1]]).intersection(core.buffer(-1.0)).length > 1.0:
                ok = False
            if ok and best[0] > 1.0:
                seg = LineString([p + t * 0.5, best[1] - t * 0.5]) if best[0] > 1.5 else None
                if seg is not None and any(seg.intersects(o) for k, o in enumerate(others) if k != idx):
                    ok = False
                # 车道延伸不能穿过安全岛、导流区
                if ok and nondrive is not None and LineString([p, best[1]]).intersection(nondrive).length > 0.3:
                    ok = False
            if ok:
                cs = cs + [tuple(best[1])]
        # 起点 → 人行横道外缘（出口）；途中没有人行横道时，沿车道走向回溯到核心边界
        # （space.trace：不横穿标线、不出路面、不穿停止线；只到核心边界为止，不伸进路口内部）
        p, t = _end(g, 0)
        ray = LineString([p + t * 0.05, p + t * search])
        pts = []
        if crosswalks:
            tgt = unary_union([c.boundary for c in crosswalks])
            x = ray.intersection(tgt)
            pts = [q for q in getattr(x, "geoms", [x]) if q.geom_type == "Point"]
        if not pts and space is not None and stops and not core.contains(Point(p)):
            d = space.trace(p, t, core, allow_stops=(), max_d=search)
            if d is not None and d > 1.0:
                q = p + t * d
                seg = LineString([p + t * 0.5, q - t * 0.5]) if d > 1.5 else None
                ok = seg is None or not any(seg.intersects(o) for k, o in enumerate(others) if k != idx)
                if ok and nondrive is not None and LineString([p, q]).intersection(nondrive).length > 0.3:
                    ok = False
                if ok:
                    cs = [tuple(q)] + cs
            ln["geom"] = LineString(cs)
            continue
        if pts and not core.contains(Point(p)):
            q = min(pts, key=lambda q: Point(p).distance(q))
            qd = Point(p).distance(q)
            ok = space is None or qd <= 1.0 or space.clear(p + t * 0.3, np.asarray(q.coords[0]),
                                                           allow_stops=set(range(len(space.stops))))
            if ok and qd > 1.5:
                seg = LineString([p + t * 0.5, np.asarray(q.coords[0]) - t * 0.5])
                ok = not any(seg.intersects(o) for k, o in enumerate(others) if k != idx)
            if ok and nondrive is not None and LineString([p, np.asarray(q.coords[0])]).intersection(nondrive).length > 0.3:
                ok = False
            if ok:
                cs = [(q.x, q.y)] + cs
        ln["geom"] = LineString(cs)
    return lanes


def junction_core(seed, stops, crosswalks, r=90.0, link=50.0):
    """停止线按 50 m 单连接聚类，取含最近停止线的一簇；加上邻近人行横道，取凸包。"""
    C = Point(seed)
    near = [s for s in stops if s.distance(C) < r and s.length >= 2.5]
    if not near:
        return C.buffer(15.0), [], C
    near.sort(key=lambda s: s.distance(C))
    cluster = [near[0]]
    rest = near[1:]
    changed = True
    while changed:
        changed = False
        for s in list(rest):
            if min(s.distance(c) for c in cluster) < link:
                cluster.append(s)
                rest.remove(s)
                changed = True
    hull0 = unary_union(cluster).convex_hull
    cws = [c for c in crosswalks if c.intersects(hull0.buffer(12.0))]
    core = unary_union(cluster + cws).convex_hull
    if core.geom_type != "Polygon":
        core = core.buffer(8.0)
    return core, cluster, core.centroid


def _tangent_at(line, s, h=1.0):
    a = line.interpolate(max(0.0, s - h))
    b = line.interpolate(min(line.length, s + h))
    v = np.array([b.x - a.x, b.y - a.y])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _bearing_vec(deg):
    r = np.radians(deg)
    return np.array([np.sin(r), np.cos(r)])


def arrows_of(main):
    a = main[main["要素类型"] == "符号轮廓"]
    a = a[a["分类代码"].astype(str).str.startswith(ARROW_PREFIX)]
    out = []
    for g, rot, code in zip(a.geometry, a["旋转角度"], a["分类代码"]):
        if rot is None or np.isnan(rot):
            continue
        c = g.centroid
        out.append((np.array([c.x, c.y]), _bearing_vec(float(rot)), code))
    return out


def movement(code):
    c = code[len(ARROW_PREFIX):]
    res = []
    for k, v in (("front_left", "l"), ("front_right", "r"), ("straight", "S"), ("left", "L"), ("right", "R"), ("uturn", "T")):
        if k in c:
            res.append(v)
            c = c.replace(k, "")
    return "".join(sorted(res))


def select_and_trim(lanes, core, keep_r=160.0, touch=30.0, stops=(), crosswalks=(), core_cut=None):
    """只保留与路口相连的车道：在停止线、人行横道处截断（不用凸包裁），
    端点距核心 ≤ touch，且截到 keep_r 内。"""
    zone = core.buffer(keep_r)
    cutter = unary_union([s.buffer(0.05) for s in stops] + [c.buffer(0.3) for c in crosswalks])
    kept = []
    for ln in lanes:
        g0 = ln["geom"].difference(core_cut) if core_cut is not None and not core_cut.is_empty else ln["geom"]
        g = g0.difference(cutter).intersection(zone) if not cutter.is_empty else g0.intersection(zone)
        parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "LineString" and p.length >= 6.0]
        for p in parts:
            e0, e1 = Point(p.coords[0]), Point(p.coords[-1])
            if min(e0.distance(core), e1.distance(core)) <= touch:
                q = dict(ln)
                q["geom"] = p
                kept.append(q)
    return kept


def orient(lanes, arrows, stops, core, P, refs, space=None, opposing=None, cluster_stops=None, stop_dirs=None):
    """方向投票：箭头(3)、停止线端点(3)、黄线在左(2)、出口端无停止线(1)、参考线(0.5)。
    出口端判据有两种：端点贴着核心；或端点沿车道走向回溯能进入核心、途中不穿停止线（出口道标线常在
    口门外十几米才开始画，端点离核心远）。进口车道的端点回溯会穿过本路口停止线，所以不会被误判。
    opposing：对向分隔（黄线、中央隔离），邻道借方向不能跨过它。"""
    stop_u = unary_union(stops) if stops else None
    cl_u = unary_union(cluster_stops) if cluster_stops else None
    for ln in lanes:
        g = ln["geom"]
        score = 0.0
        ev = []
        # 箭头
        for p, v, code in arrows:
            pt = Point(p)
            if pt.distance(g) > 0.5 * P["w_mode"] + 0.3:
                continue
            s = g.project(pt)
            t = _tangent_at(g, s)
            dt = float(np.dot(t, v))
            if dt > 0.7:
                score += 3
                ev.append("arrow+")
            elif dt < -0.7:
                score -= 3
                ev.append("arrow-")
        # 停止线：车道的下游端贴着停止线。停止线方向已知时还要看车道在它哪一侧：
        # 车身在停止线上游、端点切向与停止线方向一致才是“驶到停止线”；车身在下游的是从路口驶出的车道
        # （小路口无人行横道时，出口车道起点紧挨着对向进口的停止线端头）
        if stop_u is not None:
            d0 = Point(g.coords[0]).distance(stop_u)
            d1 = Point(g.coords[-1]).distance(stop_u)
            if min(d0, d1) < 3.0 and abs(d0 - d1) > 5:
                which = 1 if d1 < d0 else 0
                vote = 3 if which == 1 else -3
                e = Point(g.coords[which])
                k = min(range(len(stops)), key=lambda j: stops[j].distance(e))
                sp = stops[k].project(e)
                if not (0.3 < sp < stops[k].length - 0.3):
                    vote = 0  # 端点落在停止线端头之外：是挨着它的对向车道，不是驶到这条停止线
                if vote and stop_dirs is not None:
                    dk = stop_dirs[k]
                    if dk is not None:
                        p_, u_ = _end(g, which)  # u_：端点处向外的切向
                        c_ = float(np.dot(u_, dk))
                        if c_ < -0.5:
                            vote = -vote  # 车身在停止线下游：这一端是起点
                        elif c_ <= 0.5:
                            vote = 0  # 与停止线斜交，不作证据
                if vote:
                    score += vote
                    ev.append("stop")
        # 黄线在左
        yl = ln.get("yellow_pts")
        if yl is not None and len(yl):
            # 只认车道旁边的黄线样本：接回/截断后残留的远处样本（开创大道西出口在截断端外 90 m）不作证据
            yl = np.asarray(yl)
            yl = yl[shapely.distance(g, shapely.points(yl)) <= 1.5 * P["w_mode"]] if len(yl) else yl
        if yl is not None and len(yl):
            vs = []
            for y in yl[:: max(1, len(yl) // 30)]:
                s = g.project(Point(y))
                t = _tangent_at(g, s)
                c = g.interpolate(s)
                vs.append(np.sign(t[0] * (y[1] - c.y) - t[1] * (y[0] - c.x)))
            m = float(np.mean(vs))
            if abs(m) > 0.5:
                score += 2 * np.sign(m)
                ev.append("yellow")
        # 贴核心但没有停止线的一端是出口起点
        if stop_u is not None:
            e0, e1 = Point(g.coords[0]), Point(g.coords[-1])
            hits = []
            for which, e, sign in ((0, e0, +1), (1, e1, -1)):
                if e.distance(stop_u) <= 6.0:
                    continue
                if e.distance(core) < 4.0:
                    hits.append(("exit", sign))
                elif space is not None and cl_u is not None:
                    p, t = _end(g, which)
                    d = space.trace(p, t, core, allow_stops=(), max_d=40.0)
                    if d is not None:
                        hits.append(("exit_trace", sign))
            if len(hits) == 1:  # 两端都像出口（短的路口内片段）时不投票
                score += 1 * hits[0][1]
                ev.append(hits[0][0])
        # 渠化右转候选（无箭头、无标线）：右行规则下只能右转，取右转的那个方向
        if abs(score) < 1e-9 and ln.get("slip_candidate"):
            a_ = _turn_deg(g)
            if abs(a_) >= 20:
                score += 1.0 if a_ < 0 else -1.0
                ev.append("slip_right")
        # 参考线（只作兜底）
        if abs(score) < 1e-9:
            for r in refs:
                if r.distance(g.centroid) > 2.0 * P["w_mode"]:
                    continue
                s = g.length / 2
                t = _tangent_at(g, s)
                c = g.interpolate(s)
                rs = r.project(c)
                rt = _tangent_at(r, rs)
                dt = float(np.dot(t, rt))
                if abs(dt) > 0.9:
                    score += 0.5 * np.sign(dt)
                    ev.append("ref")
                    break
        ln["score"] = score
        ln["evidence"] = ev
    # 无证据的车道向相邻同向车道借方向（中间不能隔黄线）
    for _ in range(3):
        for ln in lanes:
            if ln["score"] != 0 or ln.get("has_yellow"):
                continue
            g = ln["geom"]
            c = g.interpolate(0.5, normalized=True)
            t = _tangent_at(g, g.length / 2)
            best = None
            for o in lanes:
                if o is ln or o["score"] == 0:
                    continue
                d = o["geom"].distance(c)
                if P["neighbor_lo"] < d < P["neighbor_hi"]:
                    s = o["geom"].project(c)
                    to = _tangent_at(o["geom"], s)
                    if opposing is not None and LineString([c, o["geom"].interpolate(s)]).intersects(opposing):
                        continue  # 中间隔着黄线/中央隔离：对向车道，不借
                    if abs(np.dot(t, to)) > 0.9 and (best is None or d < best[0]):
                        best = (d, np.sign(np.dot(t, to)) * np.sign(o["score"]))
            if best:
                ln["score"] = 0.5 * best[1]
                ln["evidence"].append("neighbor")
    for ln in lanes:
        if ln["score"] < 0:
            ln["geom"] = LineString(list(ln["geom"].coords)[::-1])
            if ln.get("widths") is not None:
                ln["widths"] = ln["widths"][::-1]
        ln["direction"] = "known" if ln["score"] != 0 else "unknown"
    return lanes


def _turn_deg(g):
    """首 3 m 与末 3 m 走向的有符号夹角（度）：正为左转，负为右转。"""
    c = np.asarray(g.coords)
    if len(c) < 4:
        return 0.0
    v0 = c[min(3, len(c) - 1)] - c[0]
    v1 = c[-1] - c[max(0, len(c) - 4)]
    return float(np.degrees(np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], np.dot(v0, v1))))


def _section_width(p, t, bounds, reach=25.0):
    """p 处垂直于 t 的车行道横断面宽度：两侧到第一条路缘/隔离/黄线的距离之和；任一侧没碰到返回 None。"""
    n = np.array([-t[1], t[0]])
    d = []
    for sgn in (1, -1):
        ray = LineString([p, p + sgn * n * reach])
        x = ray.intersection(bounds)
        if x.is_empty:
            return None
        d.append(min(Point(p).distance(q) for q in getattr(x, "geoms", [x])))
    return d[0] + d[1]


def label_events(lanes, core, stops, cluster, arrows, P, space, crosswalks=(), bounds=None):
    """端点类型（结构判据）：
    终点：stopline 贴本路口停止线 / other_junction 贴其他停止线或人行横道 / cut ROI 截断 / lane_drop / open
    起点：junction 与核心之间是空地 / other_junction / cut / lane_add / open"""
    cl_stops = [stops[k] for k in cluster]
    cl_u = unary_union(cl_stops) if cl_stops else None
    oth = [s_ for k, s_ in enumerate(stops) if k not in set(cluster)] + [c for c in crosswalks if not c.intersects(core)]
    oth_u = unary_union([shapely.make_valid(g) for g in oth]) if oth else None  # 人行横道面偶有自交
    far = core.buffer(P["keep_r"] - 3)
    tol = 0.5 * P["w_mode"]
    cwb = [l for l, t in (bounds or []) if t in ("edge", "barrier", "yellow", "sep")]
    cw_bounds = unary_union(cwb) if cwb else None
    allow = set(range(len(stops)))
    for ln in lanes:
        g = ln["geom"]
        types = []
        role = ln.get("role")
        for k, e in enumerate((Point(g.coords[0]), Point(g.coords[-1]))):
            ep = np.asarray(e.coords[0])
            if k == 1 and cl_u is not None and e.distance(cl_u) < tol + 1.0:
                types.append("stopline")
                continue
            # 只有朝向路口的一端才可能是“路口”：进口道的起点（远端）、出口道的终点（远端）离路口而去，
            # 不能因为它和核心之间是空地就记成 junction（否则早断的远端被洗成“已归类”）
            far_end = (role == "approach" and k == 0) or (role == "exit" and k == 1)
            q = np.asarray(core.exterior.interpolate(core.exterior.project(e)).coords[0])
            p_, t_ = _end(g, k)
            if (not far_end) and (core.contains(e) or space.clear(ep, q, allow_stops=allow)
                                  or space.trace(p_, t_, core, allow_stops=allow, max_d=40.0) is not None):
                types.append("junction")
            elif oth_u is not None and e.distance(oth_u) < tol + 1.0:
                types.append("other_junction")
            elif (not far.contains(e)
                  or (far_end and ln.get("far_cut"))):
                # 截断只认真实几何：ROI 保留半径外，或本车道远端确实到了 Step 2 车行道建模范围的尽头（frenet 按轴线实测给出）
                types.append("cut")
            else:
                near = any(P["neighbor_lo"] < o["geom"].distance(e) < P["neighbor_hi"] for o in lanes if o is not ln)
                # 真的展宽/收窄：车行道横断面在端点前后变宽/变窄至少 0.6 个车道；
                # 标线缺损、箭头遮挡造成的断点前后断面宽度不变，记为 open（断裂），不报增减
                real = False
                if near and cw_bounds is not None:
                    p_, t_ = _end(g, k)
                    t_ = -t_ if k == 0 else t_  # 行驶方向
                    wb = _section_width(p_ - t_ * 10.0, t_, cw_bounds)
                    wa = _section_width(p_ + t_ * 10.0, t_, cw_bounds)
                    if wb is not None and wa is not None:
                        real = (wa - wb if k == 0 else wb - wa) >= 0.6 * ln["width"]
                types.append(("lane_add" if k == 0 else "lane_drop") if real else "open")
        ln["start"], ln["end"] = types
        if ln.get("dp_add"):
            ln["start"] = "lane_add"
        if ln.get("dp_drop"):
            ln["end"] = "lane_drop"
        if ln.get("role") == "slip":
            # 渠化右转两端：上游从进口道分出（分流点），下游并入出口道（汇入点）
            ln["start"] = "diverge" if ln["start"] in ("open", "lane_add", "junction") else ln["start"]
            ln["end"] = "merge" if ln["end"] in ("open", "lane_drop", "junction") else ln["end"]
        mv = []
        for p, v, code in arrows:
            pt = Point(p)
            if pt.distance(g) < 0.5 * P["w_mode"] + 0.3:
                s_ = g.project(pt)
                if float(np.dot(_tangent_at(g, s_), v)) > 0.7:
                    mv.append((round(g.length - s_, 1), movement(code)))
        ln["arrows"] = sorted(mv)
        a = _turn_deg(g)
        ln["turn_deg"] = float(a)
        # 渠化右转：不能含左转/直行/掉头箭头，也不能止于主进口停止线（贴中央导流区的左转微弯道常有 frac_island>=0.3）
        non_right = any(bool(set(m) - {"R", "r"}) for _, m in mv)
        island_side = ln.get("frac_island", 0.0) >= 0.3
        is_dp_main = ln.get("source") == "frenet_dp" and ln.get("role") in ("approach", "exit")
        ln["slip"] = bool(
            not non_right
            and ln["end"] != "stopline"
            and not is_dp_main
            and (
                (a < -45 and ln["start"] != "cut")
                or (island_side and a < -20 and g.length < 150 and g.distance(core) < 25.0)
                or (ln.get("slip_candidate") and a < -20)
                or ln.get("role") == "slip"
            )
        )
        # 疑似非机动车道：无箭头、全程不出现虚线、80% 以上样本贴路缘/路侧护栏。只标记不删除（短路段上贴路缘的机动车道也可能如此）
        # 非机动车道在路缘和白实线之间：一侧是标线。两侧都是路缘、一侧贴岛的是渠化右转道，不是非机动车道。
        ln["nonmotor_suspect"] = bool(not mv and ln.get("frac_dash", 1.0) < 0.05 and ln.get("frac_curb", 0.0) > 0.8
                                      and "arrow_inferred" not in ln.get("evidence", [])
                                      and not ln["slip"] and not island_side and ln.get("allow") != "bus")
    return lanes


def _parallel_ref(p, v, bounds, r=16.0, max_sin=0.17):
    """箭头附近最近的、与箭头方向平行的边界线，返回 (线, 带符号偏移)。"""
    pt = Point(p)
    best = None
    for l, t in bounds:
        if t not in ("edge", "barrier", "white", "yellow", "dash") or l.length < 15:
            continue
        if t == "edge" and l.is_closed and l.length < 150:
            continue  # 安全岛、导流区的小闭合边界：岛头弯曲，不作方向参考
        d = l.distance(pt)
        if d > r or d < 0.5:
            continue
        s_ = l.project(pt)
        tt = _tangent_at(l, s_, 6.0)  # 取 ±6 m 平均切向，抑制局部抖动
        if abs(tt[0] * v[1] - tt[1] * v[0]) > max_sin:
            continue
        q = l.interpolate(s_)
        side = np.sign(tt[0] * (p[1] - q.y) - tt[1] * (p[0] - q.x))  # +1 在线的左侧
        # 优先：离得近、覆盖箭头上下游范围长
        cover = min(s_, l.length - s_)
        key = d - 0.05 * min(cover, 40)
        if best is None or key < best[0]:
            best = (key, l, side * d, tt)
    return best


def _parallel_lane(ref, p_dn, p_up, v, up_ext, dn_ext, arrows_xy, tol):
    """沿箭头旁的平行边界按箭头的横向偏移生成车道线（弯道进口不再是直线）。
    返回沿行驶方向、从最上游箭头再往上 up_ext 到最下游箭头往下 dn_ext 的折线；
    曲线偏离任一箭头超过 tol 或边界不够长时返回 None。"""
    if ref is None:
        return None
    _, l, off, _ = ref
    s0 = l.project(Point(p_dn))
    span = float(np.dot(p_dn - p_up, v)) + up_ext + dn_ext + 10.0
    sub = substring(l, max(0.0, s0 - span), min(l.length, s0 + span))
    if sub.length < 5:
        return None
    try:
        oc = sub.offset_curve(off)
    except Exception:
        return None
    if oc.is_empty or oc.geom_type != "LineString" or oc.length < 5:
        return None
    if np.dot(_tangent_at(oc, oc.project(Point(p_dn))), v) < 0:
        oc = LineString(list(oc.coords)[::-1])
    a_dn, a_up = oc.project(Point(p_dn)), oc.project(Point(p_up))
    lo, hi = a_up - up_ext, a_dn + dn_ext
    line = substring(oc, max(0.0, lo), min(oc.length, hi))
    if line.geom_type != "LineString" or line.length < 2:
        return None
    # 边界在需要的范围内提前结束：沿端点切向直线延长补足（最多 30 m，更长就不如直接用直线）
    cs = list(line.coords)
    if hi > oc.length:
        ext = hi - oc.length
        if ext > 30.0:
            return None
        p, t = _end(line, 1)
        cs = cs + [tuple(p + t * ext)]
    if lo < 0.0:
        ext = -lo
        if ext > 30.0:
            return None
        p, t = _end(line, 0)
        cs = [tuple(p + t * ext)] + cs
    line = LineString(cs)
    if any(line.distance(Point(q)) > tol for q in arrows_xy):
        return None
    return line


def arrow_lanes(lanes, app_arrows, stops, drivable, core, P, bounds=(), crosswalks=()):
    """无分隔线的宽进口：本路口进口箭头（topology.approach_arrows）中没有被车道覆盖的，按车道成组生成中心线。
    方向取最近平行边界；成组横向容差 0.35 W；
    上游延伸 = 本路口“箭头到停止线距离”的中位数（数据）；下游截到停止线或人行横道。"""
    if drivable is None or not app_arrows:
        return lanes
    w = P["w_mode"]
    covered = unary_union([l["geom"] for l in lanes]) if lanes else None
    dists = [d for *_, d in app_arrows]
    up_ext = float(np.clip(np.median(dists), 10.0, 60.0)) if dists else 20.0
    P["arrow_up_ext"] = up_ext
    unc = []
    for p, v, code, k, dist in app_arrows:
        pt = Point(p)
        if not drivable.buffer(0.5).contains(pt):
            continue
        if covered is not None and pt.distance(covered) < 0.5 * w:
            continue
        ref = _parallel_ref(p, v, bounds)
        if ref is not None:
            tt = ref[3] if np.dot(ref[3], v) > 0 else -ref[3]
            unc.append((p, tt, code, k, dist, ref))
        else:
            unc.append((p, v, code, k, dist, None))
    groups = []
    for p, v, code, k, dist, ref in unc:
        for g in groups:
            q, u = g["p"][0], g["v"]
            d = p - q
            if g["k"] == k and np.dot(u, v) > 0.95 and abs(u[0] * d[1] - u[1] * d[0]) < P["arrow_group_lat"]:
                g["p"].append(p)
                g["d"].append(dist)
                g["ref"].append(ref)
                break
        else:
            groups.append(dict(p=[p], v=v, k=k, d=[dist], ref=[ref]))
    new = []
    for g in groups:
        v = g["v"]
        Pp = np.array(g["p"])
        s_ = Pp @ v
        i_dn = int(np.argmax(s_))
        p_dn, p_up = Pp[i_dn], Pp[np.argmin(s_)]
        d_dn = float(np.min(g["d"]))  # 最下游箭头到其停止线的射线距离
        line = _parallel_lane(g["ref"][i_dn], p_dn, p_up, v, up_ext, d_dn + 0.5, Pp, 0.5 * w)
        if line is None:  # 没有可跟随的平行边界：直线
            line = LineString([p_up - v * up_ext, p_dn + v * (d_dn + 0.5)])
        a_dn = line.project(Point(p_dn))
        cut_at = []
        x = line.intersection(stops[g["k"]])
        cut_at += [line.project(q) for q in getattr(x, "geoms", [x]) if q.geom_type == "Point"]
        for c in crosswalks:
            x = line.intersection(c.boundary)
            cut_at += [line.project(q) for q in getattr(x, "geoms", [x]) if q.geom_type == "Point"]
        cut_at = [a for a in cut_at if a > a_dn - 0.5]
        if cut_at:
            line = substring(line, 0.0, min(cut_at))
        part = line.intersection(drivable.buffer(0.3))
        if covered is not None:
            # 与已有车道平行重叠或交叉的部分都扣掉（离已有车道不足 0.45 W 即视为同一条车道）
            part = part.difference(covered.buffer(0.45 * w))
        parts = [q for q in getattr(part, "geoms", [part]) if q.geom_type == "LineString"]
        parts = [q for q in parts if q.distance(Point(p_dn)) < 0.8]
        if not parts or parts[0].length < 2 * w:
            continue
        new.append(dict(geom=parts[0], width=w, n_norm=0, n_edge=0, n_anch=0, yellow_pts=np.zeros((0, 2)),
                        has_yellow=False, score=3.0, evidence=["arrow_inferred"], direction="known",
                        arrow_inferred=True))
    return lanes + new


def _evidence_weight(ln):
    return ln.get("n_norm", 0) + ln.get("n_edge", 0) + 10 * len(ln.get("arrows", [])) + abs(ln.get("score", 0)) * 5


def _absorb(long_g, short_g, w):
    """把 short 超出 long 两端、且横向偏差 < 0.6 W 的部分接到 long 上（两者同向）。"""
    cs = list(long_g.coords)
    L = long_g.length
    s0 = long_g.project(Point(short_g.coords[0]))
    s1 = long_g.project(Point(short_g.coords[-1]))
    # 尾部延长：short 终点投影到 long 末端之外
    e = Point(short_g.coords[-1])
    if s1 >= L - 1e-6 and e.distance(Point(cs[-1])) > 0.5:
        a = short_g.project(Point(cs[-1]))
        tail = substring(short_g, a, short_g.length)
        if tail.length > 0.5 and tail.distance(Point(cs[-1])) < 0.6 * w:
            cs = cs + list(tail.coords)[1:]
    # 头部延长
    b = Point(short_g.coords[0])
    if s0 <= 1e-6 and b.distance(Point(cs[0])) > 0.5:
        a = short_g.project(Point(cs[0]))
        head = substring(short_g, 0.0, a)
        if head.length > 0.5 and head.distance(Point(cs[0])) < 0.6 * w:
            cs = list(head.coords)[:-1] + cs
    return LineString(cs)


def dedupe(lanes, w):
    """两条车道中心线不可能相距 < 0.4 W（真实车道中心至少相隔一个车道下限，与行驶方向无关）。
    每一对：较短一条有一半以上落在较长一条的 0.4 W 缓冲内 → 删掉证据少的那条；
    重叠不到一半但超过 2 W → 把证据少的那条的重叠段裁掉，保留其余部分。"""
    changed = True
    rounds = 0
    while changed and rounds < 5:
        changed = False
        rounds += 1
        n = len(lanes)
        keep = [True] * n
        G = [l["geom"] for l in lanes]
        tree = shapely.STRtree(G)
        for i in range(n):
            for k in tree.query(G[i].buffer(0.4 * w), predicate="intersects"):
                k = int(k)
                if k <= i or not keep[i] or not keep[k]:
                    continue
                short, long_ = (k, i) if G[k].length <= G[i].length else (i, k)
                x = G[short].intersection(G[long_].buffer(0.4 * w))
                if x.is_empty:
                    continue
                weak = short if _evidence_weight(lanes[short]) <= _evidence_weight(lanes[long_]) else long_
                strong = long_ if weak == short else short
                if x.length >= 0.5 * G[short].length:
                    c = x.centroid
                    same = np.dot(_tangent_at(G[short], G[short].project(c)),
                                  _tangent_at(G[long_], G[long_].project(c))) > 0.5
                    # 同一条车道被识别了两次：几何以长的为准，方向以方向证据更强（|方向分|大）的为准，
                    # 短的超出长车道两端的部分接上去（例如接到停止线的那一段）
                    gs, gl = G[short], G[long_]
                    if not same:
                        if abs(lanes[short].get("score", 0)) > abs(lanes[long_].get("score", 0)):
                            gl = LineString(list(gl.coords)[::-1])
                            lanes[long_]["score"] = lanes[short].get("score", 0)
                            lanes[long_]["role"] = lanes[short].get("role", lanes[long_].get("role"))
                        else:
                            gs = LineString(list(gs.coords)[::-1])
                    lanes[long_]["geom"] = _absorb(gl, gs, w)
                    G[long_] = lanes[long_]["geom"]
                    lanes[long_]["evidence"] = list(dict.fromkeys(lanes[long_]["evidence"] + lanes[short]["evidence"]))
                    keep[short] = False
                    changed = True
                elif x.length > 2 * w:
                    rest = G[weak].difference(G[strong].buffer(0.4 * w))
                    parts = [q for q in getattr(rest, "geoms", [rest]) if q.geom_type == "LineString" and q.length >= 2 * w]
                    if parts:
                        lanes[weak]["geom"] = max(parts, key=lambda q: q.length)
                    else:
                        keep[weak] = False
                    changed = True
        lanes = [l for l, kp in zip(lanes, keep) if kp]
    return lanes


def direction_from_stops(lanes, stops, dirs_list, w):
    """仍未定向的车道（多为路口内渠化片段）：离本路口某条停止线下游口门更近的一端是起点。
    车辆只能从停止线驶出后进入这些片段。只看 3 W 以内的停止线，按距离加权投票。"""
    for ln in lanes:
        if ln["direction"] != "unknown":
            continue
        g = ln["geom"]
        e0, e1 = np.asarray(g.coords[0]), np.asarray(g.coords[-1])
        vote = 0.0
        for s_, d in zip(stops, dirs_list):
            if d is None:
                continue
            m = np.asarray(s_.interpolate(0.5, normalized=True).coords[0])
            d0, d1 = s_.distance(Point(e0)), s_.distance(Point(e1))
            near, other = (e0, e1) if d0 < d1 else (e1, e0)
            dn = min(d0, d1)
            if dn > 3 * w or np.dot(near - m, d) < -1.0:  # 太远，或在停止线上游一侧
                continue
            vote += (1.0 if d0 < d1 else -1.0) / (1.0 + dn)
        if abs(vote) > 1e-6:
            if vote < 0:
                ln["geom"] = LineString(list(g.coords)[::-1])
            ln["score"] = 0.5
            ln["direction"] = "known"
            ln["evidence"].append("from_stop")
    return lanes


def propagate_direction(lanes, w, max_iter=5):
    """未定向车道：一端与已定向车道端点相接且走向连续 → 继承方向。"""
    for _ in range(max_iter):
        changed = False
        for ln in lanes:
            if ln["direction"] != "unknown":
                continue
            g = ln["geom"]
            p0, t0 = _end(g, 0)
            p1, t1 = _end(g, 1)
            vote = 0
            for o in lanes:
                if o is ln or o["direction"] == "unknown":
                    continue
                q0, u0 = _end(o["geom"], 0)  # o 的起点（向外切向指向上游）
                q1, u1 = _end(o["geom"], 1)  # o 的终点（向外切向指向下游）
                # o 终点接我某一端：o 流出方向 = 我该端向内方向
                for pe, te, sign in ((p0, t0, +1), (p1, t1, -1)):
                    if np.linalg.norm(pe - q1) < 0.5 * w and np.dot(u1, -te) > 0.87:
                        vote += sign  # 接在我起点 → 我正向；接在我终点 → 我反向
                    if np.linalg.norm(pe - q0) < 0.5 * w and np.dot(u0, -te) > 0.87:
                        vote -= sign  # o 起点接在我起点 → 我应反向；接在我终点 → 我正向
            if vote != 0:
                if vote < 0:
                    ln["geom"] = LineString(list(g.coords)[::-1])
                ln["score"] = 0.5 if vote > 0 else 0.5
                ln["direction"] = "known"
                ln["evidence"].append("topology")
                changed = True
        if not changed:
            break
    return lanes
