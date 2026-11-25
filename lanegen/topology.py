"""路口的结构判据（替代固定距离阈值）。

核心观察（来自数据，与路口大小无关）：
  * 路段上一定有虚线、黄线或公交线这类“路段标线”；路口内部没有。
  * 同一路口的进口停止线彼此“相向”：各自的行驶方向都指向对方所在一侧。
  * 两条停止线之间如果是一段没有路段标线、几乎全在路面内的空地，它们属于同一路口；
    两个相邻路口之间必然隔着一段画了标线的路段。
"""
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .boundaries import SEGMENT_MARKINGS


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _first_stop_hit(p, v, stops, tree, cw_tree, crosswalks, max_ray):
    """箭头射线碰到的第一条停止线；若在它之前先穿过人行横道（已越过一个路口），返回 None。"""
    ray = LineString([p, p + v * max_ray])
    best = None
    for k in tree.query(ray, predicate="intersects"):
        x = ray.intersection(stops[k])
        if x.is_empty:
            continue
        d = Point(p).distance(x)
        if best is None or d < best[0]:
            best = (d, k)
    if best is None:
        return None
    if cw_tree is not None:
        seg = LineString([p, p + v * best[0]])
        for j in cw_tree.query(seg, predicate="intersects"):
            if Point(p).distance(seg.intersection(crosswalks[j])) < best[0] - 0.5:
                return None
    return best


def stop_directions(stops, arrows, bounds, max_ray=250.0, crosswalks=()):
    """每条停止线的行驶方向（指向路口一侧的单位向量），没有证据为 None。
    证据 1：箭头沿前进方向射线碰到的第一条停止线是它 → 箭头方向。
    证据 2：停止线两侧 2–12 m 的平行线与纵向标线的交点数，多的一侧是上游。"""
    n = len(stops)
    votes = [np.zeros(2) for _ in range(n)]
    cnt = [0] * n
    if n == 0:
        return []
    tree = shapely.STRtree(stops)
    cw_tree = shapely.STRtree(list(crosswalks)) if len(crosswalks) else None
    for p, v, code in arrows:
        best = _first_stop_hit(p, v, stops, tree, cw_tree, list(crosswalks), max_ray)
        if best:
            votes[best[1]] += v / (1.0 + best[0] / 30.0)  # 近处箭头权重大
            cnt[best[1]] += 1
    longi = [l for l, t in bounds if t in ("white", "dash", "yellow", "bus", "sep")]
    ltree = shapely.STRtree(longi) if longi else None
    out = []
    for k, s in enumerate(stops):
        c = np.asarray(s.coords)
        dv = _unit(c[-1] - c[0])
        nv = np.array([-dv[1], dv[0]])
        if cnt[k] > 0 and np.dot(_unit(votes[k]), nv) != 0:
            # 箭头方向投到停止线法向上，得到严格垂直于停止线的方向
            out.append(nv if np.dot(votes[k], nv) > 0 else -nv)
            continue
        if ltree is None:
            out.append(None)
            continue
        side = []
        for sign in (1, -1):
            hits = 0
            for off in (2.0, 5.0, 8.0, 12.0):
                seg = LineString([c[0] - dv * 0.3 + nv * off * sign, c[-1] + dv * 0.3 + nv * off * sign])
                hits += len(ltree.query(seg, predicate="intersects"))
            side.append(hits)
        if side[0] == side[1]:
            out.append(None)
        else:
            up = nv if side[0] > side[1] else -nv
            out.append(-up)
    return out


class Space:
    """路口空间判据：路面、路段标线、停止线。"""

    def __init__(self, drivable, bounds, stops, nondrive=None, w=3.3):
        self.w = w
        # “在路面内”包含安全岛、导流区：它们属于路口内部
        area = drivable if nondrive is None or drivable is None else drivable.union(nondrive)
        self.drivable = area.buffer(0.5) if area is not None else None
        seg = [l for l, t in bounds if t in SEGMENT_MARKINGS]
        self.seg = unary_union(seg) if seg else None
        # 全部纵向标线（含白实线车道分隔线）：判断连线是否“顺着路段走”
        longi = [l for l, t in bounds if t in SEGMENT_MARKINGS or t == "white"]
        self.longi = unary_union(longi) if longi else None
        self.stops = stops
        self.stop_tree = shapely.STRtree(stops) if stops else None

    def clear(self, a, b, allow_stops=(), check_marks=True):
        """a→b 之间是否是“路口内部的空地”：
        离开路面的最长连续一段不超过一个车道宽（允许斜穿路口转角），不穿路段标线，不穿其他停止线。"""
        from .params import F
        seg = LineString([a, b])
        if seg.length < 0.5:
            return True
        if self.drivable is not None:
            out = seg.difference(self.drivable)
            if not out.is_empty:
                longest = max((g.length for g in getattr(out, "geoms", [out])), default=0.0)
                if longest > F["clear_out_k"] * self.w:
                    return False
        if check_marks and self.seg is not None and seg.intersects(self.seg):
            return False
        if self.stop_tree is not None:
            for k in self.stop_tree.query(seg, predicate="intersects"):
                if k not in allow_stops:
                    return False
        return True

    def along_marks(self, seg, r=2.5, min_cos=0.9, all_longi=False):
        """seg 上有多大比例的长度紧贴（r 米内）与之平行的路段标线：顺着路段走的连线比例高，路口内部为 0。
        all_longi=True 时白实线车道分隔线也算（有的路段只画白实线）。"""
        ref = self.longi if all_longi else self.seg
        if ref is None or seg.length < 1.0:
            return 0.0
        a, b = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1])
        u = _unit(b - a)
        near = seg.buffer(r).intersection(ref)
        if near.is_empty:
            return 0.0
        L = 0.0
        for g in getattr(near, "geoms", [near]):
            if g.geom_type != "LineString" or g.length < 0.5:
                continue
            if g.distance(seg) < 0.05:
                continue  # 被连线横穿的标线（斜穿中心线）不算“顺着走”
            t = _unit(np.asarray(g.coords[-1]) - np.asarray(g.coords[0]))
            if abs(np.dot(t, u)) >= min_cos:
                L += g.length
        return min(1.0, L / seg.length)

    def crosses_transverse(self, seg, max_sin=0.5):
        """seg 是否横穿（夹角 > 30°）路段标线。顺着车道走向、贴着平行标线走（弯道上会擦到）不算。"""
        if self.seg is None or not seg.intersects(self.seg):
            return False
        a, b = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1])
        u = _unit(b - a)
        x = seg.intersection(self.seg)
        for q in getattr(x, "geoms", [x]):
            if q.is_empty:
                continue
            p = q if q.geom_type == "Point" else q.representative_point()
            for l in getattr(self.seg, "geoms", [self.seg]):
                if l.distance(p) > 0.05:
                    continue
                s = l.project(p)
                t = _unit(np.asarray(l.interpolate(min(l.length, s + 1.0)).coords[0])
                          - np.asarray(l.interpolate(max(0.0, s - 1.0)).coords[0]))
                if abs(t[0] * u[1] - t[1] * u[0]) > max_sin:
                    return True
        return False

    def trace(self, p, t, target, allow_stops=(), max_d=60.0, block=None):
        """从车道端点 p 沿切向 t 射出，到达 target（路口核心）的距离；途中
        离开路面超过一个车道宽、横穿路段标线、穿过其他停止线或 block（别的路口的人行横道）则返回 None。
        出口车道的标线常在口门外 5–20 m 才开始画，用直线连到核心最近点会斜穿相邻车道的标线，
        沿车道自身走向回溯才是“这条车道从路口驶出”的判据。"""
        from .params import F
        p = np.asarray(p, float)
        ray = LineString([p, p + t * max_d])
        x = ray.intersection(target)
        if x.is_empty:
            return None
        d = min(Point(p).distance(q) for q in getattr(x, "geoms", [x]) if not q.is_empty)
        if d < 0.5:
            return d
        seg = LineString([p, p + t * d])
        if self.drivable is not None:
            out = seg.difference(self.drivable)
            if not out.is_empty and max((g.length for g in getattr(out, "geoms", [out])), default=0.0) > F["clear_out_k"] * self.w:
                return None
        if self.stop_tree is not None:
            for k in self.stop_tree.query(seg, predicate="intersects"):
                if k not in allow_stops:
                    return None
        if block is not None and seg.intersects(block):
            return None
        if self.crosses_transverse(seg):
            return None
        return d


def junction_cluster(seed, stops, dirs, space, crosswalks, search_r=150.0, front_tol=None, mouth=1.5, back_max=30.0):
    """相向 + 中间不隔着别的路口进口 + 连线在路面内 → 同一路口；返回 (核心多边形, 簇内停止线序号)。"""
    from .params import F
    front_tol = F["front_tol"] if front_tol is None else front_tol
    C = Point(seed)
    cand = [k for k, s in enumerate(stops) if s.distance(C) < search_r and s.length >= 2.0]
    if not cand:
        return C.buffer(15.0), []
    mid = {k: np.asarray(stops[k].interpolate(0.5, normalized=True).coords[0]) for k in cand}
    L = {k: stops[k].length for k in cand}

    cw_list = list(crosswalks)

    def _mouth(k):
        """口门点：停止线中点沿行驶方向前移 mouth 米。停止线后退设置（停止线与前方人行横道之间还有
        一段画着车道线的路段）时，口门取到人行横道远侧——否则口门连线必然穿过这段车道线。"""
        if dirs[k] is None:
            return mid[k]
        d = dirs[k]
        p0 = mid[k] + d * mouth
        ray = LineString([mid[k], mid[k] + d * 40.0])
        best = None
        for c in cw_list:
            if not ray.intersects(c):
                continue
            x = ray.intersection(c)
            far = max(Point(mid[k]).distance(Point(q)) for q in shapely.get_coordinates(x))
            near = min(Point(mid[k]).distance(Point(q)) for q in shapely.get_coordinates(x))
            if best is None or near < best[0]:
                best = (near, far)
        if best is None:
            return p0
        near, far = best
        if near <= 8.0:
            return mid[k] + d * (far + 0.5)
        stretch = LineString([mid[k] + d * mouth, mid[k] + d * near])
        beyond = LineString([mid[k] + d * (far + 0.5), mid[k] + d * (far + 15.0)])
        # 停止线到人行横道之间是画着车道线的路段，而人行横道之后是没有路段标线的路口内部
        # （人行横道之后仍是画线路段的，是路段中的人行横道，不是本路口）
        if space.along_marks(stretch) >= 0.3 and space.along_marks(beyond) < 0.1:
            return mid[k] + d * (far + 0.5)
        return p0

    mouths = {k: _mouth(k) for k in cand}

    def crosses_third_stop(link, a, b, end_tol=1.0):
        for u in cand:
            if u in (a, b) or not link.intersects(stops[u]):
                continue
            x = link.intersection(stops[u])
            for q in getattr(x, "geoms", [x]):
                if q.is_empty:
                    continue
                s_ = stops[u].project(q if q.geom_type == "Point" else q.representative_point())
                if end_tol < s_ < stops[u].length - end_tol:
                    return True
        return False

    def mouth_pt(k):
        return mouths[k]

    def separated(a, b):
        """a、b 之间夹着一条与 a 或 b 反向、迎面开来的停止线 → 中间隔着另一个路口的进口。"""
        ma, mb = mid[a], mid[b]
        ab = mb - ma
        n2 = float(np.dot(ab, ab))
        if n2 < 1e-6:
            return False
        for u in cand:
            if u in (a, b) or dirs[u] is None:
                continue
            t = float(np.dot(mid[u] - ma, ab)) / n2
            if not (0.05 < t < 0.95):
                continue
            r = mid[u] - ma
            lat = abs(ab[0] * r[1] - ab[1] * r[0]) / np.sqrt(n2)
            if lat > 2 * max(L[a], L[b], L[u]) + 5:  # 横向范围取自停止线长度（≈进口道宽）
                continue
            for e in (a, b):
                if dirs[e] is None:
                    continue
                # u 与 e 反向、u 朝 e 开过来、且 u 在 e 前方超过一个进口道宽
                if (np.dot(dirs[u], dirs[e]) < -0.95 and np.dot(dirs[u], mid[e] - mid[u]) > 0
                        and np.dot(dirs[e], mid[u] - mid[e]) > L[e]):
                    return True
        return False

    adj = {k: set() for k in cand}
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            da, db = dirs[a], dirs[b]
            # 相向判据：b 不能在 a 的上游太远。错位 T 口、主辅路分幅时同一路口的停止线会错开
            # 约一个进口道宽，容差取两条停止线长度（≈ 进口道宽）的一半之和；相邻路口间隔一个街区，远大于此。
            # 超过容差但在 back_max 内：两口门连线不能顺着路段标线走（那是相邻路口沿路段排开的停止线）。
            tol = max(front_tol, 0.5 * (L[a] + L[b]))
            back = max(-np.dot(mid[b] - mid[a], da) if da is not None else -1e9,
                       -np.dot(mid[a] - mid[b], db) if db is not None else -1e9)
            if back > back_max:
                continue
            if back > tol and space.along_marks(LineString([mouth_pt(a), mouth_pt(b)])) > 0.15:
                continue
            if separated(a, b):
                continue
            # 口门连线顺着纵向标线（含白实线）走了三成以上：穿的是路段，不是路口内部
            link = LineString([mouth_pt(a), mouth_pt(b)])
            # 同一进口前后串列的两条停止线（待转区停止线、非机动车前置停止线）相距不到 12 m，连线必然顺着车道线
            series = (da is not None and db is not None and np.dot(da, db) > 0.9
                      and np.linalg.norm(mid[b] - mid[a]) < 12.0)
            # 迎面相对的近距停止线（路段中的信号人行横道）：连线以小角度斜穿各条车道线，下面单独判
            facing = (da is not None and db is not None and np.dot(da, db) < -0.9
                      and np.linalg.norm(mouth_pt(b) - mouth_pt(a)) < 40.0)
            if not series and not facing and space.along_marks(link, all_longi=True) > 0.3:
                continue
            # 口门连线穿过第三条停止线的中段：从这个路口的内部钻进了别的进口道（两个相邻路口被一条斜线连起来）
            if crosses_third_stop(link, a, b):
                continue
            # 相邻路口之间一定隔着画了路段标线（虚线/黄线/公交线）的路段；路口内部没有
            ma, mb = mouth_pt(a), mouth_pt(b)
            ok = space.clear(ma, mb, allow_stops=set(cand), check_marks=True)
            if not ok and da is not None and db is not None and np.dot(da, db) < -0.9 \
                    and np.linalg.norm(mb - ma) < 40.0 and space.clear(ma, mb, allow_stops=set(cand), check_marks=False):
                # 迎面相对的两条停止线（路段中的信号人行横道、窄路口）：口门连线只横穿中心线和口门附近的车道线（≤ 4 根），
                # 且不顺着路段标线走
                seg_ = LineString([ma, mb])
                x = seg_.intersection(space.seg) if space.seg is not None else None
                nx = 0 if x is None or x.is_empty else len(getattr(x, "geoms", [x]))
                ok = nx <= 4 and space.along_marks(seg_) < 0.5
            if ok:
                adj[a].add(b)
                adj[b].add(a)
    # 所有连通簇都建核心，取核心离种子最近的一簇（种子在核心内距离为 0；并列取停止线多的）。
    # 这样种子在同一路口范围内移动不改变结果，不依赖“离种子最近的那条停止线”。
    def build_core(cl):
        geoms = [stops[k] for k in cl]
        hull0 = unary_union(geoms).convex_hull.buffer(2.0)
        cws = [c for c in crosswalks if c.intersects(hull0)]
        core = unary_union(geoms + cws).convex_hull
        if core.geom_type != "Polygon":
            core = core.buffer(8.0)
        return core

    seen_all = set()
    best = None
    for k0 in cand:
        if k0 in seen_all:
            continue
        comp = {k0}
        stack = [k0]
        while stack:
            k = stack.pop()
            for j in adj[k]:
                if j not in comp:
                    comp.add(j)
                    stack.append(j)
        seen_all |= comp
        if all(dirs[k] is None for k in comp):
            continue  # 没有任何方向证据的孤立横线（多为非停止线）
        core = build_core(sorted(comp))
        key = (round(core.distance(C), 1), -len(comp))
        if best is None or key < best[0]:
            best = (key, core, sorted(comp))
    if best is None:
        return C.buffer(15.0), []
    comp = prune_series(best[2], mid, dirs, adj, C)
    if comp != best[2]:
        return build_core(comp), comp
    return best[1], best[2]


def prune_series(comp, mid, dirs, adj, C, max_series=50.0, max_lat=12.0, far_min=80.0):
    """簇内同向、在同一条路上（横向 < 12 m）、前后相距 > 50 m 的两条停止线不属于同一路口：
    远离种子的那条属于上下游的相邻路口，去掉后取仍与离种子最近的停止线相连的部分。
    阈值取 50 m 而不是 30 m：留出集里有两级停车（预信号）进口，前后两条停止线相距约 35 m，
    两段都画着导向箭头，属于同一路口（番禺 5934）；真正并进来的相邻路口停止线在 60 m 以外（增城 12763）。
    另外远的那条须离种子（信号灯）80 m 以上才拆。"""
    comp = list(comp)
    changed = True
    while changed and len(comp) > 1:
        changed = False
        for i, a in enumerate(comp):
            for b in comp[i + 1:]:
                da, db = dirs[a], dirs[b]
                # 方向取停止线法向，斜交路口与实际行驶方向可差 30°，同向判据放到 37°（cos 0.8），沿两者平均方向量距
                if da is None or db is None or np.dot(da, db) < 0.8:
                    continue
                u = _unit(da + db)
                d = mid[b] - mid[a]
                if abs(np.dot(d, u)) > max_series and abs(u[0] * d[1] - u[1] * d[0]) < max_lat:
                    far = a if np.linalg.norm(mid[a]) > np.linalg.norm(mid[b]) else b
                    if Point(mid[far]).distance(C) < far_min:
                        continue  # 离信号灯 80 m 内的串列停止线多是多级停车的大路口（黄埔 13197 西进口三四级），不拆
                    comp.remove(far)
                    changed = True
                    break
            if changed:
                break
        if changed:
            # 重新取连通部分（从离种子最近的停止线出发）
            root = min(comp, key=lambda k: Point(mid[k]).distance(C))
            keep, stack = {root}, [root]
            while stack:
                k = stack.pop()
                for j in adj[k]:
                    if j in comp and j not in keep:
                        keep.add(j)
                        stack.append(j)
            comp = sorted(keep)
    return sorted(comp)


def belongs(ln, cluster_stops, core, space, stop_idx_all, tol_end, block=None):
    """车道是否属于本路口（结构判据，车道须已定向）：
    进口：终点贴着本路口的停止线；
    出口：起点到路口核心最近点之间是空地，或起点沿车道走向回溯能到达核心（不横穿标线、不穿其他路口）；
    无停止线的进口：终点到核心之间是空地且车道朝向核心，或终点沿走向前推能到达核心。
    block：别的路口的人行横道（回溯不能越过）。"""
    g = ln["geom"]
    e1 = Point(g.coords[-1])
    e0 = Point(g.coords[0])
    for s in cluster_stops:
        if e1.distance(s) < tol_end:
            return "approach"
    allow = set(stop_idx_all)
    p0 = np.asarray(e0.coords[0])
    t0 = _unit(p0 - np.asarray(g.interpolate(min(4.0, g.length)).coords[0]))  # 指向上游（路口）
    if core.contains(e0) or space.clear(p0, np.asarray(core.exterior.interpolate(core.exterior.project(e0)).coords[0]),
                                        allow_stops=allow):
        return "exit"
    if space.trace(p0, t0, core, allow_stops=allow, block=block) is not None:
        return "exit"
    if core.contains(e1):
        return "approach"
    p1 = np.asarray(e1.coords[0])
    q = np.asarray(core.exterior.interpolate(core.exterior.project(e1)).coords[0])
    t = _unit(p1 - np.asarray(g.interpolate(max(0, g.length - 4)).coords[0]))
    if np.dot(t, _unit(q - p1)) > 0.5 and space.clear(p1, q, allow_stops=allow):
        return "approach"
    if space.trace(p1, t, core, allow_stops=allow, block=block) is not None:
        return "approach"
    return None


def approach_arrows(arrows, stops, cluster, max_ray=250.0, crosswalks=()):
    """箭头沿前进方向碰到的第一条停止线属于本路口（且途中没穿过人行横道）→ 本路口进口箭头。
    返回 [(p, v, code, 停止线序号, 距离)]。"""
    if not stops:
        return []
    tree = shapely.STRtree(stops)
    cw_tree = shapely.STRtree(list(crosswalks)) if len(crosswalks) else None
    out = []
    cl = set(cluster)
    # 停止线两端各外延 3 m 的版本：渠化右转（free right）车道不停车，停止线只画到它左侧的车道线为止，
    # 右转箭头的射线从停止线端头外 1–2 m 擦过（开创大道×荔红二路东、西进口），整条右转车道因此被丢
    ext = {}
    for k in cl:
        c = np.asarray(stops[k].coords)
        if len(c) < 2 or stops[k].length < 1.0:
            continue
        a, b = c[0], c[-1]
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        ext[k] = LineString([a - 3.0 * t, b + 3.0 * t])
    for p, v, code in arrows:
        best = _first_stop_hit(p, v, stops, tree, cw_tree, list(crosswalks), max_ray)
        if best and best[1] in cl:
            out.append((p, v, code, best[1], best[0]))
            continue
        if best is not None:
            continue  # 射线先碰到别的路口的停止线
        ray = LineString([p, p + v * 60.0])
        cand = []
        for k, e in ext.items():
            x = ray.intersection(e)
            if x.is_empty or x.geom_type != "Point":
                continue
            d = Point(p).distance(x)
            if cw_tree is not None:
                seg = LineString([p, p + v * d])
                if any(Point(p).distance(seg.intersection(crosswalks[j])) < d - 0.5
                       for j in cw_tree.query(seg, predicate="intersects")):
                    continue
            cand.append((d, k))
        if cand:
            d, k = min(cand)
            out.append((p, v, code, k, d))
    return out


def expand_core(core, cluster, stops, dirs, lane_geoms, space, crosswalks=(), max_ray=200.0, mouth=1.5, max_reach=40.0):
    """从每条本路口停止线的路口一侧沿行驶方向发射线，碰到“横向穿过射线”的车道中心线 → 该停止线正前方
    （停止线全宽 × 到碰撞点的距离）的四边形并入核心，让 T 形路口没有停止线的主路一侧也落进路口核心。
    射线在碰到人行横道、路段标线、其他停止线或驶出路面时停止；与射线近乎平行的车道不算。
    返回 (核心, [(四边形, 方向)])：四边形只用来切断“横穿”它的车道（主路直行车道在路口处断开），
    顺着它走的车道不切（旧做法用细长射线带当刀，把进口、出口车道拦腰切出缺口）。"""
    if not lane_geoms:
        return core, []
    tree = shapely.STRtree(lane_geoms)
    # 本路口自己的人行横道（与核心相交）不算越界，只有别处的人行横道才让射线停下
    other_cw = [c for c in crosswalks if not c.intersects(core)]
    cw_u = unary_union([c.boundary for c in other_cw]) if other_cw else None
    quads = []
    for k in cluster:
        d = dirs[k]
        if d is None:
            continue
        s_ = stops[k]
        others = unary_union([stops[j] for j in range(len(stops)) if j != k])
        reach = []
        for f in (0.15, 0.5, 0.85):
            p0 = np.asarray(s_.interpolate(f, normalized=True).coords[0]) + d * (mouth + 1.0)
            ray = LineString([p0, p0 + d * max_ray])
            stop_at = max_ray
            for obst in (cw_u, space.seg, others):
                if obst is None:
                    continue
                x = ray.intersection(obst)
                for q in getattr(x, "geoms", [x]):
                    if not q.is_empty:
                        stop_at = min(stop_at, Point(p0).distance(q))
            if space.drivable is not None:
                inside = ray.intersection(space.drivable)
                first = next((g for g in getattr(inside, "geoms", [inside]) if not g.is_empty and g.distance(Point(p0)) < 0.5), None)
                if first is None:
                    continue
                stop_at = min(stop_at, first.length)
            best = None
            for j in tree.query(ray, predicate="intersects"):
                g = lane_geoms[j]
                x = ray.intersection(g)
                for q in getattr(x, "geoms", [x]):
                    if q.geom_type != "Point":
                        continue
                    dd = Point(p0).distance(q)
                    if dd >= stop_at:
                        continue
                    t = _unit(np.asarray(g.interpolate(min(g.length, g.project(q) + 1)).coords[0]) - np.asarray(
                        g.interpolate(max(0, g.project(q) - 1)).coords[0]))
                    if abs(t[0] * d[1] - t[1] * d[0]) < 0.5:  # 夹角 < 30°：与射线近乎平行
                        continue
                    if best is None or dd < best:
                        best = dd
            # 只并入到“同侧车行道”为止：T 形路口主路近侧一幅的宽度与支路进口道宽同量级（停止线长 + 12 m 封顶），
            # 再远就是伸进别的路臂了（四边形会把核心撑进路臂，出口车道被截短、丢掉“从核心驶出”的方向证据）
            if best is not None and best <= min(max_reach, s_.length + 12.0):
                reach.append(best + mouth + 1.0)
        if not reach:
            continue
        a, b = np.asarray(s_.coords[0]), np.asarray(s_.coords[-1])
        r = max(reach) + 0.5 * space.w
        quad = Polygon([a, b, b + d * r, a + d * r])
        if quad.is_valid and quad.area > 1.0:
            quads.append((quad, d))
    if not quads:
        return core, []
    grown = unary_union([core] + [q for q, _ in quads]).buffer(0.2).buffer(-0.2)
    if grown.geom_type == "MultiPolygon":
        grown = max(grown.geoms, key=lambda g: g.intersection(core).area)
    return (grown if grown.geom_type == "Polygon" else core), quads


def cut_transverse(geom, quads, max_cos=0.7):
    """把 geom 横穿四边形（与四边形方向夹角 > 45°、且从一条长边穿到另一条长边）的部分去掉：
    T 形路口主路直行车道在路口处断开。只在四边形里起止的车道（别的进口、出口）不切。"""
    rm = []
    for quad, d in quads:
        if not geom.intersects(quad):
            continue
        c = np.asarray(quad.exterior.coords)
        width = np.linalg.norm(c[1] - c[0])  # 停止线长度
        x = geom.intersection(quad)
        for q in getattr(x, "geoms", [x]):
            if q.geom_type != "LineString" or q.length < 0.7 * width:
                continue
            u = _unit(np.asarray(q.coords[-1]) - np.asarray(q.coords[0]))
            if abs(np.dot(u, d)) < max_cos:
                rm.append(q.buffer(0.05))
    if not rm:
        return geom
    return geom.difference(unary_union(rm))
