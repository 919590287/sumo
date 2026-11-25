"""每条车行道的参数化横断面拟合。

一条“道路”= 一条路臂（一个连通车行道）：可能是双向（有黄线/隔离栏/内部中分空隙）或单向（被实体中分带
与另一幅分开）。方向组成先由本幅证据判定，判不了再按右侧通行看相邻平行车行道。

坐标：s 沿轴线从口门向外，u 为轴线左侧距离（面向外看）。右侧通行：
  进口（驶向路口）车道在中线 u 较大一侧，从中线往外编号 k=0（最左）…；
  出口（驶离路口）车道在中线 u 较小一侧，同样从中线往外 k=0（最左）。
每个断面、每侧的模型：锚点 a（中线边缘）、车道数 n、车道宽 w → 合成边界 b_k = a + σ·k·w。
打分 = 合成边界与观测标线/路缘的匹配残差 + 该有却没有的分界线 + 解释不了的标线 + 解释不了的路面宽度
       + 箭头压线 + 宽度先验。沿程 DP 选 (n, w)，变化要付出代价。
"""
import numpy as np
import shapely
from shapely.geometry import LineString, Point

from lanegen.junction import movement
from .legs import extend_inward

TAU = 0.5            # 匹配容差 m
W_GRID = np.round(np.arange(2.6, 4.01, 0.1), 2)
N_MAX = 6
C_DIV, C_EXTRA, C_LEFT, C_OVER, C_ARR, C_STRIP = 1.0, 1.2, 1.1, 6.0, 1.5, 0.4
C_CHG, C_CHG_UP, C_WSTEP = 5.0, 3.0, 0.8
W_PRIOR = (3.25, 0.4, 0.3)
DS = 2.0
S_LEN = 70.0
DIV_TYPES = ("dash", "white", "bus")
MED_TYPES = ("yellow", "sep")


def _frame(line, s):
    s = float(np.clip(s, 0, line.length))
    p = np.asarray(line.interpolate(s).coords[0])
    a = np.asarray(line.interpolate(max(0.0, s - 1.5)).coords[0])
    b = np.asarray(line.interpolate(min(line.length, s + 1.5)).coords[0])
    t = b - a
    t = t / (np.linalg.norm(t) + 1e-9)
    return p, t, np.array([-t[1], t[0]])


def _tangent_at(line, pt):
    s = line.project(pt)
    a = np.asarray(line.interpolate(max(0.0, s - 0.6)).coords[0])
    b = np.asarray(line.interpolate(min(line.length, s + 0.6)).coords[0])
    t = b - a
    return t / (np.linalg.norm(t) + 1e-9)


class Evidence:
    """路口窗口内的证据（局部米制坐标）。可行驶面扣掉隔离设施：护栏两侧是两幅路。"""

    def __init__(self, B, arrows, walls):
        self.B = B
        self.D = B["drivable"]
        self.Dcut = B["drivable"].difference(walls) if walls is not None else B["drivable"]
        self.bounds = [(l, t) for l, t in B["bounds"] if t != "edge"]
        self.tree = shapely.STRtree([l for l, _ in self.bounds])
        self.stops = B["stops"]
        self.cross = B["crosswalks"]
        self.arrows = arrows  # [(xy, heading_vec, code)]
        self.nondrive = B["nondrive"]
        self.walls = walls


def _in_cells(cells, grid, p, r=2):
    xs, ys = grid["xs"], grid["ys"]
    c = int(np.clip(np.searchsorted(xs, p[0]) - 1, 0, len(xs) - 1))
    q = int(np.clip(np.searchsorted(ys, p[1]) - 1, 0, len(ys) - 1))
    return bool(cells[max(0, q - r):q + r + 1, max(0, c - r):c + r + 1].any())


def make_roads(legs, center):
    roads = []
    for L in legs:
        ax, sm = extend_inward(L["axis"], center)
        roads.append(dict(leg=L, axis=ax, s_mouth=sm, cells=L["cells"]))
    for i, r in enumerate(roads):
        others = np.zeros_like(r["cells"])
        for j, q in enumerate(roads):
            if j != i:
                others |= q["cells"]
        r["others"] = others
        # 有平行的另一幅（中分带另一侧）→ 本幅是单向车行道，不再并入旁边空隙外的路面（那是辅道/支路）
        p, t, nrm = _frame(r["axis"], r["s_mouth"] + 12)
        r["has_twin"] = False
        for j, q in enumerate(roads):
            if j == i:
                continue
            pq, tq, _ = _frame(q["axis"], q["axis"].project(Point(p)))
            d = pq - p
            if np.dot(t, tq) >= np.cos(np.radians(30)) and abs(d @ t) <= 25 and 2 <= abs(d @ nrm) <= 35:
                r["has_twin"] = True
    return roads


# ------------------------------------------------------------------ 断面观测

def section(road, ev, grid, s, half=26.0):
    """断面观测。区间：轴线所在的那段（可行驶面扣掉护栏后）；若区间内没有黄线/隔离栏，
    再并入紧邻（间隙 ≤ 16 m）、属于本路臂且不属于其他路臂的一段（同一路臂内部的中分空隙）。"""
    ax = road["axis"]
    p, t, nrm = _frame(ax, s)
    seg = LineString([p - nrm * half, p + nrm * half])
    inter = seg.intersection(ev.Dcut)
    ivs = []
    for g in getattr(inter, "geoms", [inter]):
        if g.geom_type != "LineString" or g.length < 0.3:
            continue
        u = np.sort(((np.asarray(g.coords) - p) @ nrm))
        ivs.append((float(u[0]), float(u[-1])))
    if not ivs:
        return None
    ivs.sort()
    d = [0.0 if a - 0.3 <= 0 <= b + 0.3 else min(abs(a), abs(b)) for a, b in ivs]
    k0 = int(np.argmin(d))
    if d[k0] > 3.0:
        return None
    obs_u, obs_t = [], []
    for k in ev.tree.query(seg):
        line, typ = ev.bounds[k]
        x = seg.intersection(line)
        for q in getattr(x, "geoms", [x]):
            if q.is_empty or q.geom_type != "Point":
                continue
            if abs(np.dot(_tangent_at(line, q), t)) < 0.85:  # 只要纵向线
                continue
            obs_u.append(float((np.asarray(q.coords[0]) - p) @ nrm))
            obs_t.append(typ)
    obs_u, obs_t = np.array(obs_u), np.array(obs_t, dtype=object)
    ks = [k0]
    a, b = ivs[k0]
    has_med = ((obs_u > a + 0.3) & (obs_u < b - 0.3) & np.isin(obs_t, MED_TYPES)).any()
    if not has_med and not road["has_twin"]:
        nb = []
        for j, gap in ((k0 + 1, ivs[k0 + 1][0] - b if k0 + 1 < len(ivs) else None),
                       (k0 - 1, a - ivs[k0 - 1][1] if k0 >= 1 else None)):
            if gap is None or gap > 16.0 or gap < 1.2:
                continue
            lo2, hi2 = ivs[j]
            if hi2 - lo2 < 3.0:
                continue
            mid = p + nrm * (lo2 + hi2) / 2
            if _in_cells(road["cells"], grid, mid, 0) and not _in_cells(road["others"], grid, mid, 1):
                nb.append((gap, j))
        if nb:
            ks.append(min(nb)[1])
    ks.sort()
    keep = [ivs[k] for k in range(ks[0], ks[-1] + 1)]
    lo, hi = keep[0][0], keep[-1][1]
    holes = [(keep[i][1], keep[i + 1][0]) for i in range(len(keep) - 1) if keep[i + 1][0] - keep[i][1] >= 1.2]
    m = (obs_u >= lo - 0.6) & (obs_u <= hi + 0.6)
    obs_u, obs_t = list(obs_u[m]), list(obs_t[m])
    for a2, b2 in holes:
        obs_u += [a2, b2]
        obs_t += ["hole", "hole"]
    return dict(s=s, p=p, t=t, n=nrm, lo=lo, hi=hi, holes=holes,
                u=np.array(obs_u), typ=np.array(obs_t, dtype=object))


def _median_track(secs, i_from):
    """中线：先定主导来源（黄线 > 隔离栏 > 内部空隙，至少 20% 断面出现），再逐断面取与前一断面最近的候选。
    返回 (a_out, a_in, 来源) 数组或 None。"""
    cands = []
    for i, sc in enumerate(secs):
        c = {}
        if i >= i_from:
            u, ty = sc["u"], sc["typ"]
            for kind in MED_TYPES:
                m = np.isin(ty, (kind,)) & (u > sc["lo"] + 0.5) & (u < sc["hi"] - 0.5)
                if m.any():
                    uu = np.sort(u[m])
                    groups = [[uu[0]]]
                    for v in uu[1:]:
                        if v - groups[-1][-1] <= 0.6:
                            groups[-1].append(v)
                        else:
                            groups.append([v])
                    c[kind] = [(min(g), max(g)) for g in groups]
            if sc["holes"]:
                c["hole"] = list(sc["holes"])
        cands.append(c)
    n = max(1, len(secs) - i_from)
    src = None
    for kind in ("yellow", "sep", "hole"):
        if sum(1 for c in cands if kind in c) >= 0.2 * n:
            src = kind
            break
    if src is None:
        return None
    idx = [i for i, c in enumerate(cands) if src in c]
    # 起点：出现次数最多的位置附近
    allc = [((a + b) / 2) for i in idx for a, b in cands[i][src]]
    ref = float(np.median(allc))
    a_out, a_in, have = np.zeros(len(secs)), np.zeros(len(secs)), np.zeros(len(secs), bool)
    for i in idx:
        a, b = min(cands[i][src], key=lambda g: abs((g[0] + g[1]) / 2 - ref))
        if abs((a + b) / 2 - ref) > 4.0:
            continue
        a_out[i], a_in[i], have[i] = a, b, True
        ref = 0.7 * ref + 0.3 * (a + b) / 2
    hi = np.nonzero(have)[0]
    if len(hi) < 2:
        return None
    x = np.arange(len(secs))
    return np.interp(x, hi, a_out[hi]), np.interp(x, hi, a_in[hi]), src


def _stop_info(road, ev, grid, s_max=60.0):
    """与本车行道相交的停止线：横跨轴线方向、与该处车行道区间重叠。取最长的一条（渠化右转道的短停止线
    也可能横在轴线附近）。返回 (s, u0, u1, 覆盖率)。"""
    ax = road["axis"]
    best = None
    for st in ev.stops:
        if st.length < 2.0:
            continue
        s = ax.project(st.centroid)
        if s > s_max:
            continue
        p, t, nrm = _frame(ax, s)
        cs = np.asarray(st.coords)
        dirv = cs[-1] - cs[0]
        dirv = dirv / (np.linalg.norm(dirv) + 1e-9)
        if abs(np.dot(dirv, t)) > 0.4:
            continue
        uu = (cs - p) @ nrm
        if np.abs((cs - p) @ t).max() > 4.0:
            continue
        sec = section(road, ev, grid, max(0.0, s + 1.5))
        if sec is None:
            continue
        ov = min(uu.max(), sec["hi"]) - max(uu.min(), sec["lo"])
        if ov < 0.5 * (uu.max() - uu.min()) or ov < 1.5:
            continue
        cov = ov / max(1e-6, sec["hi"] - sec["lo"])
        cand = (float(s), float(uu.min()), float(uu.max()), float(cov), float(st.length))
        if best is None or cand[4] > best[4]:
            best = cand
    return best[:4] if best else None


def _crosswalk_s(road, ev, s_max=55.0):
    ax = road["axis"]
    out = None
    for cw in ev.cross:
        if cw.distance(ax) > 0.5:
            continue
        x = ax.intersection(cw)
        if x.is_empty:
            continue
        ss = [ax.project(Point(c)) for g in getattr(x, "geoms", [x]) for c in g.coords]
        if min(ss) > s_max:
            continue
        v = max(ss)
        out = v if out is None else max(out, v)
    return out


# ------------------------------------------------------------------ 打分

def side_costs(sec, a, sigma, arrows_here):
    """返回 C[n, wi]（n=0..N_MAX）。sigma=+1 进口侧（u 增大方向），-1 出口侧。"""
    lim = sec["hi"] if sigma > 0 else sec["lo"]
    avail = sigma * (lim - a)
    u, ty = sec["u"], sec["typ"]
    rel = sigma * (u - a)                    # 观测在本侧的“离中线距离”
    side = rel > 0.25
    div_u = rel[side & np.isin(ty, DIV_TYPES)]
    dash_u = rel[side & (ty == "dash")]
    outer_u = np.r_[rel[side & np.isin(ty, ("white", "barrier", "sep", "bus", "dash", "hole"))], avail]
    W = W_GRID[None, :]
    C = np.full((N_MAX + 1, len(W_GRID)), np.inf)
    for n in range(0, N_MAX + 1):
        if n == 0:
            C[0, :] = 0.0 if avail < 2.2 else C_LEFT * max(0.0, avail - 0.8) + 2.0
            continue
        k = np.arange(1, n)[:, None]
        bk = k * W                                           # (n-1, nw)
        c = np.zeros(len(W_GRID))
        if n > 1:
            if len(div_u):
                d = np.abs(bk[:, :, None] - div_u[None, None, :]).min(-1)
                c += C_DIV * np.minimum(d / TAU, 1.0).sum(0) ** 1
            else:
                c += C_DIV * (n - 1)
        bn = n * W_GRID
        gap = avail - bn
        over = np.maximum(0.0, -gap - 0.4)
        c += C_OVER * over
        do = np.abs(bn[:, None] - outer_u[None, :]).min(-1)
        marked = do <= TAU
        strip = np.maximum(0.0, gap - 0.8)
        c += np.where(gap <= 0.8, 0.5 * np.minimum(do / TAU, 1.0) ** 2,
                      np.where(marked, np.minimum(do / TAU, 1.0) ** 2 + C_STRIP + 0.15 * strip, C_LEFT * strip))
        # 解释不了的分界线（车道内部）+ 外侧带里的虚线
        if len(div_u):
            allb = np.concatenate([np.zeros((1, len(W_GRID))), bk, bn[None, :]], 0)  # (n+1, nw)
            dd = np.abs(allb[:, :, None] - div_u[None, None, :]).min(0)            # (nw, m)
            inside = (div_u[None, :] > 0.4) & (div_u[None, :] < bn[:, None] - 0.4)
            c += C_EXTRA * ((dd > TAU) & inside).sum(-1)
        if len(dash_u):
            c += C_EXTRA * (dash_u[None, :] > bn[:, None] + 0.5).sum(-1)
        for au in arrows_here:
            r = sigma * (au - a)
            allb = np.arange(0, n + 1)[:, None] * W
            dmin = np.abs(allb - r).min(0)
            c += C_ARR * ((r < 0) | (r > bn) | (dmin < 0.7))
        mu, sd, wt = W_PRIOR
        c += wt * ((W_GRID - mu) / sd) ** 2
        C[n] = c
    return C


def dp(Cs, sigma, forbid_zero):
    """沿 s（口门→外）DP。状态 (n, wi)。"""
    T = len(Cs)
    nN, nW = Cs[0].shape
    states = [(n, w) for n in range(nN) for w in range(nW)]
    S = len(states)
    nn = np.array([s[0] for s in states])
    ww = np.array([s[1] for s in states])
    trans = np.zeros((S, S))
    dn = nn[None, :] - nn[:, None]        # 从 i(前一断面，更靠口门) 到 j(更外)
    dw = np.abs(ww[None, :] - ww[:, None])
    trans += np.where(dn != 0, C_CHG + 3.0 * (np.abs(dn) - 1), 0.0)
    trans += np.where((dn == 0) & (dw == 1), C_WSTEP, 0.0)
    trans += np.where((dn == 0) & (dw > 1), np.inf, 0.0)
    if sigma > 0:   # 进口：越往外车道越多 = 往路口方向在减车道，少见
        trans += np.where(dn > 0, C_CHG_UP, 0.0)
    cost = np.array([Cs[0][n, w] for n, w in states])
    if forbid_zero:
        cost = np.where(nn == 0, np.inf, cost)
    back = []
    for t in range(1, T):
        tot = cost[:, None] + trans
        b = np.argmin(tot, 0)
        cost = tot[b, np.arange(S)] + np.array([Cs[t][n, w] for n, w in states])
        if forbid_zero:
            cost = np.where(nn == 0, np.inf, cost)
        back.append(b)
    k = int(np.argmin(cost))
    path = [k]
    for b in reversed(back):
        k = int(b[k])
        path.append(k)
    path.reverse()
    return [states[k] for k in path], float(cost.min())



# ------------------------------------------------------------------ 主过程

def observe(road, ev, grid):
    """第一遍：断面、停止线、箭头、中线来源。方向组成留到 decide_modes 统一定。"""
    ax = road["axis"]
    stop = _stop_info(road, ev, grid)
    cws = _crosswalk_s(road, ev)
    sm = road.get("s_mouth", 0.0)
    s0_in = stop[0] if stop else (cws if cws is not None else sm)
    s0_out = cws if cws is not None else (stop[0] if stop else sm)
    s_start = min(s0_in, s0_out)
    s_end = min(ax.length - 1.0, s_start + S_LEN)
    secs = []
    for s in np.arange(s_start, s_end, DS):
        sec = section(road, ev, grid, s)
        if sec is not None and sec["hi"] - sec["lo"] >= 2.5:
            secs.append(sec)
    if len(secs) < 5:
        return None
    ss = np.array([sc["s"] for sc in secs])
    i0_in = int(np.searchsorted(ss, s0_in - 0.5))
    i0_out = int(np.searchsorted(ss, s0_out - 0.5))
    # 无停止线/人行横道定位时，起点断面可能还在路口开阔区里：跳过明显比下游宽的断面
    wd = np.array([sc["hi"] - sc["lo"] for sc in secs])
    ref_w = float(np.median(wd[min(len(wd) - 1, max(i0_in, i0_out) + 3):])) if len(wd) > 6 else float(np.median(wd))
    while i0_in < len(secs) - 4 and wd[i0_in] > 1.5 * ref_w + 1.0:
        i0_in += 1
    while i0_out < len(secs) - 4 and wd[i0_out] > 1.5 * ref_w + 1.0:
        i0_out += 1
    i_med = int(np.searchsorted(ss, max(s0_in, s0_out) + 1.5))
    if i_med >= len(secs) - 3:
        i_med = min(i0_in, i0_out)
    if min(i0_in, i0_out) >= len(secs) - 3:
        return None
    arr = []
    for xy, hv, code in ev.arrows:
        pt = Point(xy)
        if ax.distance(pt) > 20:
            continue
        s = ax.project(pt)
        if s < 0.5:
            continue
        i = int(np.argmin(np.abs(ss - s)))
        sc = secs[i]
        p, t, nrm = _frame(ax, s)
        u = float((np.asarray(xy) - p) @ nrm)
        if not (sc["lo"] - 0.3 <= u <= sc["hi"] + 0.3) or abs(ss[i] - s) > 3:
            continue
        inward, outward = np.dot(hv, -t) > 0.5, np.dot(hv, t) > 0.5
        if inward or outward:
            arr.append(dict(s=s, u=u, inward=bool(inward), code=code, mv=movement(code)))
    med = _median_track(secs, i_med)
    width = float(np.median([sc["hi"] - sc["lo"] for sc in secs[i_med:]]))
    return dict(axis=ax, secs=secs, stop=stop, s0_in=s0_in, s0_out=s0_out, i0_in=i0_in, i0_out=i0_out,
                i_med=i_med, arrows=arr, med=med, width=width, road=road)


def own_mode(o):
    """仅凭本幅证据定方向组成；定不了返回 None。"""
    if o["med"] is not None:
        return "two_way"
    n_in = sum(1 for a in o["arrows"] if a["inward"])
    n_out = sum(1 for a in o["arrows"] if not a["inward"])
    stop = o["stop"]
    if stop and stop[3] >= 0.7 and n_out == 0:
        return "in_only"
    if n_in and not n_out:
        # 进口箭头覆盖整个断面宽度 → 单向进口
        us = [a["u"] for a in o["arrows"] if a["inward"]]
        sc = o["secs"][o["i_med"]]
        if (max(us) - min(us)) + 3.5 >= 0.7 * (sc["hi"] - sc["lo"]):
            return "in_only"
    if n_out and not n_in and not stop:
        return "out_only"
    if stop and stop[3] < 0.7:
        return "two_way"
    return None


def decide_modes(obs):
    """本幅证据定不了的，按右侧通行看左右相邻的平行车行道：面向外看，左邻是进口 → 本幅是出口；
    右邻是出口 → 本幅是进口。仍定不了 → 假定双向（中线取路面中线，记为推断）。"""
    for o in obs:
        o["mode"] = own_mode(o)
    for _ in range(2):
        for o in obs:
            if o["mode"] is not None:
                continue
            p, t, nrm = _frame(o["axis"], o["s0_out"] + 10)
            left = right = None
            for q in obs:
                if q is o or q["mode"] is None:
                    continue
                pq, tq, _ = _frame(q["axis"], q["axis"].project(Point(p)))
                if np.dot(t, tq) < np.cos(np.radians(30)):
                    continue
                d = pq - p
                lat, lon = float(d @ nrm), float(d @ t)
                if abs(lon) > 25 or abs(lat) > 35 or abs(lat) < 2:
                    continue
                if lat > 0 and (left is None or lat < left[0]):
                    left = (lat, q["mode"])
                if lat < 0 and (right is None or -lat < right[0]):
                    right = (-lat, q["mode"])
            if left and left[1] in ("in_only", "two_way"):
                o["mode"], o["mode_src"] = "out_only", "相邻车行道推断"
            elif right and right[1] in ("out_only", "two_way"):
                o["mode"], o["mode_src"] = "in_only", "相邻车行道推断"
    for o in obs:
        if o["mode"] is None:
            o["mode"], o["mode_src"] = "two_way", "假定"
        o.setdefault("mode_src", "本幅证据")


def fit_obs(o):
    secs, mode = o["secs"], o["mode"]
    T = len(secs)
    i_med = o["i_med"]
    inferred = []
    if mode == "two_way":
        if o["med"] is not None:
            a_out, a_in, med_src = o["med"]
        elif o["stop"]:
            a_out = a_in = np.full(T, o["stop"][1])
            med_src = "stopline"
        else:
            a_out = a_in = np.array([(sc["lo"] + sc["hi"]) / 2 for sc in secs])
            med_src = "center_assumed"
            inferred.append("无中线标线：按路面中线假定双向")
    elif mode == "in_only":
        a_in = np.array([sc["lo"] for sc in secs]); a_out = a_in.copy(); med_src = "one_way"
    else:
        a_out = np.array([sc["hi"] for sc in secs]); a_in = a_out.copy(); med_src = "one_way"
    a_in, a_out = np.array(a_in, float), np.array(a_out, float)
    a_in[:i_med] = a_in[i_med]
    a_out[:i_med] = a_out[i_med]
    k = 2
    a_in = np.convolve(np.pad(a_in, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    a_out = np.convolve(np.pad(a_out, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    res = dict(o)
    res.update(mode=mode, med_src=med_src, inferred=inferred, a_in=a_in, a_out=a_out)
    for sigma, key, a, i0 in ((+1, "in", a_in, o["i0_in"]), (-1, "out", a_out, o["i0_out"])):
        if (mode == "in_only" and sigma < 0) or (mode == "out_only" and sigma > 0):
            res[key] = dict(i0=i0, n=np.zeros(T, int), w=np.full(T, 3.25), a=a, sigma=sigma, cost=0.0)
            continue
        Cs = []
        for sc, ai in list(zip(secs, a))[i0:]:
            ah = [ar["u"] for ar in o["arrows"] if ar["inward"] == (sigma > 0) and abs(ar["s"] - sc["s"]) <= 1.6]
            Cs.append(side_costs(sc, ai, sigma, ah))
        path, cost = dp(Cs, sigma, forbid_zero=True)
        n = np.array([path[0][0]] * i0 + [p[0] for p in path])
        w = np.array([W_GRID[path[0][1]]] * i0 + [W_GRID[p[1]] for p in path])
        res[key] = dict(i0=i0, n=n, w=w, a=a, sigma=sigma, cost=cost)
    _diagnose(res)
    _lane_moves(res)
    return res


def lane_u(side, i, k):
    """第 i 个断面、本侧第 k 条车道（从中线数）的中心横坐标 u。"""
    return side["a"][i] + side["sigma"] * (k + 0.5) * side["w"][i]


def match_lanes(side, i, j):
    """断面 i 的车道 → 断面 j 的车道（按横向位置最近；超过 0.6 个车道宽视为在 j 处不存在 → -1）。"""
    ni, nj = int(side["n"][i]), int(side["n"][j])
    if nj == 0:
        return [-1] * ni
    uj = np.array([lane_u(side, j, k) for k in range(nj)])
    out = []
    for k in range(ni):
        d = np.abs(uj - lane_u(side, i, k))
        m = int(np.argmin(d))
        out.append(m if d[m] <= 0.6 * side["w"][j] else -1)
    return out


def _diagnose(res):
    """拟合质量：分界线有观测支撑的比例、解释不了的标线数、剩余路面宽度。"""
    sup, tot, extra, left = 0, 0, 0, []
    for key in ("in", "out"):
        side = res[key]
        sigma = side["sigma"]
        for i in range(side["i0"], len(res["secs"])):
            sc = res["secs"][i]
            n, w, a = int(side["n"][i]), float(side["w"][i]), side["a"][i]
            if n == 0:
                continue
            rel = sigma * (sc["u"] - a)
            du = rel[(rel > 0.25) & np.isin(sc["typ"], DIV_TYPES)]
            for b in np.arange(1, n) * w:
                tot += 1
                sup += bool(len(du) and np.abs(du - b).min() <= TAU)
            allb = np.arange(0, n + 1) * w
            extra += int(sum(1 for v in du if v < n * w - 0.4 and np.abs(allb - v).min() > TAU))
            lim = sc["hi"] if sigma > 0 else sc["lo"]
            left.append(sigma * (lim - a) - n * w)
    res["quality"] = dict(divider_support=round(sup / tot, 3) if tot else None, dividers=tot,
                          unexplained_marks=extra, leftover_p50=round(float(np.median(left)), 2) if left else None)


def _lane_moves(res):
    """进口每条车道的箭头转向（停止线外 45 m 内）。按横向位置把箭头所在车道对到口门车道；
    对不上的（展宽前就分流走了，如进渠化右转道的右转车道）记在 pre_mouth。"""
    side = res["in"]
    i0 = side["i0"]
    n = int(side["n"][i0])
    res["lane_mv"], res["pre_mouth"] = [], []
    if n == 0:
        return
    ss = np.array([sc["s"] for sc in res["secs"]])
    mv = [set() for _ in range(n)]
    for ar in res["arrows"]:
        if not ar["inward"] or ar["s"] > res["s0_in"] + 45:
            continue
        i = max(int(np.argmin(np.abs(ss - ar["s"]))), i0)
        ni, wi = int(side["n"][i]), float(side["w"][i])
        k = int(np.floor((ar["u"] - side["a"][i]) / wi))
        if not (0 <= k < ni):
            continue
        kk = match_lanes(side, i, i0)[k]
        if kk >= 0:
            mv[kk] |= set(ar["mv"])
        else:
            res["pre_mouth"].append(dict(s=float(ss[i]), i=i, k=k, mv=ar["mv"], u=float(lane_u(side, i, k))))
    res["lane_mv"] = ["".join(sorted(m)) for m in mv]


def fit_all(legs, ev, grid, center):
    roads = make_roads(legs, center)
    obs = [o for o in (observe(r, ev, grid) for r in roads) if o is not None]
    decide_modes(obs)
    fits = [fit_obs(o) for o in obs]
    for i, f in enumerate(fits):
        f["id"] = f"R{i}"
    return fits
