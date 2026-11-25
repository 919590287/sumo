"""射线中点法生成车道中心线样本，再把样本串成车道中心线。

思路：沿每条边界每 1 m 取一点，向两侧法向发射线，找第一条大致平行的边界。
  - 距离在 [W_MIN, W_MAX]：两条边界之间是一条车道，中点即车道中心（normal / edgepair）。
  - 距离在 (W_MAX, W_WIDE]：宽走廊（展宽、收窄的过渡段），在离标线 W_REF/2 处生成“贴边候选”
    （anchored）。原车道的贴边候选与过渡前后的正常样本连续，新增车道的贴边候选在走廊够宽时出现，
    与分隔线出现后的正常样本连续。
法向取自每条边界自己的局部切向，所以弯道、渠化右转道都按局部方向量宽度。
"""
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, dijkstra
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from shapely.geometry import LineString, Point

from .boundaries import MARKING_TYPES

RAY_REACH = 12.0  # 射线搜索上限：≥ 2 × 单车道物理上限(1.35 × 4.2 m)，判定阈值由 params 按数据给出
STEP = 1.0
from .params import F


def _max_par_sin():
    """运行时读取 params.F（敏感性测试改 F 后立即生效，不在导入时固化）。"""
    return np.sin(np.radians(F["par_deg"]))

TYPES = ["white", "dash", "yellow", "bus", "barrier", "edge", "sep"]


def _segments(bounds):
    A, B, T = [], [], []
    for l, t in bounds:
        c = np.asarray(l.coords)[:, :2]
        if len(c) < 2:
            continue
        A.append(c[:-1])
        B.append(c[1:])
        T.append(np.full(len(c) - 1, TYPES.index(t)))
    if not A:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=int)
    A = np.vstack(A)
    B = np.vstack(B)
    T = np.concatenate(T)
    ok = np.linalg.norm(B - A, axis=1) > 1e-6
    return A[ok], B[ok], T[ok]


def _samples(bounds, step=STEP):
    P, Tn, St = [], [], []
    for l, t in bounds:
        L = l.length
        if L < 1.0:
            continue
        d = np.arange(step / 2, L, step)
        pts = shapely.get_coordinates(shapely.line_interpolate_point(l, d))
        a = shapely.get_coordinates(shapely.line_interpolate_point(l, np.clip(d - 0.5, 0, L)))
        b = shapely.get_coordinates(shapely.line_interpolate_point(l, np.clip(d + 0.5, 0, L)))
        tv = b - a
        n = np.linalg.norm(tv, axis=1)
        ok = n > 1e-6
        P.append(pts[ok])
        Tn.append(tv[ok] / n[ok, None])
        St.append(np.full(ok.sum(), TYPES.index(t)))
    if not P:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=int)
    return np.vstack(P), np.vstack(Tn), np.concatenate(St)


def cast(bounds):
    A, B, ST = _segments(bounds)
    P, T, SRC = _samples(bounds)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    # 两个方向的射线
    P2 = np.vstack([P, P])
    N2 = np.vstack([N, -N])
    T2 = np.vstack([T, T])
    S2 = np.concatenate([SRC, SRC])
    start = P2 + N2 * 0.05
    end = P2 + N2 * RAY_REACH
    rays = shapely.linestrings(np.stack([start, end], 1))
    segs = shapely.linestrings(np.stack([A, B], 1))
    tree = shapely.STRtree(segs)
    ri, si = tree.query(rays, predicate="intersects")
    D = B[si] - A[si]
    Dl = np.linalg.norm(D, axis=1)
    Dh = D / Dl[:, None]
    n = N2[ri]
    p = P2[ri]
    w = A[si] - p
    den = n[:, 0] * D[:, 1] - n[:, 1] * D[:, 0]
    s = (w[:, 0] * D[:, 1] - w[:, 1] * D[:, 0]) / np.where(np.abs(den) < 1e-12, np.nan, den)
    par = np.abs(T2[ri, 0] * Dh[:, 1] - T2[ri, 1] * Dh[:, 0])
    ok = (s > 0.12) & (s <= RAY_REACH) & np.isfinite(s)  # 不跳过贴得很近的双线
    ri, si, s, par = ri[ok], si[ok], s[ok], par[ok]
    # 每条射线取最近的命中；最近命中不平行则这条射线作废（被横向物挡住）
    order = np.lexsort((s, ri))
    ri, si, s, par = ri[order], si[order], s[order], par[order]
    first = np.ones(len(ri), bool)
    first[1:] = ri[1:] != ri[:-1]
    ri, si, s, par = ri[first], si[first], s[first], par[first]
    good = par < _max_par_sin()
    ri, si, s = ri[good], si[good], s[good]
    return dict(P=P2[ri], N=N2[ri], T=T2[ri], src=S2[ri], hit=ST[si], d=s)


def local_width(Pts, d, w_global, cell=20.0, min_n=30, lo=2.2, hi=5.2, T=None, nbin=12):
    """每个样本的“本地典型车道宽”：周围约 60 m 窗口内、走向相近（±22.5°）的射线间距中位数，
    按格心双线性插值成连续场。
    同一路口不同臂的车道宽可以差 1 m 以上（老城区 2.8 m 与主干道 3.75 m 并存），不能用一个全局值；
    也不能不分方向混算——相交的窄支路、非机动车道会把垂直方向主干道的上限拉低，把 3.75 m 的
    标准车道误判成宽走廊。格子边界处也不能跳变，否则一条车道会被拦腰切断。"""
    ok = (d >= lo) & (d <= hi)
    fx = Pts[:, 0] / cell - 0.5
    fy = Pts[:, 1] / cell - 0.5
    gx = np.floor(Pts[:, 0] / cell).astype(int)
    gy = np.floor(Pts[:, 1] / cell).astype(int)
    if T is None:
        ob = np.zeros(len(d), int)
        nbin = 1
    else:
        ang = np.mod(np.arctan2(T[:, 1], T[:, 0]), np.pi)
        ob = np.minimum((ang / np.pi * nbin).astype(int), nbin - 1)
    buckets, allb = {}, {}
    for x, y, o, v in zip(gx[ok], gy[ok], ob[ok], d[ok]):
        buckets.setdefault((x, y, o), []).append(v)
        allb.setdefault((x, y), []).append(v)
    cache = {}

    def val(x, y, o):
        key = (x, y, o)
        if key not in cache:
            vals = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for do in ((-1, 0, 1) if nbin > 1 else (0,)):
                        vals += buckets.get((x + dx, y + dy, (o + do) % nbin), [])
            if len(vals) < min_n:  # 本方向样本不够：退回不分方向
                vals = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        vals += allb.get((x + dx, y + dy), [])
            cache[key] = float(np.clip(np.median(vals), 2.5, 4.5)) if len(vals) >= min_n else w_global
        return cache[key]

    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    wx = fx - i0
    wy = fy - j0
    out = np.empty(len(d))
    for k in range(len(d)):
        x, y, o = i0[k], j0[k], ob[k]
        out[k] = ((1 - wx[k]) * (1 - wy[k]) * val(x, y, o) + wx[k] * (1 - wy[k]) * val(x + 1, y, o)
                  + (1 - wx[k]) * wy[k] * val(x, y + 1, o) + wx[k] * wy[k] * val(x + 1, y + 1, o))
    return out


def on_road(c, dv):
    """射线中点是否在（外扩 0.3 m 的）可行驶面上。dv 为 None 时全为 True。"""
    if dv is None:
        return np.ones(len(c["d"]), bool)
    ctr = c["P"] + c["N"] * (c["d"] / 2)[:, None]
    return shapely.contains_xy(dv, ctr[:, 0], ctr[:, 1])


def emit(c, P, drivable=None):
    """drivable：车道中心必须落在可行驶面上。射线从每条边界向两侧发射，路缘、岛边的“外侧”射线会在
    岛内、绿化带内找到另一条平行边界（窄中央分隔岛两条边相距 3 m 左右），中点不在路面上的样本一律丢掉。"""
    from .params import F
    Pp, N, T, src, hit, d = c["P"], c["N"], c["T"], c["src"], c["hit"], c["d"]
    dv = drivable.buffer(0.3) if drivable is not None and not drivable.is_empty else None
    on = on_road(c, dv)
    # 本地车道宽只用路面上的射线统计（岛内、绿化带内的假间距不参与）
    wl = local_width(Pp, np.where(on, d, -1.0), P["w_mode"], T=T)
    W_MIN = np.maximum(2.2, F["w_min_k"] * wl)
    W_MAX = np.minimum(F["w_max_k"] * wl, 2 * W_MIN - 0.2)
    W_WIDE = 2 * W_MAX
    mk = np.array([TYPES.index(t) for t in MARKING_TYPES])  # 只有路面标线算“有标线”；隔离设施+路缘的走廊需箭头/停止线支撑
    src_m = np.isin(src, mk)
    hit_m = np.isin(hit, mk)
    out = []
    lane = (d >= W_MIN) & (d <= W_MAX) & on
    kind = np.where(src_m | hit_m, 0, 1)  # 0 normal 1 edgepair
    ctr = Pp + N * (d / 2)[:, None]
    out.append(dict(X=ctr[lane], T=T[lane], w=d[lane], kind=kind[lane],
                    src=src[lane], hit=hit[lane], sp=Pp[lane], hp=(Pp + N * d[:, None])[lane]))
    ctr = Pp + N * (wl / 2)[:, None]
    on_w = shapely.contains_xy(dv, ctr[:, 0], ctr[:, 1]) if dv is not None else np.ones(len(d), bool)
    wide = (d > W_MAX) & (d <= W_WIDE) & src_m & on_w
    out.append(dict(X=ctr[wide], T=T[wide], w=wl[wide], kind=np.full(wide.sum(), 2),
                    src=src[wide], hit=hit[wide], sp=Pp[wide], hp=(Pp + N * wl[:, None])[wide]))
    P["w_local_range"] = [round(float(np.min(wl)), 2), round(float(np.max(wl)), 2)] if wl.size else []
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def link(S, P, max_ang=15.0):
    r, max_lat = P["link_r"], P["link_lat"]
    X, T = S["X"], S["T"]
    tree = cKDTree(X)
    pairs = tree.query_pairs(r, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(X)), len(X), None
    i, j = pairs[:, 0], pairs[:, 1]
    dot = np.abs((T[i] * T[j]).sum(1))
    v = X[j] - X[i]
    lat = np.abs(T[i, 0] * v[:, 1] - T[i, 1] * v[:, 0])
    ok = (dot > np.cos(np.radians(max_ang))) & (lat < max_lat)
    i, j = i[ok], j[ok]
    dist = np.linalg.norm(X[j] - X[i], axis=1) + 1e-3
    n = len(X)
    G = coo_matrix((dist, (i, j)), shape=(n, n)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    return lab, ncomp, G


def chain(idx, S, G, return_rest=False):
    """把一个分量的样本排成有序折线：MST 直径路径 + 1 m 分箱中位数 + 平滑。"""
    X = S["X"][idx]
    if len(idx) < 3:
        return None
    sub = G[idx][:, idx]
    mst = minimum_spanning_tree(sub + sub.T)
    mst = mst + mst.T
    d0 = dijkstra(mst, indices=0, directed=False)
    a = int(np.nanargmax(np.where(np.isinf(d0), -1, d0)))
    da, pred = dijkstra(mst, indices=a, directed=False, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isinf(da), -1, da)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    if len(path) < 2:
        return None
    line = LineString(X[path])
    if line.length < 2:
        return None
    s = shapely.line_locate_point(line, shapely.points(X))
    # 只保留离路径近的样本；分支上的样本由 chain_all 另起一条车道
    dd = shapely.distance(shapely.points(X), line)
    keep = dd < 0.6
    if return_rest:
        rest = idx[~keep]
    s, Xk, wk, kk = s[keep], X[keep], S["w"][idx][keep], S["kind"][idx][keep]
    bins = np.floor(s / 1.0).astype(int)
    ub = np.unique(bins)
    cx, cy, cw = [], [], []
    for b_ in ub:
        inb = bins == b_
        nrm = inb & (kk != 2)
        use = nrm if nrm.any() else inb  # 有正常样本时不用贴边候选
        cx.append(np.median(Xk[use, 0]))
        cy.append(np.median(Xk[use, 1]))
        cw.append(np.median(wk[nrm]) if nrm.any() else np.nan)
    cx, cy, cw = np.array(cx), np.array(cy), np.array(cw)
    if len(cx) < 3:
        return None
    # 贴边候选只负责连通：夹在正常样本之间的纯贴边段，位置用前后正常样本插值
    nb = np.array([np.any((bins == b_) & (kk != 2)) for b_ in ub])
    if nb.sum() >= 2:
        first, last = np.argmax(nb), len(nb) - 1 - np.argmax(nb[::-1])
        mid = (~nb) & (np.arange(len(nb)) > first) & (np.arange(len(nb)) < last)
        if mid.any():
            xs = ub.astype(float)
            cx[mid] = np.interp(xs[mid], xs[nb], cx[nb])
            cy[mid] = np.interp(xs[mid], xs[nb], cy[nb])
    cx, cy = smooth_xy(cx, cy)
    if return_rest:
        return np.stack([cx, cy], 1), cw, idx[keep], rest
    return np.stack([cx, cy], 1), cw


def _follows_samples(xy, X, T, max_sin=0.26):
    """折线走向与其样本所在边界的走向一致（夹角中位数 < 15°）。
    一个连通分量里混进相邻车道时，MST 直径会在两条车道之间斜穿，走向与样本明显不符。"""
    line = LineString(xy)
    if line.length < 2:
        return False
    s = shapely.line_locate_point(line, shapely.points(X))
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, np.clip(s - 1.0, 0, line.length)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, np.clip(s + 1.0, 0, line.length)))
    t = b - a
    n = np.linalg.norm(t, axis=1)
    ok = n > 1e-6
    if not ok.any():
        return False
    t = t[ok] / n[ok, None]
    sn = np.abs(t[:, 0] * T[ok, 1] - t[:, 1] * T[ok, 0])
    return float(np.median(sn)) < max_sin


def chain_all(idx, S, G, min_n=6, depth=3):
    """一个分量可能含 Y 形分叉、展宽分出的车道、渠化右转支路：MST 直径只能取走其中一条，
    剩下的样本按连通性再分组，各自再取直径，最多 depth 层。返回 [(xy, cw, 用到的样本序号)]。"""
    out = []
    todo = [(idx, 0)]
    while todo:
        ix, dep = todo.pop()
        if len(ix) < min_n or not np.any(S["kind"][ix] != 2):
            continue
        r = chain(ix, S, G, return_rest=True)
        if r is None:
            continue
        xy, cw, used, rest = r
        if not _follows_samples(xy, S["X"][used], S["T"][used]):
            continue  # 分量里串进了别的车道，直径路径斜穿车道，不是一条车道
        out.append((xy, cw, used))
        if dep >= depth or len(rest) < min_n:
            continue
        sub = G[rest][:, rest]
        nc, lab = connected_components(sub + sub.T, directed=False)
        cnt = np.bincount(lab)
        for k in np.nonzero(cnt >= min_n)[0]:
            todo.append((rest[lab == k], dep + 1))
    return out


def smooth_xy(cx, cy, win=None):
    win = int(round(F["smooth_m"])) if win is None else win
    n = len(cx)
    w = min(win, n if n % 2 else n - 1)
    if w >= 5:
        from scipy.signal import savgol_filter
        cx = savgol_filter(cx, w, 2, mode="interp")
        cy = savgol_filter(cy, w, 2, mode="interp")
    elif w >= 3:
        cx = uniform_filter1d(cx, w, mode="nearest")
        cy = uniform_filter1d(cy, w, mode="nearest")
    return cx, cy
