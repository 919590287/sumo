"""路臂发现：可行驶面的测地距离超水平集过滤（persistence）。

思路：从路口中心出发算同层可行驶面内的测地距离 g。把 {g >= ρ} 的连通块随 ρ 从大到小跟踪：
远处每条车行道是一个独立连通块，往里走到路口开阔区时它们并到一起——并入处就是口门。
  - 弯路、无标线支路、无参考线支路都能发现（只要有路面）；
  - 物理中分带/隔离栏两侧的车行道是两个块（各自单向）；
  - 远端又分岔（邻口）的分支在更大的 ρ 先合并，算同一条路臂；
  - 中途接入的辅道、渠化右转道在核心外先并入某条路臂，记为该路臂的分支，不单独成臂。
"""
import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

RES = 0.5          # 栅格分辨率 m
EXT = 130.0        # 半窗口 m
L_MIN = 28.0       # 路臂最短持续长度（并入点以外还要延伸这么远）
W_MIN = 2.2        # 路臂平均宽度下限（按扣掉护栏后的栅格算；3.5 m 两侧带护栏的单车道支路约 2.7 m，要保留）
R_CORE = 48.0      # 并入点测地距离上限：更远处的并合视为路臂自身的分岔（邻口、辅道）
STEP = 1.0         # 过滤步长 m


def rasterize(drivable, walls, center, ext=EXT, res=RES):
    cx, cy = center
    xs = np.arange(cx - ext, cx + ext, res) + res / 2
    ys = np.arange(cy - ext, cy + ext, res) + res / 2
    X, Y = np.meshgrid(xs, ys)
    m = shapely.contains_xy(drivable, X, Y)
    if walls is not None and not walls.is_empty:
        m &= ~shapely.contains_xy(walls, X, Y)
    return m, xs, ys


def geodesic(mask, src):
    """8 邻域测地距离（米）。src = (row, col)。"""
    H, W = mask.shape
    idx = -np.ones(mask.shape, int)
    idx[mask] = np.arange(mask.sum())
    rows, cols, wts = [], [], []
    for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5)):
        # 用切片对齐 a(r,c) 与 b(r+dr, c+dc)
        r0, r1 = max(0, -dr), H - max(0, dr)
        c0, c1 = max(0, -dc), W - max(0, dc)
        a = idx[r0:r1, c0:c1]
        b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ok = (a >= 0) & (b >= 0)
        if dr == 1 and dc != 0:  # 对角：两条正交邻边都通才允许（防止斜穿 1 格厚的墙）
            o1 = idx[r0 + dr:r1 + dr, c0:c1] >= 0
            o2 = idx[r0:r1, c0 + dc:c1 + dc] >= 0
            ok &= o1 & o2
        rows.append(a[ok]); cols.append(b[ok]); wts.append(np.full(ok.sum(), w * RES))
    r = np.concatenate(rows); c = np.concatenate(cols); w = np.concatenate(wts)
    G = csr_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(mask.sum(), mask.sum()))
    d = dijkstra(G, indices=idx[src], directed=False)
    out = np.full(mask.shape, np.inf)
    out[mask] = d
    return out


def nearest_cell(mask, xs, ys, p):
    rr, cc = np.nonzero(mask)
    k = np.argmin((xs[cc] - p[0]) ** 2 + (ys[rr] - p[1]) ** 2)
    return rr[k], cc[k]


def filtration(dist, mask):
    """超水平集过滤。返回路臂列表：dict(rho=并入测地距离, cells=该块在并入时刻的布尔掩膜, maxd=最远测地距离)。"""
    fin = np.isfinite(dist)
    gmax = float(dist[fin].max())
    levels = np.arange(np.floor(gmax), -STEP, -STEP)
    prev_lab, prev_n = None, 0
    comp_info = {}  # 上一层块号 → dict(maxd, legs=[leg ids])
    legs = []
    eight = np.ones((3, 3), int)
    for rho in levels:
        sup = fin & (dist >= rho)
        lab, n = ndimage.label(sup, structure=eight)
        maxd = ndimage.maximum(np.where(fin, dist, 0), lab, index=np.arange(1, n + 1)) if n else []
        info = {}
        if prev_lab is not None:
            # 上一层每个块落在本层哪个块里
            on = prev_lab > 0  # 上一层超水平集 ⊂ 本层，每个旧块整体落在一个新块里
            pairs = np.unique(np.stack([prev_lab[on], lab[on]], 1), axis=0)
            children = {}
            for k, parent in pairs:
                children.setdefault(int(parent), []).append(int(k))
        else:
            children = {}
        for j in range(1, n + 1):
            ch = children.get(j, [])
            info_j = dict(maxd=float(maxd[j - 1]), legs=[])
            sig = [c for c in ch if comp_info[c]["maxd"] - rho >= L_MIN]
            if len(sig) >= 2 and rho <= R_CORE:
                for c in sig:
                    if comp_info[c]["legs"]:
                        continue
                    cells = prev_lab == c
                    legs.append(dict(rho=float(rho + STEP), cells=cells, maxd=comp_info[c]["maxd"]))
                    comp_info[c]["legs"] = [len(legs) - 1]
            for c in ch:
                info_j["legs"] += comp_info[c]["legs"]
            info[j] = info_j
        comp_info = info
        prev_lab, prev_n = lab, n
    return legs


def _smooth(P, k=3):
    if len(P) < 2 * k + 1:
        return P
    Q = P.copy()
    for i in range(len(P)):
        a, b = max(0, i - k), min(len(P), i + k + 1)
        Q[i] = P[a:b].mean(0)
    Q[0], Q[-1] = P[0], P[-1]
    return Q


def leg_axis(leg, dist, xs, ys, s_len=95.0, step=2.0):
    """沿测地壳层取截面质心串成轴线（口门 → 外）。远端分岔时跟随与前一点最近的分支。"""
    cells = leg["cells"]
    rho = leg["rho"]
    pts = []
    prev = None
    for t in np.arange(rho, min(leg["maxd"], rho + s_len), step):
        sh = cells & (dist >= t) & (dist < t + step)
        if sh.sum() < 3:
            break
        lab, n = ndimage.label(sh, structure=np.ones((3, 3), int))
        best, bd = None, np.inf
        for j in range(1, n + 1):
            rr, cc = np.nonzero(lab == j)
            if len(rr) < 3:
                continue
            c = np.array([xs[cc].mean(), ys[rr].mean()])
            d = 0.0 if prev is None else np.hypot(*(c - prev))
            # 首截面取最大块；之后取最近块
            score = -len(rr) if prev is None else d
            if score < bd:
                best, bd = c, score
        if best is None or (prev is not None and bd > 3 * step + 6):
            break
        pts.append(best)
        prev = best
    if len(pts) < 4:
        return None
    return smooth_line(LineString(pts))


def smooth_line(line, sigma=6.0, step=1.0):
    """按弧长重采样后高斯平滑（端点不动）。轴线只是坐标系，平滑不影响拟合精度，只让断面不乱摆。"""
    from scipy.ndimage import gaussian_filter1d
    L = line.length
    if L < 3 * step:
        return line
    ss = np.linspace(0, L, max(4, int(L / step) + 1))
    P = np.array([line.interpolate(v).coords[0] for v in ss])
    Q = np.stack([gaussian_filter1d(P[:, 0], sigma / step, mode="nearest"),
                  gaussian_filter1d(P[:, 1], sigma / step, mode="nearest")], 1)
    Q[0], Q[-1] = P[0], P[-1]
    # 端部向内收敛：前 sigma 米线性过渡回原始点，避免端点处折角
    m = int(sigma / step)
    for i in range(min(m, len(Q) // 2)):
        a = i / m
        Q[i] = a * Q[i] + (1 - a) * P[i]
        Q[-1 - i] = a * Q[-1 - i] + (1 - a) * P[-1 - i]
    return LineString(Q)


def extend_inward(line, center, max_ext=32.0, trim=10.0, base=(12.0, 32.0)):
    """把口门端（起点）沿直线段方向向路口中心外推，让停止线、人行横道、进口箭头落在轴线上。
    口门附近 trim 米的测地质心会被转角开口拉偏（花都大十字西臂偏 45°），先截掉，再用 base 区间的方向外推。
    返回 (新轴线, 原口门在新轴线上的 s)。"""
    from shapely.ops import substring
    L = line.length
    if L < base[1] + 5:
        trim, base = 0.0, (0.0, min(L, 15.0))
    core = substring(line, trim, L) if trim > 0 else line
    a = np.asarray(line.interpolate(base[0]).coords[0])
    b = np.asarray(line.interpolate(base[1]).coords[0])
    t = a - b
    t = t / (np.linalg.norm(t) + 1e-9)
    p0 = np.asarray(core.coords[0])
    dc = float(np.dot(np.asarray(center) - p0, t))
    ext = float(np.clip(dc - 2.0, 0.0, max_ext))
    if ext < 1.0:
        return core, 0.0
    # 外推直线段与保留的测地轴线之间可能有折角（断面在折角处会前后翻折）：拼接后再平滑一次
    joined = LineString(np.vstack([p0 + t * ext, np.asarray(core.coords)]))
    return smooth_line(joined, sigma=5.0), max(0.0, ext - trim)


def width_of(leg, dist, a=8.0, b=24.0):
    sh = leg["cells"] & (dist >= leg["rho"] + a) & (dist < leg["rho"] + b)
    return sh.sum() * RES * RES / (b - a)


def find_legs(drivable, walls, center):
    mask, xs, ys = rasterize(drivable, walls, center)
    src = nearest_cell(mask, xs, ys, center)
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3), int))
    mask = lab == lab[src]
    dist = geodesic(mask, src)
    raw = filtration(dist, mask)
    legs = []
    for L in raw:
        w = width_of(L, dist)
        if w < W_MIN:
            continue
        ax = leg_axis(L, dist, xs, ys)
        if ax is None or ax.length < 15:
            continue
        L["width"] = w
        L["axis"] = ax
        legs.append(L)
    return legs, dict(mask=mask, dist=dist, xs=xs, ys=ys, src=src)


def refine_center(legs, c0):
    """各路臂口门点的质心。（轴线交点法对轴线方向误差太敏感：花都大十字会被推出 25 m，导致两臂并成一臂。）"""
    if not legs:
        return np.asarray(c0, float)
    return np.mean([np.asarray(L["axis"].coords[0]) for L in legs], 0)


def walls_of(B):
    """物理阻断：隔离设施（中分护栏、路侧护栏）。路侧护栏也要算：它把主路与旁边的辅道、渠化区隔开
    （只算中分护栏时花都大十字四臂会连成两臂）。"""
    bar = [l for l, t in B["bounds"] if t in ("barrier", "sep")]
    if not bar:
        return None
    return unary_union([l.buffer(0.4) for l in bar])
