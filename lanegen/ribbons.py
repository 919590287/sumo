"""Phase 3 切片：以纵向标线链为骨架的车道带剖分（替代有标线车行道上的逐断面 Viterbi）。

旧做法每个断面独立地在 K 条候选分界里吸附标线，分界线 k 只是“第 k 条”的排序号，没有身份；
相邻断面把 k 吸到不同的真实标线上时，后续插值 / 平滑会把分界线斜着拉过整条车道，形成阶梯压线。

这里先沿车行道把各断面的标线观测串成有身份的标线链（虚线缺口按横向连续性桥接），
每个断面上相邻两条链（或链与车行道边界）之间就是一条车道带（ribbon）：
  - 有标线处 1 条带 = 1 条车道，中心严格居中于两条真实标线之间；
  - 过宽的无标线带（无标线宽出口、展宽渐变前段）按本带宽度和带内箭头组数等分；
  - 车道数只在标线链起止、或无标线带宽度容不下时改变。
输出与 _viterbi_partition 同构 (K_seq, B_cand, SN_cand)，下游分段 / 追踪 / 契约不变。
"""
import math

import numpy as np
from scipy.ndimage import median_filter

_TYPES = ("dash", "white", "yellow", "bus", "sep", "barrier")
_EDGE_PAD = 1.0       # 离车行道边界更近的标线不作内部分界（与 _sample_frenet 的 1.1 m 内部筛选同量级）
_MIN_CHAIN_M = 8.0    # 标线链最短长度（更短的是箭头残段、斑马线边角、渠化斜纹）
_MIN_LANE_W = 2.25    # 一条车道的最小宽度
_WIDE_K = 1.35        # 带宽 > _WIDE_K × 车道宽 才在带内细分
_GORE_K = 0.6         # 带宽 < _GORE_K × 车道宽：楔形区 / 导流带，不放车道


def mark_chains(F, max_gap_m=18.0):
    """把各断面的纵向标线观测串成链。返回 [dict(i0, i1, d(np.ndarray, 长 M，链外为 nan), types)]。
    关联规则：与活跃链按“上次位置 + 斜率外推”比横向距离，容差 0.30 m + 0.03 m/断面；最近者优先，一对一。"""
    M, ds = F["M"], F["ds"]
    G = max(2, int(math.ceil(max_gap_m / ds)))
    obs = [[(dv, t) for dv, t in F["obs_all"][i] if t in _TYPES] for i in range(M)]
    tracks = []
    for i in range(M):
        cand = []
        for ti, tr in enumerate(tracks):
            gap = i - tr["last"]
            if gap < 1 or gap > G:
                continue
            pred = tr["dv"][-1] + tr["slope"] * gap
            tol = 0.30 + 0.03 * gap
            for oi, (dv, t) in enumerate(obs[i]):
                e = abs(dv - pred)
                if e <= tol:
                    cand.append((e, ti, oi))
        cand.sort()
        used_t, used_o = set(), set()
        for e, ti, oi in cand:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti); used_o.add(oi)
            tr = tracks[ti]
            dv, t = obs[i][oi]
            gap = i - tr["last"]
            tr["slope"] = 0.7 * tr["slope"] + 0.3 * (dv - tr["dv"][-1]) / gap
            tr["st"].append(i); tr["dv"].append(dv); tr["types"].add(t); tr["last"] = i
        for oi, (dv, t) in enumerate(obs[i]):
            if oi not in used_o:
                tracks.append(dict(st=[i], dv=[dv], types={t}, last=i, slope=0.0))
    out = []
    for tr in tracks:
        st, dv = np.asarray(tr["st"]), np.asarray(tr["dv"], float)
        if (st[-1] - st[0]) * ds < _MIN_CHAIN_M or len(st) < 3:
            continue
        if len(dv) >= 3:
            dv = median_filter(dv, size=3, mode="nearest")
        d = np.full(M, np.nan)
        rng = np.arange(st[0], st[-1] + 1)
        d[rng] = np.interp(rng, st, dv)
        out.append(dict(i0=int(st[0]), i1=int(st[-1]), d=d, types=tr["types"]))
    return out


def _arrow_groups(F, i, lo, hi, role, w):
    """带 [lo, hi] 内本断面附近的箭头组数（进口往下游认 30 m，与 Viterbi 下限规则一致）。"""
    s_i = F["ss"][i]
    lo_s, hi_s = (s_i - 30.0, s_i + 4.0) if role == "approach" else (s_i - 4.0, s_i + 4.0)
    das = sorted(da for sa, da, _, _ in F["arrows"] if lo_s <= sa <= hi_s and lo + 0.4 <= da <= hi - 0.4)
    if not das:
        return 0
    return 1 + sum(1 for k in range(1, len(das)) if das[k] - das[k - 1] > 0.52 * w)


def ribbon_partition(F, role, min_cover=0.5):
    """返回 (K_seq, B_cand, SN_cand, info)；本车行道内部标线覆盖不足时返回 None（交还 Viterbi）。"""
    M = F["M"]
    d_R, d_L, w = F["d_R"], F["d_L"], F["w_cw"]
    chains = mark_chains(F)
    # 内部分界：链在本断面落在 (d_R + pad, d_L - pad) 内
    div = [[] for _ in range(M)]
    for ci, ch in enumerate(chains):
        for i in range(ch["i0"], ch["i1"] + 1):
            dv = ch["d"][i]
            if d_R[i] + _EDGE_PAD < dv < d_L[i] - _EDGE_PAD:
                div[i].append((dv, ci))
    covered = np.array([bool(x) for x in div])
    if covered.mean() < min_cover:
        return None
    # 相邻分界很近（< 0.45 m）是同一条线的重复观测：保留更长的链。
    # 更宽但容不下车道的带（导流三角、分合流渐变的楔形区）保留两侧分界，带内车道数为 0。
    clen = [ch["i1"] - ch["i0"] for ch in chains]
    for i in range(M):
        div[i].sort()
        keep = []
        for dv, ci in div[i]:
            if keep and dv - keep[-1][0] < 0.45:
                if clen[ci] > clen[keep[-1][1]]:
                    keep[-1] = (dv, ci)
                continue
            keep.append((dv, ci))
        div[i] = keep

    # 车道带身份 = (右侧链, 左侧链)，-1 / -2 表示车行道右 / 左边界。每个身份的车道数在其全程取一个值，
    # 只在宽度容不下时逐断面减少 —— 带内车道数不随宽度噪声抖动。
    rib = []
    for i in range(M):
        edges = [(d_R[i], -1)] + div[i] + [(d_L[i], -2)]
        rib.append([(edges[k][0], edges[k + 1][0], (edges[k][1], edges[k + 1][1])) for k in range(len(edges) - 1)])
    widths, groups = {}, {}
    for i in range(M):
        for lo, hi, key in rib[i]:
            widths.setdefault(key, []).append(hi - lo)
            groups[key] = max(groups.get(key, 0), _arrow_groups(F, i, lo, hi, role, w))
    n_key = {}
    for key, ws in widths.items():
        wm = float(np.median(ws))
        n = 1 if wm <= _WIDE_K * w else max(1, int(round(wm / w)))
        n_key[key] = max(n, groups.get(key, 0))

    K_seq = np.zeros(M, int)
    B_cand = [None] * M
    SN_cand = [None] * M
    for i in range(M):
        # 每条带的车道数：身份全程一致的 n_key，宽度容不下时减少；窄于 _GORE_K × 车道宽的带为 0（楔形区），
        # 其宽度并入相邻车道（车道边界取带的中线），车道中心不会落在楔形区两侧的标线上。
        segs = []
        for lo, hi, key in rib[i]:
            wb = hi - lo
            n = 0 if wb < _GORE_K * w else max(1, min(n_key[key], int(wb // _MIN_LANE_W)))
            segs.append([lo, hi, n])
        lanes = [(lo, hi, n) for lo, hi, n in segs if n > 0]
        if not lanes:  # 整幅都窄：一条车道
            lanes = [(segs[0][0], segs[-1][1], 1)]
        # 两条车道带之间隔着 0 车道带时，共用边界取间隙中线；外侧的 0 车道带不归任何车道
        b, sn = [lanes[0][0]], [True]
        for k, (lo, hi, n) in enumerate(lanes):
            if k > 0 and lo - b[-1] > 1e-6:
                b[-1] = 0.5 * (b[-1] + lo); sn[-1] = False
            for j in range(1, n):
                b.append(lo + (hi - lo) * j / n); sn.append(False)
            b.append(hi); sn.append(True)
        K = len(b) - 1
        K_seq[i] = K
        # 与 _viterbi_partition 同构：B_cand[i][K]；其余 K 置 None
        row_b = [None] * (K + 1)
        row_s = [None] * (K + 1)
        row_b[K] = np.asarray(b, float)
        row_s[K] = np.asarray(sn, bool)
        B_cand[i], SN_cand[i] = row_b, row_s
    K_seq = _clean_runs(K_seq, B_cand, SN_cand, F, role)
    info = dict(n_chains=len(chains), cover=float(covered.mean()))
    return K_seq, B_cand, SN_cand, info


def _clean_runs(K_seq, B_cand, SN_cand, F, role, min_end=4, min_mid=5):
    """车道数短毛刺段（中间 < 10 m、端头 < 8 m）并入较长的相邻段：把相邻段的分界按端点平移过来。"""
    M = len(K_seq)
    changed = True
    while changed:
        changed = False
        runs, s0 = [], 0
        for i in range(1, M + 1):
            if i == M or K_seq[i] != K_seq[s0]:
                runs.append((s0, i, int(K_seq[s0])))
                s0 = i
        if len(runs) <= 1:
            break
        for r, (a, b, kv) in enumerate(runs):
            is_end = r == 0 or r == len(runs) - 1
            if (b - a) >= (min_end if is_end else min_mid):
                continue
            if r == 0:
                src = runs[1]
            elif r == len(runs) - 1:
                src = runs[-2]
            else:
                src = max(runs[r - 1], runs[r + 1], key=lambda x: x[1] - x[0])
            ks = src[2]
            ref = src[0] if src[0] >= b else src[1] - 1  # 相邻段离本段最近的断面
            rb, rs = B_cand[ref][ks], SN_cand[ref][ks]
            if rb is None:
                continue
            for i in range(a, b):
                # 相邻段的相对分界比例，套到本断面的边界上
                frac = (rb - rb[0]) / max(rb[-1] - rb[0], 1e-6)
                lo, hi = F["d_R"][i], F["d_L"][i]
                row_b = [None] * (ks + 1)
                row_s = [None] * (ks + 1)
                row_b[ks] = lo + frac * (hi - lo)
                row_s[ks] = np.zeros(ks + 1, bool)
                row_s[ks][0] = row_s[ks][-1] = True
                B_cand[i] = _pad(B_cand[i], ks, row_b[ks])
                SN_cand[i] = _pad(SN_cand[i], ks, row_s[ks])
                K_seq[i] = ks
            changed = True
            break
    return K_seq


def _pad(row, K, val):
    row = list(row) + [None] * max(0, K + 1 - len(row))
    row[K] = val
    return row


# ---------------------------------------------------------------- 标线导向的 Frenet 参考轴

def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def mark_axis(axis, poly, B, min_piece=8.0, min_cover=0.6):
    """用车行道内最长的纵向标线（及其共线续接段）做 Frenet 参考轴，两端沿切向直线外延到原轴两端的投影处。
    Step 2 的轴线跟随多边形中线，在展宽、导流岛开口处会出现 20–30° 的 S 形摆动，法线随之旋转，
    d 空间里的车道就斜穿真实标线。标线本身与车道平行，用它做参考轴，平行标线的 d 基本恒定。
    覆盖不足原轴长 min_cover 时返回 None（沿用原轴）。"""
    import shapely
    from shapely.geometry import LineString, Point
    inner = poly.buffer(-0.5)
    if inner.is_empty:
        return None
    a0 = np.asarray(axis.coords[0]); a1 = np.asarray(axis.coords[-1])
    t_ax = _unit(a1 - a0)
    pieces = []
    for l, t in B["bounds"]:
        if t not in ("dash", "white", "yellow", "bus") or not l.intersects(inner):
            continue
        x = l.intersection(inner)
        for q in getattr(x, "geoms", [x]):
            if q.geom_type != "LineString" or q.length < min_piece:
                continue
            c = np.asarray(q.coords)[:, :2]
            if float(np.dot(c[-1] - c[0], t_ax)) < 0:
                c = c[::-1]
            # 只要与原轴大致同向（< 35°）的纵向标线
            if float(np.dot(_unit(c[-1] - c[0]), t_ax)) < 0.82:
                continue
            pieces.append(c)
    if not pieces:
        return None
    pieces.sort(key=lambda c: -LineString(c).length)
    ref = pieces[0]
    used = {0}
    changed = True
    while changed:  # 前后续接共线段（虚线 → 停止线前的实线等）
        changed = False
        for k, c in enumerate(pieces):
            if k in used:
                continue
            te = _unit(ref[-1] - ref[max(0, len(ref) - 4)])
            ts = _unit(ref[min(len(ref) - 1, 3)] - ref[0])
            for end, tt, sign in ((ref[-1], te, 1), (ref[0], ts, -1)):
                p = c[0] if sign > 0 else c[-1]
                v = p - end
                along = float(np.dot(v, tt)) * sign
                lat = abs(float(tt[0] * v[1] - tt[1] * v[0]))
                ang = float(np.dot(_unit(c[-1] - c[0]), tt))
                if -1.0 <= along <= 25.0 and lat <= 0.8 + 0.02 * max(along, 0) and ang > 0.97:
                    ref = np.vstack([ref, c]) if sign > 0 else np.vstack([c, ref])
                    used.add(k); changed = True
                    break
            if changed:
                break
    ref_line = LineString(ref)
    s0, s1 = float(np.dot(a0 - ref[0], t_ax)), float(np.dot(a1 - ref[-1], t_ax))
    cover = ref_line.length / max(axis.length, 1e-6)
    if cover < min_cover:
        return None
    ts = _unit(ref[min(len(ref) - 1, 3)] - ref[0])
    te = _unit(ref[-1] - ref[max(0, len(ref) - 4)])
    pts = list(ref)
    if s0 < -0.5:
        pts.insert(0, ref[0] + ts * s0)
    if s1 > 0.5:
        pts.append(ref[-1] + te * s1)
    out = LineString(pts)
    return out if out.is_valid and out.length >= 6.0 else None
