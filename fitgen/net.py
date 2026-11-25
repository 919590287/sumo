"""拟合模型 → SUMO。

拓扑完全由模型构造：每条车行道每侧按车道数分段成边链；路口一个中心节点；进口车道按箭头
（缺箭头按几何可达的转向默认分配）连到出口车道；停止线前就分流走的右转车道（pre_mouth）
在可行驶面里绕开路口核心找通道，建成独立的渠化右转边，从分流点接到目标出口的汇入点。
"""
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, MultiPoint, Point

from lanegen.sumo_check import _tool
from .fit import lane_u, match_lanes, _frame
from .legs import smooth_line

IDEAL = {"S": 0.0, "L": 90.0, "R": -90.0, "T": 180.0, "l": 45.0, "r": -45.0}
TOL = {"S": 35.0, "L": 55.0, "R": 55.0, "T": 40.0, "l": 30.0, "r": 30.0}


# ------------------------------------------------------------------ 几何

def _pt(f, i, u):
    sc = f["secs"][i]
    return sc["p"] + sc["n"] * u


def side_runs(f, key):
    """本侧从口门往外的分段：[(i_a, i_b, n, w)]，i_a..i_b 含两端断面（相邻段共享边界断面）。"""
    side = f[key]
    i0, T = side["i0"], len(f["secs"])
    n = side["n"]
    if n[i0] == 0:
        return []
    runs, a = [], i0
    for i in range(i0 + 1, T):
        if n[i] != n[i - 1]:
            runs.append((a, i, int(n[a]), float(np.mean(side["w"][a:i]))))
            a = i
    runs.append((a, T - 1, int(n[a]), float(np.mean(side["w"][a:T]))))
    # 太短的末段（< 3 个断面）并进前一段
    if len(runs) > 1 and runs[-1][1] - runs[-1][0] < 3:
        x = runs.pop()
        runs[-1] = (runs[-1][0], x[1], runs[-1][2], runs[-1][3])
    return runs


def group_center(f, key, i, n=None):
    side = f[key]
    n = int(side["n"][i]) if n is None else n
    return side["a"][i] + side["sigma"] * n * side["w"][i] / 2


def heading(f, key):
    """口门处行驶方向（单位向量）：进口指向路口，出口指离路口。"""
    side = f[key]
    _, t, _ = _frame(f["axis"], f["secs"][side["i0"]]["s"] + 3)
    return -t if key == "in" else t


def turn_angle(h_in, h_out):
    return float(np.degrees(np.arctan2(h_in[0] * h_out[1] - h_in[1] * h_out[0], h_in @ h_out)))


# ------------------------------------------------------------------ 转向

def exit_for(fits, fa, letter):
    """进口 fa 的转向字母 → 目标出口（角度最接近理想值、在容差内）。"""
    h = heading(fa, "in")
    best = None
    for fb in fits:
        if fb["out"]["n"][fb["out"]["i0"]] == 0:
            continue
        ang = turn_angle(h, heading(fb, "out"))
        if letter == "T":
            d = 180.0 - abs(ang)
        else:
            if fb is fa:
                continue
            d = abs(ang - IDEAL[letter])
        if d <= TOL[letter] and (best is None or d < best[0]):
            best = (d, fb)
    return None if best is None else best[1]


def default_moves(n, avail):
    """无箭头进口：左到右依次 L、S、R 的可用转向分块；车道少于转向数时共用。"""
    order = [m for m in "LSR" if m in avail]
    if not order:
        return [""] * n
    if n == 1:
        return ["".join(order)]
    if n < len(order):
        return ["LS", "SR"][:n] if order == list("LSR") else ["".join(order)] * n
    mv = []
    extra = n - len(order)
    for m in order:
        k = 1 + (extra if m == "S" else 0) if "S" in order else 1 + (extra // 2 if m == "L" else extra - extra // 2)
        mv += [m] * k
    return mv[:n]


def lane_moves(fits, f):
    n = int(f["in"]["n"][f["in"]["i0"]])
    avail = {m for m in "LSR" if exit_for(fits, f, m) is not None}
    known = [set(m) for m in f.get("lane_mv", [])] + [set()] * (n - len(f.get("lane_mv", [])))
    if not any(known):
        mv = default_moves(n, avail)
        f["mv_src"] = "默认（无箭头）"
        return mv
    # 部分车道缺箭头：沿用相邻已知车道
    mv = ["".join(sorted(k)) for k in known]
    for k in range(n):
        if not mv[k]:
            left = next((mv[j] for j in range(k - 1, -1, -1) if mv[j]), "")
            right = next((mv[j] for j in range(k + 1, n) if mv[j]), "")
            mv[k] = "S" if ("S" in left and "S" in right) else (left or right)
    f["mv_src"] = "箭头"
    return mv


# ------------------------------------------------------------------ 渠化右转通道

class PathFinder:
    """可行驶栅格上的最短路（8 邻域）。"""

    def __init__(self, mask, xs, ys):
        self.mask, self.xs, self.ys = mask, xs, ys
        self.res = float(xs[1] - xs[0])
        H, W = mask.shape
        idx = -np.ones(mask.shape, int)
        idx[mask] = np.arange(mask.sum())
        rows, cols, wts = [], [], []
        for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5)):
            r0, r1 = max(0, -dr), H - max(0, dr)
            c0, c1 = max(0, -dc), W - max(0, dc)
            a = idx[r0:r1, c0:c1]
            b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ok = (a >= 0) & (b >= 0)
            rows.append(a[ok]); cols.append(b[ok]); wts.append(np.full(ok.sum(), w))
        r = np.concatenate(rows); c = np.concatenate(cols); w = np.concatenate(wts)
        n = int(mask.sum())
        self.G = csr_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(n, n))
        rr, cc = np.nonzero(mask)
        self.cell_xy = np.stack([xs[cc], ys[rr]], 1)

    def _cell(self, p):
        d = np.hypot(*(self.cell_xy - p).T)
        k = int(np.argmin(d))
        return k if d[k] < 2.5 else None

    def path(self, p, targets, detour=3.0, sigma=4.0):
        """p → targets（按顺序的候选点）中第一个可达且不绕远的点；返回 (目标序号, 折线)。"""
        src = self._cell(np.asarray(p))
        if src is None:
            return None
        dist, pred = dijkstra(self.G, indices=src, return_predecessors=True, directed=False)
        for j, q in enumerate(targets):
            t = self._cell(np.asarray(q))
            if t is None or not np.isfinite(dist[t]):
                continue
            L = dist[t] * self.res
            if L > detour * np.hypot(*(np.asarray(q) - p)) + 20:
                continue
            seq = [t]
            while seq[-1] != src:
                seq.append(pred[seq[-1]])
            P = self.cell_xy[seq[::-1]].copy()
            P[0], P[-1] = p, q
            return j, smooth_line(LineString(P), sigma=sigma, step=1.0)
        return None


def SlipFinder(grid, core):
    """渠化右转通道：可行驶栅格扣掉路口核心（只能绕岛外侧走）。"""
    X, Y = np.meshgrid(grid["xs"], grid["ys"])
    mask = grid["mask"] & ~shapely.contains_xy(core, X, Y)
    return PathFinder(mask, grid["xs"], grid["ys"])


def reroute_blocked(netd, d, ev, grid, clearance=1.0):
    """编译后检查：路口内部连接（netconvert 自由画的曲线）落进安全岛/导流区的，改用可行驶面里
    离障碍至少 clearance 米的最短路作为该连接的自定义形状（SUMO connection shape），由“先找通道再成线”保证不穿岛。"""
    from scipy import ndimage
    nd = ev.nondrive
    if nd is None or nd.is_empty:
        return 0
    ndb = nd.buffer(-0.5)
    root = ET.parse(os.path.join(d, "net.net.xml")).getroot()
    lane_shape, lane_end = {}, {}
    for e in root.iter("edge"):
        for ln in e.iter("lane"):
            P = np.array([[float(v) for v in q.split(",")[:2]] for q in ln.get("shape").split()])
            lane_shape[ln.get("id")] = P
    nxt = {}   # 内部车道 → 下一段内部车道（经内部路口拆开的连接）
    conns = []
    for c in root.iter("connection"):
        if c.get("from").startswith(":"):
            if c.get("via"):
                nxt[f'{c.get("from")}_{c.get("fromLane")}'] = c.get("via")
        elif c.get("via"):
            conns.append(c)
    todo = []
    for c in conns:
        chain, v = [], c.get("via")
        while v and len(chain) < 6:
            chain.append(v)
            v = nxt.get(v)
        shp = np.vstack([lane_shape[v] for v in chain if v in lane_shape])
        if len(shp) < 2 or LineString(shp).intersection(ndb).length <= 0.3:
            continue
        a, la, b, lb = c.get("from"), int(c.get("fromLane")), c.get("to"), int(c.get("toLane"))
        todo.append((a, la, b, lb, lane_shape[f"{a}_{la}"][-1], lane_shape[f"{b}_{lb}"][0]))
    if not todo:
        return 0
    m = grid["mask"]
    er = ndimage.binary_erosion(m, iterations=max(1, int(round(clearance / (grid["xs"][1] - grid["xs"][0])))))
    pf = PathFinder(er, grid["xs"], grid["ys"])
    netd.setdefault("con_shapes", {})
    n = 0
    for a, la, b, lb, p, q in todo:
        for sg in (3.0, 1.5, 0.5):       # 平滑会把急弯（掉头绕岛头）切回岛里：逐级减小平滑直到不压岛
            r = pf.path(p, [q], detour=4.0, sigma=sg)
            if r is None:
                break
            if r[1].intersection(ndb).length <= 0.3:
                netd["con_shapes"][(a, la, b, lb)] = np.asarray(r[1].coords)
                n += 1
                break
    return n


def core_polygon(fits):
    pts = []
    for f in fits:
        for key in ("in", "out"):
            side = f[key]
            i0 = side["i0"]
            if side["n"][i0] == 0:
                continue
            a = side["a"][i0]
            b = a + side["sigma"] * side["n"][i0] * side["w"][i0]
            pts += [_pt(f, i0, a), _pt(f, i0, b)]
    return MultiPoint(pts).convex_hull


# ------------------------------------------------------------------ 编译

def build(fits, center, grid, ev):
    nodes = {"J": dict(x=center[0], y=center[1], type="traffic_light" if ev.stops else "priority")}
    edges, cons = {}, []
    chains = {}       # (fid, key) → [(edge_id, i_a, i_b, n)]  approach: 外→内；exit: 内→外
    notes = []

    def add_edge(eid, frm, to, shape, n, w, kind):
        edges[eid] = dict(id=eid, frm=frm, to=to, shape=shape, n=n, w=w, kind=kind)

    # 路口节点形状 = 各口门断面围成的多边形：边在停止线/人行横道处结束，内部连接只在这片冲突区里
    # （不给形状时 netconvert 会把大斜交口的进口边往回切十几米）
    core = core_polygon(fits)
    nodes["J"]["shape"] = [tuple(xy) for xy in np.asarray(core.exterior.coords)[:-1]]
    core = core.buffer(1.0)
    slips = []
    sf = None
    for f in fits:
        pm = [p for p in f.get("pre_mouth", []) if set(p["mv"]) & set("Rr")]
        if not pm:
            continue
        if sf is None:
            sf = SlipFinder(grid, core)
        p0 = min(pm, key=lambda p: p["s"])
        side = f["in"]
        i_lane, k_lane = p0["i"], p0["k"]
        # 分流点：往口门方向找该车道最后存在的断面
        i_div = i_lane
        while i_div - 1 >= side["i0"] and match_lanes(side, i_div, i_div - 1)[k_lane] >= 0:
            k_lane = match_lanes(side, i_div, i_div - 1)[k_lane]
            i_div -= 1
        fb = exit_for(fits, f, "R")
        if fb is None:
            notes.append(f"{f['id']} 右转车道在停止线前分流，但找不到右转出口")
            continue
        sb = fb["out"]
        nb = sb["n"]
        tg = []
        for j in range(sb["i0"], len(fb["secs"])):
            if nb[j] == 0:
                continue
            tg.append((j, _pt(fb, j, lane_u(sb, j, int(nb[j]) - 1))))
        start = _pt(f, i_div, lane_u(side, i_div, k_lane))
        r = sf.path(start, [q for _, q in tg[2:]])
        if r is None:
            notes.append(f"{f['id']} 右转车道在停止线前分流，可行驶面里找不到绕开路口核心的通道 → 改为在停止线右转")
            continue
        j_merge = tg[2 + r[0]][0]
        slips.append(dict(fa=f, i_div=i_div, k=k_lane, fb=fb, j_merge=j_merge, shape=r[1]))
        f.setdefault("force_cut", set()).add(i_div)
        fb.setdefault("force_cut_out", set()).add(j_merge)
        f["slip_R"] = True

    for f in fits:
        fid = f["id"]
        for key in ("in", "out"):
            runs = side_runs(f, key)
            if not runs:
                continue
            cuts = f.get("force_cut" if key == "in" else "force_cut_out", set())
            if cuts:
                new = []
                for a, b, n, w in runs:
                    pts = [a] + sorted(c for c in cuts if a < c < b) + [b]
                    for x, y in zip(pts[:-1], pts[1:]):
                        new.append((x, y, n, w))
                runs = new
            side = f[key]
            chain = []
            for j, (a, b, n, w) in enumerate(runs):
                idx = list(range(a, b + 1))
                pts = [_pt(f, i, group_center(f, key, i, n)) for i in idx]
                if key == "in":
                    pts = pts[::-1]
                    frm = f"{fid}i{j + 1}" if j + 1 < len(runs) else f"{fid}iE"
                    to = "J" if j == 0 else f"{fid}i{j}"
                else:
                    frm = "J" if j == 0 else f"{fid}o{j}"
                    to = f"{fid}o{j + 1}" if j + 1 < len(runs) else f"{fid}oE"
                for nd, q in ((frm, pts[0]), (to, pts[-1])):
                    if nd != "J" and nd not in nodes:
                        nodes[nd] = dict(x=q[0], y=q[1], type="priority")
                eid = f"{fid}_{key}{j}"
                add_edge(eid, frm, to, pts, n, w, key)
                chain.append((eid, a, b, n))
            chains[(fid, key)] = chain
            # 段间车道连接（按横向位置对齐）
            for (e1, a1, b1, n1), (e2, a2, b2, n2) in zip(chain[:-1], chain[1:]):
                # 进口：e2 在外、e1 在内（e2 → e1）；出口：e1 → e2。边界断面 b1 == a2
                ib = a2
                if key == "in":
                    up, dn, nu, nd_ = e2, e1, n2, n1
                    m = _match_counts(side, ib, nu, nd_, ib + 1, ib - 1)
                else:
                    up, dn, nu, nd_ = e1, e2, n1, n2
                    m = _match_counts(side, ib, nu, nd_, ib - 1, ib + 1)
                for ku, kd in m:
                    cons.append((up, nu - 1 - ku, dn, nd_ - 1 - kd))

    # 渠化右转边与分合流连接
    for k, sl in enumerate(slips):
        fa, fb = sl["fa"], sl["fb"]
        ca, cb = chains[(fa["id"], "in")], chains[(fb["id"], "out")]
        up = next(c for c in ca if c[1] == sl["i_div"])           # 分流点外侧的边（从 i_div 往外）
        dn = next(c for c in ca if c[2] == sl["i_div"])           # 分流点内侧的边
        bu = next(c for c in cb if c[2] == sl["j_merge"])         # 汇入点内侧的出口边
        bd = next(c for c in cb if c[1] == sl["j_merge"])         # 汇入点外侧的出口边
        frm = edges[up[0]]["to"]
        to = edges[bd[0]]["frm"]
        sid = f"S{k}_{fa['id']}_{fb['id']}"
        add_edge(sid, frm, to, list(np.asarray(sl["shape"].coords)), 1, 3.3, "slip")
        nu = up[3]
        # 分流：进口外侧边上该车道 → 渠化边；其余车道照常（已在段间连接里），但要去掉该车道原来的并入
        ku = sl["k"]
        cons[:] = [c for c in cons if not (c[0] == up[0] and c[1] == nu - 1 - ku and c[2] == dn[0])]
        cons.append((up[0], nu - 1 - ku, sid, 0))
        cons.append((sid, 0, bd[0], 0))

    # 路口内连接
    jcons = []
    for fa in fits:
        if fa["in"]["n"][fa["in"]["i0"]] == 0:
            continue
        ein = chains[(fa["id"], "in")][0]
        n = ein[3]
        mv = lane_moves(fits, fa)
        if fa.get("slip_R"):
            mv = [m.replace("R", "").replace("r", "") or m for m in mv]
        fa["lane_mv_final"] = mv
        by = {}
        for k, m in enumerate(mv):
            for letter in m:
                by.setdefault(letter, []).append(k)
        for letter, ks in by.items():
            fb = exit_for(fits, fa, letter)
            if fb is None:
                notes.append(f"{fa['id']} 车道 {ks} 的 {letter} 没有对应出口")
                continue
            eout = chains[(fb["id"], "out")][0]
            nb = eout[3]
            if letter in "LlT":
                pairs = [(k, min(j, nb - 1)) for j, k in enumerate(ks)]
            elif letter in "Rr":
                pairs = [(k, max(0, nb - 1 - j)) for j, k in enumerate(reversed(ks))]
            else:
                pairs = [(k, max(0, nb - 1 - j)) for j, k in enumerate(reversed(ks))]
            for ka, kb in pairs:
                jcons.append((ein[0], n - 1 - ka, eout[0], nb - 1 - kb))
    starts = [chains[(f["id"], "in")][-1][0] for f in fits if (f["id"], "in") in chains]
    ends = [chains[(f["id"], "out")][-1][0] for f in fits if (f["id"], "out") in chains]
    return dict(nodes=nodes, edges=edges, cons=cons + jcons, starts=starts, ends=ends, notes=notes,
                slips=len(slips))


def _match_counts(side, ib, nu, nd, iu, idn):
    """边界断面 ib 两侧：上游 nu 条、下游 nd 条车道的对应（从中线数的序号）。
    按两侧邻近断面的横向位置匹配；上游没对上的并到最近的下游车道，下游没被喂的由最近的上游车道喂。"""
    T = len(side["n"])
    iu = int(np.clip(iu, 0, T - 1)); idn = int(np.clip(idn, 0, T - 1))
    uu = np.array([side["a"][iu] + side["sigma"] * (k + 0.5) * side["w"][iu] for k in range(nu)])
    ud = np.array([side["a"][idn] + side["sigma"] * (k + 0.5) * side["w"][idn] for k in range(nd)])
    pairs = set()
    for k in range(nu):
        pairs.add((k, int(np.argmin(np.abs(ud - uu[k])))))
    fed = {kd for _, kd in pairs}
    for kd in range(nd):
        if kd not in fed:
            pairs.add((int(np.argmin(np.abs(uu - ud[kd]))), kd))
    return sorted(pairs)


def write_plain(netd, d):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "n.nod.xml"), "w", encoding="utf-8") as fo:
        fo.write("<nodes>\n")
        for nid, n in netd["nodes"].items():
            shp = ""
            if n.get("shape"):
                shp = ' shape="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in n["shape"]) + '"'
            fo.write(f'  <node id="{nid}" x="{n["x"]:.2f}" y="{n["y"]:.2f}" type="{n["type"]}"{shp}/>\n')
        fo.write("</nodes>\n")
    with open(os.path.join(d, "e.edg.xml"), "w", encoding="utf-8") as fo:
        fo.write("<edges>\n")
        for e in netd["edges"].values():
            shp = " ".join(f"{x:.2f},{y:.2f}" for x, y in e["shape"])
            fo.write(f'  <edge id="{e["id"]}" from="{e["frm"]}" to="{e["to"]}" numLanes="{e["n"]}" speed="13.89" '
                     f'spreadType="center" shape="{shp}">\n')
            for k in range(e["n"]):
                fo.write(f'    <lane index="{k}" width="{float(np.clip(e["w"], 2.5, 4.5)):.2f}"/>\n')
            fo.write("  </edge>\n")
        fo.write("</edges>\n")
    with open(os.path.join(d, "c.con.xml"), "w", encoding="utf-8") as fo:
        fo.write("<connections>\n")
        shapes = netd.get("con_shapes", {})
        for a, la, b, lb in sorted(set(netd["cons"])):
            extra = ""
            if (a, la, b, lb) in shapes:
                extra = ' shape="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in shapes[(a, la, b, lb)]) + '"'
            fo.write(f'  <connection from="{a}" to="{b}" fromLane="{la}" toLane="{lb}"{extra}/>\n')
        fo.write("</connections>\n")


def compile_net(netd, d):
    write_plain(netd, d)
    cmd = [_tool("netconvert"), "-n", "n.nod.xml", "-e", "e.edg.xml", "-x", "c.con.xml", "-o", "net.net.xml",
           "--offset.disable-normalization", "true", "--no-turnarounds", "true", "--geometry.remove", "false",
           "--junctions.join", "false", "--tls.guess", "false", "--edges.join", "false"]
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=120)
    warns = [l for l in (p.stderr + p.stdout).splitlines() if l.startswith("Warning")]
    return p.returncode == 0 and os.path.exists(os.path.join(d, "net.net.xml")), warns, p.stderr


def net_lines(net_path):
    import sumolib
    net = sumolib.net.readNet(net_path, withInternal=True)
    normal, internal = [], []
    for e in net.getEdges(withInternal=True):
        for ln in e.getLanes():
            (internal if e.getFunction() == "internal" else normal).append(np.asarray(ln.getShape()))
    return normal, internal


def obstacle_hits(lines, ev, shrink=0.5):
    """车道中心线落进安全岛/导流区（内收 shrink 米）的部分。"""
    nd = ev.nondrive
    if nd is None or nd.is_empty:
        return [], 0.0
    nd = nd.buffer(-shrink)
    bad, tot = [], 0.0
    for s in lines:
        if len(s) < 2:
            continue
        x = LineString(s).intersection(nd)
        for g in getattr(x, "geoms", [x]):
            if g.geom_type == "LineString" and g.length > 0.3:
                bad.append(np.asarray(g.coords))
                tot += g.length
    return bad, tot
