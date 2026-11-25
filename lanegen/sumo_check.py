"""Step 4 雏形 + SUMO 闭环评测：把 Step 3 契约（junction_model + lanes）编译成 SUMO plain XML，
netconvert 成网，按所有可达的进口→出口组合各放一辆车仿真，检查物理可行性。

与 lanegen 其余部分的启发式不同源，是“不依赖人工真值”的独立判据：
  net_ok          netconvert 成功
  net_warnings    netconvert 警告条数
  mv_cover        进口车道上导向箭头的每个转向（含掉头）是否都有到目标臂出口的连接
  dead_lanes      非末端边上没有任何后继连接的车道数（车开进去出不来）
  unfed_lanes     非源头边上没有任何来车连接的车道数（车进不去，如展宽车道漏写分流）
  turn_cross      同一进口内转向交叉的车道对数（右侧车道的转向比左侧车道更靠左）
  mv_unserved     有目标臂却没有任何车道承担的 左/直/右 转向数
  exit_fed        路口出口车道中有来车连接的比例
  route_pairs     有连接路径的 进口远端→出口远端 组合数；route_arrived 到达数
  teleports / collisions  仿真瞬移、碰撞次数

连接（转向车道分配）默认用路口拓扑联合求解 turns.cpsat_connections，heuristic_connections 作对照与兜底。

python -m lanegen.sumo_check <经度> <纬度> [输出目录]
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from .frenet import _ax, _lateral, _tan
from .junction import movement

_BIN = os.path.dirname(sys.executable)


def _tool(name):
    p = os.path.join(_BIN, name)
    return p if os.path.exists(p) else (shutil.which(name) or name)


# ---------------------------------------------------------------- 转向语义

_IDEAL = {"S": 0.0, "R": 90.0, "L": 270.0, "r": 45.0, "l": 315.0}
_SECTOR = {"S": ((0.0, 40.0), (320.0, 360.0)), "R": ((20.0, 160.0),), "L": ((200.0, 340.0),),
           "r": ((10.0, 90.0),), "l": ((270.0, 350.0),)}


def _bearing(v):
    return (math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0


def target_arm(jm, from_arm, heading, ch):
    """进口行驶方位 heading 下转向 ch 的目标臂：扇区内、离理想角最近的有出口车行道的臂；掉头 T 回本臂。"""
    if ch == "T":
        return from_arm
    best = None
    for a in jm["arms"]:
        if a["id"] == from_arm or not a["out_carriageways"]:
            continue
        d = (a["bearing"] - heading + 360.0) % 360.0
        if not any(lo <= d <= hi for lo, hi in _SECTOR[ch]):
            continue
        err = min(abs(d - _IDEAL[ch]), 360.0 - abs(d - _IDEAL[ch]))
        if best is None or err < best[0]:
            best = (err, a["id"])
    return best[1] if best else None


def lane_movements(ln, idx, n):
    """车道的转向集合：取离停止线最近的箭头；没有箭头按位置推断（最左 L、最右 R、都含 S；单车道全有）。
    返回 (转向集合, 是否来自箭头)。"""
    arr = sorted(ln.get("arrows") or [])
    if arr:
        return set(arr[0][1]), True
    if n == 1:
        return {"L", "S", "R"}, False
    if idx == 0:
        return {"S", "R"}, False
    if idx == n - 1:
        return {"L", "S"}, False
    return {"S"}, False


# ---------------------------------------------------------------- 契约 → 边链

def _section_at(secs, s):
    for sec in secs:
        if sec["s_start"] - 1e-6 <= s <= sec["s_end"] + 1e-6:
            return sec
    return secs[-1] if s > secs[-1]["s_end"] else secs[0]


def _edge_shape(axis, lanes, ids, a, b):
    """分段 [a,b] 的边中心线：各采样断面取最右、最左车道横向位置的中点。"""
    n = max(2, int(math.ceil((b - a) / 4.0)) + 1)
    pts = []
    for s in np.linspace(a, b, n):
        ds = [_lateral(axis, lanes[i]["geom"], float(s)) for i in ids]
        q = np.asarray(axis.interpolate(float(s)).coords[0])
        t = _tan(axis, float(s), 1.5)
        pts.append(q + np.array([-t[1], t[0]]) * 0.5 * (min(ds) + max(ds)))
    return pts


def _section_links(e1, p_ids, e2, ids):
    """段间节点的车道连接（两边车道都按 右→左 排列）。
    延续车道一一相连；消失车道并入最近的延续车道（zipper）；新增车道（展宽湾）从最近的延续车道
    分流进入——SUMO 里一条边一旦写了显式 connection，netconvert 就不再补其他连接，漏写会使新增车道入度为 0。
    两侧没有任何共同车道时按横向序号比例对齐。"""
    cont = {gi: ids.index(gi) for gi in p_ids if gi in ids}
    out = []
    if not cont:
        n1, n2 = len(p_ids), len(ids)
        for pi in range(n1):
            out.append((e1, pi, e2, min(n2 - 1, pi * n2 // n1)))
        for j in range(n2):
            out.append((e1, min(n1 - 1, j * n1 // n2), e2, j))
        return sorted(set(out))
    for pi, gi in enumerate(p_ids):
        if gi in cont:
            out.append((e1, pi, e2, cont[gi]))
        else:  # 车道减少：并入最近的延续车道
            pj = min((abs(pj - pi), pj) for pj, g2 in enumerate(p_ids) if g2 in cont)[1]
            out.append((e1, pi, e2, cont[p_ids[pj]]))
    back = {j: pi for pi, gi in enumerate(p_ids) if gi in cont for j in [cont[gi]]}
    for j, gi in enumerate(ids):
        if gi in cont.keys() or j in back:
            continue
        jj = min((abs(jj - j), jj) for jj in back)[1]  # 车道新增：从最近的延续车道分流
        out.append((e1, back[jj], e2, j))
    return sorted(set(out))


def build_network(res):
    """返回 dict(nodes, edges, cons, approach, exits, starts, ends)。
    edges[eid] = dict(frm, to, lanes(全局车道下标, 右→左), shape, cw, role)；cons = [(e1, l1, e2, l2)]。"""
    jm, lanes = res["model"], res["lanes"]
    core = jm["core_poly"]
    c = core.centroid
    nodes = {"J": (c.x, c.y, "traffic_light" if res.get("core_stops") else "priority")}
    edges, cons = {}, []
    approach, exits = {}, {}   # cw id → 靠路口的边 id
    starts, ends = [], []

    # 渠化右转在车行道上的分流 / 汇入切点
    extra_cut = {}
    slip_nodes = {}
    for s in jm["slips"]:
        if not s.get("lanes"):
            continue
        for key, pt, role, arm in (("d", s["diverge_pt"], "approach", s["from_arm"]), ("m", s["merge_pt"], "exit", s["to_arm"])):
            host = None
            for cw in jm["carriageways"]:
                if cw["role"] != role or cw["arm"] != arm or not cw["lanes"]:
                    continue
                ax = _ax(cw)
                dist = ax.distance(Point(pt))
                if dist < 15.0 and (host is None or dist < host[0]):
                    host = (dist, cw, float(np.clip(ax.project(Point(pt)), 3.0, ax.length - 3.0)))
            if host:
                extra_cut.setdefault(host[1]["id"], []).append((host[2], f"s{key}{s['id']}"))
                slip_nodes[(s["id"], key)] = (f"s{key}{s['id']}", host[1]["id"])
            else:
                slip_nodes[(s["id"], key)] = (f"s{key}{s['id']}", None)
                nodes[f"s{key}{s['id']}"] = (pt[0], pt[1], "priority")

    for cw in jm["carriageways"]:
        secs = cw.get("sections") or []
        if not cw["lanes"] or not secs:
            continue
        ax = _ax(cw)
        L = ax.length
        cuts = {0.0: None, L: None}
        for sec in secs[1:]:
            cuts[float(sec["s_start"])] = None
        for s_, nid in extra_cut.get(cw["id"], []):
            cuts[s_] = nid
        bs = []
        for s_ in sorted(cuts):
            if bs and s_ - bs[-1][0] < 1.0:
                if cuts[s_]:
                    bs[-1] = (bs[-1][0], cuts[s_])
                continue
            bs.append((s_, cuts[s_]))
        if bs[-1][0] < L - 0.5:
            bs.append((L, None))
        bs[-1] = (L, bs[-1][1])
        nids = []
        for k, (s_, nid) in enumerate(bs):
            if cw["role"] == "approach" and k == len(bs) - 1:
                nid = "J"
            elif cw["role"] == "exit" and k == 0:
                nid = "J"
            nid = nid or f"n{cw['id']}_{k}"
            if nid not in nodes:
                q = ax.interpolate(s_)
                nodes[nid] = (q.x, q.y, "priority")
            nids.append(nid)
        prev = None
        for k in range(len(bs) - 1):
            a, b = bs[k][0], bs[k + 1][0]
            sec = _section_at(secs, 0.5 * (a + b))
            ids = list(sec["lane_ids"])
            eid = f"c{cw['id']}_{k}"
            edges[eid] = dict(frm=nids[k], to=nids[k + 1], lanes=ids, shape=_edge_shape(ax, lanes, ids, a, b),
                              cw=cw["id"], role=cw["role"], sec=sec["section_id"])
            if prev is not None:
                p_ids = edges[prev]["lanes"]
                if p_ids == ids:
                    cons += [(prev, i, eid, i) for i in range(len(ids))]
                else:
                    cons += _section_links(prev, p_ids, eid, ids)
            prev = eid
        first, last = f"c{cw['id']}_0", f"c{cw['id']}_{len(bs) - 2}"
        if cw["role"] == "approach":
            approach[cw["id"]] = last
            starts.append(first)
        else:
            exits[cw["id"]] = first
            ends.append(last)

    # 渠化右转边
    for s in jm["slips"]:
        if not s.get("lanes"):
            continue
        (fn, fcw), (tn, tcw) = slip_nodes[(s["id"], "d")], slip_nodes[(s["id"], "m")]
        g = max((lanes[i]["geom"] for i in s["lanes"]), key=lambda g: g.length)
        eid = f"s{s['id']}"
        edges[eid] = dict(frm=fn, to=tn, lanes=list(s["lanes"]), shape=list(np.asarray(g.coords)), cw=None, role="slip")
        if fcw is None:
            starts.append(eid)
        else:  # 上游边右侧车道按序 → 绕行道车道（双车道右转匝道从 lane 0、1 分别分流）
            up = next(e for e, d in edges.items() if d["to"] == fn and d["cw"] == fcw)
            nu = len(edges[up]["lanes"])
            cons += [(up, min(j, nu - 1), eid, j) for j in range(len(s["lanes"]))]
        if tcw is None:
            ends.append(eid)
        else:  # 绕行道车道按序 → 下游边右侧车道
            dn = next(e for e, d in edges.items() if d["frm"] == tn and d["cw"] == tcw)
            nd = len(edges[dn]["lanes"])
            cons += [(eid, j, dn, min(j, nd - 1)) for j in range(len(s["lanes"]))]
    return dict(nodes=nodes, edges=edges, cons=cons, approach=approach, exits=exits, starts=starts, ends=ends)


# ---------------------------------------------------------------- 路口转向连接（启发式）

def junction_demands(res, netd):
    """每条进口车行道靠路口的边：[(cw, 边, [(车道序号, 转向集合, 来自箭头?)], 行驶方位)]。"""
    jm, lanes = res["model"], res["lanes"]
    out = []
    for cw in jm["carriageways"]:
        eid = netd["approach"].get(cw["id"])
        if eid is None:
            continue
        ids = netd["edges"][eid]["lanes"]
        ax = _ax(cw)
        heading = _bearing(_tan(ax, ax.length - 3.0, 3.0))
        mv = [(k,) + lane_movements(lanes[gi], k, len(ids)) for k, gi in enumerate(ids)]
        byp = slip_bypass(jm, cw["arm"])
        if byp and target_arm(jm, cw["arm"], heading, "R") in byp:
            # 渠化右转已承担本臂右转：无箭头时不再按位置推断主路口右转（否则会在核心内再建一条穿岛右转）
            mv = [(k, chs if obs else (chs - {"R", "r"}) or {"S"}, obs) for k, chs, obs in mv]
        out.append((cw, eid, mv, heading))
    return out


def slip_bypass(jm, from_arm):
    """从 from_arm 分流、已建出车道的渠化右转所到达的臂。"""
    return {s["to_arm"] for s in jm.get("slips", []) if s.get("lanes") and s["from_arm"] == from_arm}


def approach_options(jm, netd, cw, mv, heading):
    """进口的候选出口；进口没有观测到右转箭头时，渠化右转已到达的臂不再作为主路口右转出口。"""
    byp = slip_bypass(jm, cw["arm"])
    has_r = any(obs and chs & {"R", "r"} for _, chs, obs in mv)
    return exit_options(jm, netd, cw["arm"], heading, bypass=frozenset() if has_r else frozenset(byp))


@dataclass(frozen=True)
class ExitOption:
    edge: str
    angle: float
    lateral: float
    lanes: int
    movements: frozenset


def exit_options(jm, netd, from_arm, heading, bypass=frozenset()):
    """所有已建模出口按进口视角从右到左排序；同一箭头可以对应多个出口。
    bypass：已由渠化右转服务的目标臂，这些臂的出口去掉 R/r 语义。"""
    bearings = {a["id"]: a["bearing"] for a in jm["arms"]}
    options = []
    for cw in jm["carriageways"]:
        edge = netd["exits"].get(cw["id"])
        if edge is None:
            continue
        if cw["arm"] == from_arm:
            angle, moves = -180.0, {"T"}
        else:
            d = (bearings[cw["arm"]] - heading) % 360.0
            angle = (d + 180.0) % 360.0 - 180.0
            moves = {ch for ch, sector in _SECTOR.items() if any(lo <= d <= hi for lo, hi in sector)}
            if not moves:  # 发卡形异臂仍是候选，不归为回本臂的掉头
                moves = {"R" if angle > 0 else "L"}
            if cw["arm"] in bypass:
                moves -= {"R", "r"}
                if not moves:
                    continue
        options.append(ExitOption(edge, angle, cw.get("d_lat", 0.0),
                                  len(netd["edges"][edge]["lanes"]), frozenset(moves)))
    return sorted(options, key=lambda o: (-o.angle, o.lateral, o.edge))


def heuristic_connections(res, netd):
    """转向 → 目标臂出口边；车道按右转贴右、左转/掉头贴左、直行右对齐映射到出口车道。"""
    jm = res["model"]
    cons = []
    for cw, eid, mv, heading in junction_demands(res, netd):
        options = approach_options(jm, netd, cw, mv, heading)
        by_ch = {}
        for k, chs, _ in mv:
            for ch in chs:
                by_ch.setdefault(ch, []).append(k)
        for ch, ks in by_ch.items():
            ks = sorted(ks)
            opts = [o for o in options if ch in o.movements]
            if not opts:
                continue
            # 按左右顺序匹配两个有序序列，既支持分流也支持多车道去同一出口。
            size = max(len(opts), len(ks))
            for t in range(size):
                j = t * len(ks) // size
                o = opts[t * len(opts) // size]
                tl = min(j, o.lanes - 1)
                if ch in ("L", "l", "T"):
                    tl = max(0, o.lanes - len(ks) + j)
                cons.append((eid, ks[j], o.edge, tl))
    return sorted(set(cons))


# ---------------------------------------------------------------- 写文件、编译、仿真

def _write_plain(netd, jcons, d):
    nod = ET.Element("nodes")
    for nid, (x, y, t) in netd["nodes"].items():
        ET.SubElement(nod, "node", id=nid, x=f"{x:.2f}", y=f"{y:.2f}", type=t)
    edg = ET.Element("edges")
    return_widths = {}
    for eid, e in netd["edges"].items():
        el = ET.SubElement(edg, "edge", id=eid, attrib={"from": e["frm"], "to": e["to"]}, numLanes=str(len(e["lanes"])),
                           speed="13.89", spreadType="center",
                           shape=" ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in e["shape"]))
        return_widths[eid] = el
    con = ET.Element("connections")
    for e1, l1, e2, l2 in netd["cons"] + jcons:
        ET.SubElement(con, "connection", attrib={"from": e1, "to": e2, "fromLane": str(l1), "toLane": str(l2)})
    for name, root in (("n.nod.xml", nod), ("e.edg.xml", edg), ("c.con.xml", con)):
        ET.ElementTree(root).write(os.path.join(d, name), encoding="utf-8")


def compile_net(netd, jcons, d, lanes=None):
    _write_plain(netd, jcons, d)
    if lanes is not None:  # 车道宽度
        tree = ET.parse(os.path.join(d, "e.edg.xml"))
        for el in tree.getroot():
            for k, gi in enumerate(netd["edges"][el.get("id")]["lanes"]):
                ET.SubElement(el, "lane", index=str(k), width=f"{float(np.clip(lanes[gi]['width'], 2.5, 4.5)):.2f}")
        tree.write(os.path.join(d, "e.edg.xml"), encoding="utf-8")
    cmd = [_tool("netconvert"), "-n", "n.nod.xml", "-e", "e.edg.xml", "-x", "c.con.xml", "-o", "net.net.xml",
           "--offset.disable-normalization", "true", "--no-turnarounds", "true", "--geometry.remove", "false",
           "--junctions.join", "false", "--tls.guess", "false", "--edges.join", "false"]
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=120)
    warns = [l for l in (p.stderr + p.stdout).splitlines() if l.startswith("Warning")]
    return p.returncode == 0 and os.path.exists(os.path.join(d, "net.net.xml")), warns, p.stderr


def simulate(d, netd, max_pairs=60):
    import sumolib
    net = sumolib.net.readNet(os.path.join(d, "net.net.xml"), withInternal=False)
    pairs = []
    for a in netd["starts"]:
        if not net.hasEdge(a):
            continue
        for b in netd["ends"]:
            if not net.hasEdge(b):
                continue
            path, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b))
            if path:
                pairs.append(" ".join(e.getID() for e in path))
    pairs = pairs[:max_pairs]
    with open(os.path.join(d, "r.rou.xml"), "w", encoding="utf-8") as f:
        f.write('<routes>\n<vType id="car" length="4.5" minGap="2.0"/>\n')
        for k, r in enumerate(pairs):
            f.write(f'<vehicle id="v{k}" type="car" depart="{2 * k}" departLane="best" departSpeed="max">'
                    f'<route edges="{r}"/></vehicle>\n')
        f.write("</routes>\n")
    out = dict(route_pairs=len(pairs), route_arrived=0, teleports=0, collisions=0)
    if not pairs:
        return out
    cmd = [_tool("sumo"), "-n", "net.net.xml", "-r", "r.rou.xml", "--no-step-log", "true", "--duration-log.disable", "true",
           "--statistic-output", "stat.xml", "--collision.action", "warn", "--time-to-teleport", "90",
           "--end", str(2 * len(pairs) + 400), "--no-warnings", "true"]
    subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=300)
    sp = os.path.join(d, "stat.xml")
    if os.path.exists(sp):
        r = ET.parse(sp).getroot()
        vh = r.find("vehicles")
        tp = r.find("teleports")
        sf = r.find("safety")
        out["route_arrived"] = (int(vh.get("inserted", 0)) - int(vh.get("running", 0))) if vh is not None else 0
        out["teleports"] = int(tp.get("total", 0)) if tp is not None else 0
        out["collisions"] = int(sf.get("collisions", 0)) if sf is not None else 0
    return out


def coverage(res, netd, d):
    """转向覆盖（只算导向箭头给出的转向）与死胡同车道。"""
    import sumolib
    jm = res["model"]
    net = sumolib.net.readNet(os.path.join(d, "net.net.xml"), withInternal=False)
    exit_arm = {}
    for cw in jm["carriageways"]:
        for e, dd in netd["edges"].items():
            if dd["cw"] == cw["id"] and cw["role"] == "exit":
                exit_arm[e] = cw["arm"]
    tot = hit = candidate_hits = 0
    missing = []
    for cw, eid, mv, heading in junction_demands(res, netd):
        if not net.hasEdge(eid):
            continue
        lanes_e = net.getEdge(eid).getLanes()
        options = approach_options(jm, netd, cw, mv, heading)
        for k, chs, from_arrow in mv:
            if not from_arrow or k >= len(lanes_e):
                continue
            edges = {c.getTo().getID() for c in lanes_e[k].getOutgoing()}
            outs = {exit_arm.get(e) for e in edges}
            for ch in chs:
                tot += 1
                candidate_hits += any(o.edge in edges and ch in o.movements for o in options)
                ta = target_arm(jm, cw["arm"], heading, ch)
                if ta is not None and ta in outs:
                    hit += 1
                else:
                    missing.append((cw["id"], k, ch, ta))
    # 进口内转向交叉（右侧车道转向比左侧车道更靠左）、无人承担的转向、出口车道来车率
    arm_bear = {a["id"]: a["bearing"] for a in jm["arms"]}
    cross = unserved = 0
    fed, exit_lanes = set(), 0
    for cw, eid, mv, heading in junction_demands(res, netd):
        if not net.hasEdge(eid):
            continue
        tv_by_lane = []
        arms_served = set(slip_bypass(jm, cw["arm"]))  # 渠化右转承担的转向也算已服务
        for k, ln in enumerate(net.getEdge(eid).getLanes()):
            tvs = []
            for c in ln.getOutgoing():
                te = c.getTo().getID()
                fed.add((te, c.getToLane().getIndex()))
                ta = exit_arm.get(te)
                if ta is None:
                    continue
                arms_served.add(ta)
                d = (arm_bear[ta] - heading + 360.0) % 360.0
                tvs.append(-180.0 if ta == cw["arm"] else ((d + 180.0) % 360.0) - 180.0)
            tv_by_lane.append(tvs)
        for k1 in range(len(tv_by_lane)):
            for k2 in range(k1 + 1, len(tv_by_lane)):
                if tv_by_lane[k1] and tv_by_lane[k2] and min(tv_by_lane[k1]) < max(tv_by_lane[k2]) - 10.0:
                    cross += 1
        for ch in ("S", "L", "R"):
            ta = target_arm(jm, cw["arm"], heading, ch)
            if ta is not None and ta not in arms_served:
                unserved += 1
    for cw in jm["carriageways"]:
        te = netd["exits"].get(cw["id"])
        if te is not None and net.hasEdge(te):
            exit_lanes += len(net.getEdge(te).getLanes())
    sinks, sources = set(netd["ends"]), set(netd["starts"])
    dead = unfed = 0
    for e in net.getEdges():
        if e.getID() not in sinks:
            dead += sum(1 for ln in e.getLanes() if not ln.getOutgoing())
        if e.getID() not in sources:  # 非源头边上入度为 0 的车道（展宽新增车道漏写分流连接即属此类）
            unfed += sum(1 for ln in e.getLanes() if not ln.getIncoming())
    exit_fed = len({f for f in fed if f[0] in set(netd["exits"].values())}) / exit_lanes if exit_lanes else float("nan")
    return dict(mv_total=tot, mv_cover=(hit / tot) if tot else float("nan"), mv_missing=missing, dead_lanes=dead,
                candidate_mv_cover=(candidate_hits / tot) if tot else float("nan"),
                turn_cross=cross, mv_unserved=unserved, exit_fed=exit_fed, unfed_lanes=unfed)


def check(res, out_dir=None, conn="cpsat"):
    """对一次 pipeline.run 的结果做闭环检查，返回指标 dict。conn: heuristic / cpsat。"""
    jm = res["model"]
    m = dict(status=res.get("status"), status_reason=res.get("status_reason"), fallback_approaches=0,
             net_ok=False, net_warnings=0, mv_total=0, mv_cover=float("nan"), dead_lanes=0, route_pairs=0,
             route_arrived=0, teleports=0, collisions=0, uncompiled_lanes=len(jm.get("extra_lanes", [])))
    netd = build_network(res)
    if not netd["edges"]:
        return m
    if conn == "cpsat":
        from .turns import cpsat_connections
        jcons = cpsat_connections(res, netd, diagnostics=m)
    else:
        jcons = heuristic_connections(res, netd)
    tmp = out_dir or tempfile.mkdtemp(prefix="lanegen_sumo_")
    os.makedirs(tmp, exist_ok=True)
    try:
        ok, warns, err = compile_net(netd, jcons, tmp, res["lanes"])
        m["net_ok"], m["net_warnings"] = ok, len(warns)
        if not ok:
            m["net_error"] = err[-300:]
            return m
        m.update({k: v for k, v in coverage(res, netd, tmp).items() if k != "mv_missing"})
        m.update(simulate(tmp, netd))
        m["route_arrive_rate"] = m["route_arrived"] / m["route_pairs"] if m["route_pairs"] else float("nan")
    finally:
        if out_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    return m


if __name__ == "__main__":
    from . import pipeline
    lon, lat = float(sys.argv[1]), float(sys.argv[2])
    od = sys.argv[3] if len(sys.argv) > 3 else None
    res = pipeline.run(lon, lat, log=lambda *a: None)
    report = check(res, od)
    print(report)
    sys.exit(0 if report["status"] == "identified" and report["net_ok"] else 2)
