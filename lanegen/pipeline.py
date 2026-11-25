"""单路口车道中心线识别主流程。"""
import json
import time

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from . import roi as roi_mod
from . import boundaries, lanes as lanes_mod, junction, model as model_mod, frenet as frenet_mod
from .lanes import TYPES

_PHYS_IDX = [TYPES.index(t) for t in ("edge", "barrier", "sep")]


def _dbg(stage, lanes):
    """调试：环境变量 LANEGEN_DEBUG_BOX="x0,y0,x1,y1" 时打印各阶段与该框相交的车道。"""
    import os
    b = os.environ.get("LANEGEN_DEBUG_BOX")
    if not b:
        return
    from shapely.geometry import box
    bx = box(*[float(v) for v in b.split(",")])
    for l in lanes:
        g = l["geom"]
        if g.intersects(bx):
            print(f"  [{stage}] len={g.length:.0f} {tuple(np.round(g.coords[0]))}->{tuple(np.round(g.coords[-1]))} "
                  f"role={l.get('role')} score={l.get('score')} ev={l.get('evidence')}")


def run(lon, lat, radius_m=280, log=print, canonical=True):
    """canonical=True：种子规范化——先只跑到核心面，把种子挪到核心质心再完整跑一遍。
    路口“是谁、核心多大”不再取决于种子落在路口哪个角上（留出集最不稳的 9 个路口实测平移稳定 0.645→0.786）。"""
    if not canonical:
        return _run(lon, lat, radius_m, log)
    from pyproj import Transformer
    lon2, lat2 = lon, lat
    for _ in range(_CANON_ITERS):  # 不动点迭代：种子 → 核心质心，直到挪动 < 2 m
        core_poly, crs = _run(lon2, lat2, radius_m, lambda *a: None, core_only=True)
        c = core_poly.centroid
        if core_poly.is_empty or np.hypot(c.x, c.y) < 2.0:
            break
        lon2, lat2 = Transformer.from_crs(crs, 4326, always_xy=True).transform(c.x, c.y)
    res = _run(lon2, lat2, radius_m, log)
    res["seed_input"] = (lon, lat)
    return res


import os as _os
_CANON_ITERS = int(_os.environ.get("LANEGEN_CANON_ITERS", "3"))  # 迭代 1 次时白云 11515 平移稳定 0.63，3 次 0.96


def _recognition(model, core_stops, has_data):
    """区分几何归属契约与路口识别，只计确有车道的进出口。"""
    if not has_data:
        return "no_data", "empty_roi"
    if model.get("contract_errors"):
        return "partial", "invalid_lane_contract"
    roles = {(c["arm"], c["role"]) for c in model["carriageways"] if c.get("lanes") and c["arm"] is not None}
    if not roles:
        return "unresolved", "no_lane_bearing_arms"
    if not core_stops and len({arm for arm, role in roles}) < 3:
        return "unresolved", "junction_identity_unconfirmed"
    if not any((a != b) for a, role in roles if role == "approach" for b, other in roles if other == "exit"):
        return "partial", "missing_interarm_path"
    return "identified", None


def _run(lon, lat, radius_m=280, log=print, core_only=False):
    from . import params as params_mod, topology
    t0 = time.time()
    raw, crs = roi_mod.read_roi(lon, lat, radius_m)
    seed = (0.0, 0.0)  # 局部投影原点就是种子
    z0 = roi_mod.estimate_z0(raw["lines"], seed)
    roi = roi_mod.zfilter(raw, z0)
    log(f"  读取+Z过滤 {time.time()-t0:.1f}s  z0={z0:.2f}")

    B = boundaries.build(roi)
    log(f"  边界 {len(B['bounds'])} 条  停止线 {len(B['stops'])}")

    # 射线 → 由本 ROI 数据推导参数
    c = lanes_mod.cast(B["bounds"])
    dv = B["drivable"].buffer(0.3) if B["drivable"] is not None and not B["drivable"].is_empty else None
    on = lanes_mod.on_road(c, dv)
    # 车道宽度只用至少一侧是涂装标线的射线：两侧都是路缘/护栏/隔离带的走廊（非机动车道、人行道缝、
    # 绿化带夹缝）在 v18c 的 66 口里有 30 口把众数拉到 2.775 m，连带把 w_max 锁死在 3.75 m。
    phys = np.isin(c["src"], _PHYS_IDX) & np.isin(c["hit"], _PHYS_IDX)
    mk = on & ~phys
    P = params_mod.derive(c["d"][mk] if mk.sum() >= 300 else c["d"][on], B["dashes_raw"],
                          dash_info=(B["dash_max_gap"][0], B["dash_max_gap"][1], B.get("dash_gap_p75")))
    keep_r = P["keep_r"]
    S = lanes_mod.emit(c, P, drivable=B["drivable"])
    lab, ncomp, G = lanes_mod.link(S, P)
    log(f"  射线样本 {len(S['X'])}  分量 {ncomp}  W={P['w_mode']:.2f} [{P['w_min']:.2f},{P['w_max']:.2f}]")

    main = roi["main"]
    bikes = [g.centroid for g, cd in zip(main.geometry, main["分类代码"]) if cd == "bicycle_lane_symbol"]
    parking = unary_union([g for g, cd in zip(main.geometry, main["分类代码"]) if cd in ("onstreet_parking_space",)])
    arrows = junction.arrows_of(main)
    yellow_i = TYPES.index("yellow")

    # 路口：停止线方向 + 相向且空地连通
    stops = B["stops"]
    dirs = topology.stop_directions(stops, arrows, B["bounds"], crosswalks=B["crosswalks"])
    space = topology.Space(B["drivable"], B["bounds"], stops, B["nondrive"], w=P["w_mode"])
    core, cluster = topology.junction_cluster(seed, stops, dirs, space, B["crosswalks"])
    core_stops = [stops[k] for k in cluster]
    app_arrows = topology.approach_arrows(arrows, stops, cluster, crosswalks=B["crosswalks"])
    stop_u = unary_union(stops) if stops else None
    w = P["w_mode"]

    lanes = []
    dropped = []
    slip_pool = []
    nd_near =B["nondrive"].buffer(0.6) if B["nondrive"] is not None else None
    counts = np.bincount(lab)
    pieces = []
    for k in np.nonzero(counts >= 6)[0]:
        idx0 = np.nonzero(lab == k)[0]
        if not np.any(S["kind"][idx0] != 2):
            dropped.append(("anchored_only", None)); continue
        pieces += lanes_mod.chain_all(idx0, S, G)
    for xy, cw, idx in pieces:
        kinds = S["kind"][idx]
        n_norm = int((kinds == 0).sum())
        n_edge = int((kinds == 1).sum())
        n_anch = int((kinds == 2).sum())
        geom = LineString(xy)
        if geom.length < 2 * w:
            continue
        width = float(np.nanmedian(cw)) if np.any(np.isfinite(cw)) else w
        yl = np.vstack([S["sp"][idx][S["src"][idx] == yellow_i], S["hp"][idx][S["hit"][idx] == yellow_i]])
        nk = kinds != 2
        sr, ht = S["src"][idx][nk], S["hit"][idx][nk]
        dash_i, curb = TYPES.index("dash"), [TYPES.index("edge"), TYPES.index("barrier")]
        frac_dash = float(np.mean((sr == dash_i) | (ht == dash_i))) if nk.any() else 0.0
        frac_curb = float(np.mean(np.isin(sr, curb) | np.isin(ht, curb))) if nk.any() else 0.0
        # 贴安全岛/导流区一侧的样本比例（渠化右转道：一侧是岛，另一侧是外路缘）
        frac_island = 0.0
        if nd_near is not None and nk.any():
            spn, hpn = S["sp"][idx][nk], S["hp"][idx][nk]
            frac_island = float(np.mean(shapely.contains_xy(nd_near, spn[:, 0], spn[:, 1])
                                        | shapely.contains_xy(nd_near, hpn[:, 0], hpn[:, 1])))
        ln = dict(geom=geom, width=width, n_norm=n_norm, n_edge=n_edge, n_anch=n_anch,
                  yellow_pts=yl, has_yellow=bool(len(yl)), frac_dash=frac_dash, frac_curb=frac_curb,
                  frac_island=frac_island)
        buf = geom.buffer(0.5 * w + 0.3)
        nb = sum(1 for b in bikes if buf.contains(b))
        na = sum(1 for p, v, cd in arrows if buf.contains(Point(p)))
        if nb and not na:
            dropped.append(("bike", geom)); continue
        if not parking.is_empty and geom.intersection(parking.buffer(0.3)).length > 0.4 * geom.length:
            dropped.append(("parking", geom)); continue
        if n_norm < 0.3 * (n_norm + n_edge):
            touch_stop = stop_u is not None and geom.distance(stop_u) < 0.5 * w + 1.0
            # 渠化右转道两侧都是路缘（岛 + 外路缘），没有标线；靠“一侧贴岛、在路口旁、转过弯”认定
            slip_like = (frac_island >= 0.3 and geom.distance(core) < 25.0 and geom.length < 200.0
                         and _heading_change(geom) >= 25.0)
            if not (na or touch_stop or slip_like):
                slip_pool.append(ln)
                continue
            ln["edgepair"] = True
        lanes.append(ln)
    # 渠化右转候选：两侧都是路缘、没有标线的弯道（右转岛常是路面上的空洞，不是 traffic_island 面）。
    # 两端都接着有标线的车道（连接两条路臂）、转过 25° 以上、在路口旁、长 15–150 m；
    # 只有一端接着的是支路、小区出入口，仍然删掉
    marked = [l["geom"] for l in lanes if not l.get("edgepair")]
    mtree = shapely.STRtree(marked) if marked else None
    for ln in slip_pool:
        g = ln["geom"]
        ok = (mtree is not None and 15.0 <= g.length <= 150.0 and g.distance(core) < 30.0
              and _heading_change(g) >= 25.0)
        if ok:
            for e in (Point(g.coords[0]), Point(g.coords[-1])):
                if not any(marked[j].distance(e) < 12.0 for j in mtree.query(e.buffer(12.0))):
                    ok = False
                    break
        if ok:
            ln["edgepair"] = True
            ln["slip_candidate"] = True
            lanes.append(ln)
        else:
            dropped.append(("edgepair_unsupported", g))
    _dbg("cand", lanes)
    n_cand = len(lanes)
    blockers = unary_union(stops + [c_.boundary for c_ in B["crosswalks"]] + [core])
    marks = unary_union([l for l, t in B["bounds"] if t in ("white", "dash", "yellow", "bus")])
    lanes = junction.stitch_lanes(lanes, blockers, P=P, space=space, marks=marks)
    _dbg("stitch1", lanes)
    log(f"  候选车道 {n_cand}  接回后 {len(lanes)}")
    # 核心沿各进口行驶方向延伸到对面第一条车道（T 形路口无停止线的主路一侧）；
    # 穿过核心的车道在核心处断开，分成进口段和出口段
    core0 = core
    core, quads = topology.expand_core(core, cluster, stops, dirs, [l["geom"] for l in lanes], space,
                                       crosswalks=B["crosswalks"])
    P["core_grown_m2"] = round(core.area - core0.area, 1)
    # Step 2：紧致核心面（凸包 ∩ 可行驶面，去掉安全岛、路面空洞、绕岛走廊），所有车道在其边界处断开
    core_hull = core
    core_poly = model_mod.build_core_poly(core0, quads, B["drivable"], core_stops, w,
                                          stop_dirs=[dirs[k] for k in cluster],
                                          crosswalks=B["crosswalks"],
                                          nondrive=B["nondrive"])
    # 试过把路段标线 1.8 m 内从核心扣掉（seg_marks=space.seg）：增城 19800、番禺 5934 变好，荔湾 7913 等变差，
    # 留出集出口臂覆盖 0.895→0.886，净效果不明显，默认不开
    P["core_poly_m2"] = round(core_poly.area, 1)
    if core_only:
        return core_poly, crs

    # 先在停止线、人行横道处截断（T 形路口主路车道在路口前方四边形处断开，只切横穿的），再定向，再按结构判据归属本路口
    for ln in lanes:
        ln["geom"] = topology.cut_transverse(ln["geom"], quads)
    lanes = junction.select_and_trim(lanes, core_hull, keep_r=keep_r, touch=1e9, stops=stops,
                                     crosswalks=list(B["crosswalks"]), core_cut=core_poly.buffer(-0.5))
    core = core_poly
    _dbg("trim", lanes)
    refs = boundaries.explode_lines(roi["ref"].geometry)
    opp = [l for l, t in B["bounds"] if t in ("yellow", "sep", "barrier")]
    opposing = unary_union(opp) if opp else None
    lanes = junction.orient(lanes, arrows, stops, core, P, refs, space=space, opposing=opposing,
                            cluster_stops=core_stops, stop_dirs=dirs)
    _dbg("orient", lanes)
    lanes = junction.extend_to_junction(lanes, core_stops, core, space=space, w=w, crosswalks=B["crosswalks"],
                                        nondrive=B["nondrive"])
    _dbg("extend", lanes)
    ocw = [c_.boundary for c_ in B["crosswalks"] if not c_.intersects(core.buffer(1.0))]
    other_cw = unary_union(ocw) if ocw else None
    kept = []
    for ln in lanes:
        if ln.get("slip_candidate") and ln["geom"].distance(core) < 30.0:
            role = "slip"  # 渠化右转绕开路口核心：先于进口/出口判定（其端点回溯也可能碰到核心）
        else:
            role = topology.belongs(ln, core_stops, core, space, cluster, tol_end=0.5 * w + 1.0, block=other_cw)
        if role:
            ln["role"] = role
            kept.append(ln)
        else:
            dropped.append(("not_this_junction", ln["geom"]))
    lanes = kept
    _dbg("belongs", lanes)
    n0 = len(lanes)
    lanes = junction.arrow_lanes(lanes, app_arrows, stops, B["drivable"], core, P, bounds=B["bounds"],
                                 crosswalks=B["crosswalks"])
    for ln in lanes[n0:]:
        ln["role"] = "approach"
    n_arrow = len(lanes) - n0
    lanes = junction.stitch_lanes(lanes, blockers, P=P, space=space, marks=marks, oriented=True)
    for ln in lanes:
        ln.setdefault("score", 0.0)
        ln.setdefault("evidence", [])
        ln["direction"] = "known" if ln["score"] != 0 else "unknown"
    _dbg("stitch2", lanes)
    log(f"  箭头补出车道 {n_arrow}")
    lanes = junction.propagate_direction(lanes, w)
    lanes = junction.direction_from_stops(lanes, core_stops, [dirs[k] for k in cluster], w)
    lanes = junction.propagate_direction(lanes, w)
    _dbg("prop", lanes)
    # 路口内部的无方向碎片（导向线、待转区之间扫出来的短段）：方向未定且八成以上在核心内 → 删掉；
    # 路口内部的行驶轨迹由转向连接生成，不属于车道
    kept = []
    for ln in lanes:
        g = ln["geom"]
        if ln["direction"] == "unknown" and g.intersection(core0).length >= 0.8 * g.length:
            dropped.append(("interior_fragment", g))
        else:
            kept.append(ln)
    lanes = kept
    n_before = len(lanes)
    lanes = junction.dedupe(lanes, w)
    P["deduped"] = n_before - len(lanes)
    _dbg("dedupe", lanes)
    # 结构约束：车道不进核心面内部（延伸、接回、去重吸收都可能把端点带进核心），最后统一再裁一次，
    # 保留含车道远端（离核心远的一端）的那一段
    inner = core_poly.buffer(-0.5)
    kept = []
    for ln in lanes:
        g = ln["geom"]
        if ln.get("role") == "slip" or inner.is_empty or not g.intersects(inner):
            kept.append(ln)
            continue
        rest = g.difference(inner)
        parts = [q for q in getattr(rest, "geoms", [rest]) if q.geom_type == "LineString" and q.length >= 2.0]
        if not parts:
            dropped.append(("inside_core", g))
            continue
        e0, e1 = Point(g.coords[0]), Point(g.coords[-1])
        far = e0 if e0.distance(core_poly) > e1.distance(core_poly) else e1
        q = min(parts, key=lambda q: q.distance(far))
        if q.length < 2 * w:
            dropped.append(("inside_core", g))
            continue
        ln["geom"] = q
        kept.append(ln)
    lanes = kept
    # 停止线 ≤ 2 条的路口把与核心相连、无车道经过的空路面并入核心。必须在端点标注之前做，
    # 之后的 label_events / 模型 / 剖分 / 渲染全部用同一个核心面
    if len(core_stops) <= 2:
        core_poly = model_mod.grow_free(core_poly, lanes, B["drivable"], w)
        core = core_poly
    lanes = junction.label_events(lanes, core, stops, cluster, arrows, P, space, crosswalks=B["crosswalks"],
                                  bounds=B["bounds"])
    # 邻口：不在本路口簇里、离核心 ≥ 8 m、行驶方向背离本路口（= 下一个路口的进口停止线）的停止线
    cset = set(cluster)
    cc = np.asarray(core_poly.centroid.coords[0])
    foreign = []
    for k, s_ in enumerate(stops):
        if k in cset or s_.length < 3.0 or dirs[k] is None or s_.distance(core_poly) < 8.0:
            continue
        m_ = np.asarray(s_.interpolate(0.5, normalized=True).coords[0])
        v_ = m_ - cc
        if float(np.dot(dirs[k], v_)) > 0.5 * float(np.linalg.norm(v_)):
            foreign.append(s_)
    jm = model_mod.build(lanes, core_poly, B, arrows, w, opposing=opposing, n_stops=len(core_stops), refs=refs,
                         core_stops=core_stops, stop_dirs=[dirs[k] for k in cluster], grow=False, foreign_stops=foreign)
    core = core_poly = jm["core_poly"]
    jm["guides"] = [g for g in B.get("guides", []) if g.distance(core_poly) < 15.0]
    import os
    if not os.environ.get("LANEGEN_NO_FRENET"):
        lanes_dp = frenet_mod.partition_junction(jm, lanes, B, core_stops, arrows, bikes, parking, P)
        if lanes_dp:
            lanes = junction.label_events(lanes_dp, core, stops, cluster, arrows, P, space, crosswalks=B["crosswalks"],
                                          bounds=B["bounds"])
        else:
            # 没有任何车行道 / 渠化右转（无停止线、无路臂的小路口）：Step 1 车道全部作为附加支路登记，契约照样闭合
            for c_ in jm["carriageways"]:
                c_["lanes"], c_["sections"], c_["n_lanes"] = [], [], 0
            for s_ in jm["slips"]:
                s_["lanes"], s_["lane"] = [], None
            for ln in lanes:
                ln["carriageway"] = None
                ln.setdefault("source", "ray")
                ln.setdefault("allow", "all")
                ln.setdefault("lane_index", None)
            jm["extra_lanes"] = list(range(len(lanes)))
            jm["contract_errors"] = frenet_mod.validate_model_contract(jm, lanes)
        if jm.get("contract_errors"):
            log(f"  [契约] {len(jm['contract_errors'])} 处：{jm['contract_errors'][:3]}")
    # 潮汐/可变车道：车道一侧贴着可变车道线（≥ 一半长度）→ 标 reversible，供 Step 4 设可变导向
    rev = B.get("rev") or []
    if rev:
        ru = unary_union(rev)
        for ln in lanes:
            g = ln["geom"]
            ss = np.arange(1.0, g.length, 3.0)
            if len(ss) == 0:
                continue
            pts = shapely.line_interpolate_point(g, ss)
            near = shapely.distance(ru, pts) <= 0.5 * ln["width"] + 0.4
            ln["reversible"] = bool(np.mean(near) >= 0.5)
    bad = arrow_dir_conflicts(lanes, arrows, w)
    for i, ln in enumerate(lanes):
        ln["dir_conflict"] = i in bad
    jm["dir_conflicts"] = len(bad)
    status, reason = _recognition(jm, core_stops, any(not df.empty for df in raw.values()))
    jm.update(status=status, status_reason=reason)
    jm["cw_arrow_dir_conflict"] = sum(int(c.get("arrow_dir_conflicts", 0)) for c in jm["carriageways"])
    log(f"  路口相关车道 {len(lanes)}  路臂 {len(jm['arms'])}  车行道 {len(jm['carriageways'])}  "
        f"渠化右转 {len(jm['slips'])}  总耗时 {time.time()-t0:.1f}s")
    params_out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items()}
    return dict(lanes=lanes, dropped=dropped, core=core, core0=core0, core_hull=core_hull, model=jm,
                core_stops=core_stops, cluster=cluster, stop_dirs=dirs,
                roi=roi, B=B, crs=crs, z0=z0, w_ref=w, params=params_out, seed=(lon, lat),
                status=status, status_reason=reason)


def arrow_dir_conflicts(lanes, arrows, w):
    """实测方向冲突：每枚导向箭头找最近的车道（0.5 W + 0.3 m 内），车道走向与箭头相反（cos < -0.7）则该车道冲突。
    返回冲突车道下标集合。"""
    bad = set()
    if not lanes:
        return bad
    tree = shapely.STRtree([ln["geom"] for ln in lanes])
    for p, v, _ in arrows:
        pt = Point(p)
        cand = tree.query(pt.buffer(0.5 * w + 0.3))
        if len(cand) == 0:
            continue
        i = min(cand, key=lambda k: lanes[k]["geom"].distance(pt))
        g = lanes[i]["geom"]
        if g.distance(pt) > 0.5 * w + 0.3:
            continue
        s_ = g.project(pt)
        a = np.asarray(g.interpolate(max(0.0, s_ - 1.0)).coords[0])
        b = np.asarray(g.interpolate(min(g.length, s_ + 1.0)).coords[0])
        t = b - a
        if np.linalg.norm(t) > 1e-9 and float(np.dot(t / np.linalg.norm(t), v)) < -0.7:
            bad.add(int(i))
    return bad


def _heading_change(g):
    """折线首尾 5 m 切向的夹角（度，不分左右）。"""
    if g.length < 8:
        return 0.0
    c0 = np.asarray(g.interpolate(0).coords[0]); c1 = np.asarray(g.interpolate(5).coords[0])
    d0 = np.asarray(g.interpolate(g.length - 5).coords[0]); d1 = np.asarray(g.interpolate(g.length).coords[0])
    a, b = c1 - c0, d1 - d0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def export(res, out_dir):
    import geopandas as gpd
    rows = []
    for i, ln in enumerate(res["lanes"]):
        rows.append(dict(
            lane_id=i, width_m=round(ln["width"], 2), direction=ln["direction"],
            dir_score=round(float(ln["score"]), 1), evidence=",".join(ln["evidence"]),
            start=ln["start"], end=ln["end"], slip=ln["slip"], turn_deg=round(ln["turn_deg"], 1),
            arrows=json.dumps(ln["arrows"], ensure_ascii=False), edgepair=bool(ln.get("edgepair", False)),
            role=ln.get("role"), nonmotor_suspect=bool(ln.get("nonmotor_suspect", False)),
            arm=ln.get("arm"), carriageway=ln.get("carriageway"), dir_conflict=bool(ln.get("dir_conflict", False)),
            lane_index=ln.get("lane_index"), sec_idx=json.dumps({str(k): v for k, v in (ln.get("sec_idx") or {}).items()}),
            source=ln.get("source"), allow=ln.get("allow", "all"), reversible=bool(ln.get("reversible", False)),
            slip_from_arm=ln.get("slip_from_arm"), slip_to_arm=ln.get("slip_to_arm"),
            length_m=round(ln["geom"].length, 1), geometry=ln["geom"]))
    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=res["crs"])
        gdf.to_crs(4326).to_file(f"{out_dir}/lanes.geojson", driver="GeoJSON")
    else:
        with open(f"{out_dir}/lanes.geojson", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
    rep = dict(
        status=res["status"], status_reason=res["status_reason"],
        seed=res["seed"], z0=res["z0"], w_ref=res["w_ref"], n_lanes=len(rows),
        unknown_direction=int(sum(r["direction"] == "unknown" for r in rows)),
        lane_add=int(sum(r["start"] == "lane_add" for r in rows)),
        lane_drop=int(sum(r["end"] == "lane_drop" for r in rows)),
        slip=int(sum(r["slip"] for r in rows)),
        start_types={k: int(sum(r["start"] == k for r in rows)) for k in set(r["start"] for r in rows)},
        end_types={k: int(sum(r["end"] == k for r in rows)) for k in set(r["end"] for r in rows)},
        proj=res["crs"].to_proj4(),
    )
    jm = res.get("model")
    if jm is not None:
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform
        tf = Transformer.from_crs(res["crs"], 4326, always_xy=True).transform
        with open(f"{out_dir}/junction_model.json", "w", encoding="utf-8") as f:
            json.dump(model_mod.to_json(jm, lambda g: shp_transform(tf, g)), f, ensure_ascii=False)
        rep.update(contract_errors=len(jm.get("contract_errors", [])), dir_conflicts=jm.get("dir_conflicts", 0),
                   n_extra_lanes=len(jm.get("extra_lanes", [])),
                   n_arms=len(jm["arms"]), n_carriageways=len(jm["carriageways"]), n_slips=len(jm["slips"]),
                   arms=[dict(id=a["id"], bearing=round(a["bearing"], 1), n_in=len(a["in_carriageways"]),
                              n_out=len(a["out_carriageways"])) for a in jm["arms"]])
    with open(f"{out_dir}/report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep
