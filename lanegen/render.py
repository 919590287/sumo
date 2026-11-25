"""叠加图：深色底图（百度道路面、安全岛、标线、箭头、人行横道）+ 识别出的车道中心线。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import Point

BG = "#0d1117"
ROAD = "#3b4954"
ISLAND = "#5d7352"
CHANNEL = "#46535d"
EDGE = "#e8923a"
MARK = "#f2f2f2"
ZEBRA_BG = "#17b8a0"
LANE = "#4db8ff"
CORE = "#e58ad6"


def _polys(geoms):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon":
                continue
            out.append(MplPoly(np.asarray(p.exterior.coords)[:, :2], closed=True))
    return out


def _add_polys(ax, geoms, color, z, alpha=1.0, lw=0, ec="none"):
    ps = _polys(geoms)
    if ps:
        ax.add_collection(PatchCollection(ps, facecolor=color, edgecolor=ec, linewidths=lw, alpha=alpha, zorder=z))


def _lines(geoms):
    segs = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        for p in getattr(g, "geoms", [g]):
            if p.geom_type in ("LineString", "LinearRing"):
                segs.append(np.asarray(p.coords)[:, :2])
            elif p.geom_type == "Polygon":
                segs.append(np.asarray(p.exterior.coords)[:, :2])
                segs += [np.asarray(r.coords)[:, :2] for r in p.interiors]
    return segs


def draw(res, path, extent=None, title=None, show_events=True, dpi=110, extra_lines=()):
    """extra_lines: [(折线列表, 颜色, 线宽)]，叠在车道中心线之上（如 SUMO 编译后的车道 / 路口内部连接）。"""
    roi, B = res["roi"], res["B"]
    m = roi["main"]
    code = m["分类代码"].astype(str)
    fig, ax = plt.subplots(figsize=(13, 13))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    road = [g for g, c, cat in zip(m.geometry, code, m["原始类别"]) if c == "road_surface" and cat == 15]
    _add_polys(ax, road, ROAD, 1, ec=ROAD, lw=0.4)
    _add_polys(ax, [g for g, c in zip(m.geometry, code) if c == "channelization_area"], CHANNEL, 2)
    _add_polys(ax, [g for g, c in zip(m.geometry, code) if c == "traffic_island"], ISLAND, 3)
    # 路缘（可行驶包络边界）
    if B["drivable"] is not None:
        ax.add_collection(LineCollection(_lines([B["drivable"].boundary]), colors=EDGE, linewidths=1.0, zorder=4))
    # 人行横道底色 + 白条
    _add_polys(ax, [g for g, c in zip(m.geometry, code) if c == "crosswalk_area"], ZEBRA_BG, 5, alpha=0.55)
    z = roi["zebra"]
    if len(z):
        _add_polys(ax, list(z.geometry), MARK, 6, alpha=0.9)
    # 标线轮廓、箭头等符号
    _add_polys(ax, [g for g, t in zip(m.geometry, m["要素类型"]) if t == "标线轮廓"], MARK, 7, alpha=0.95)
    _add_polys(ax, [g for g, t, c in zip(m.geometry, m["要素类型"], code)
                    if t == "符号轮廓" and not c.startswith("parking")], MARK, 7, alpha=0.95)
    # 导流区斜纹边线
    ax.add_collection(LineCollection(_lines([g for g, c in zip(m.geometry, code) if c == "channelization_area"]),
                                     colors=MARK, linewidths=0.4, alpha=0.6, zorder=7))
    # 停止线：本路口的红色粗线，其他的暗红细线
    cs_ids = {id(s) for s in res.get("core_stops", [])}
    st = B.get("stops", [])
    ax.add_collection(LineCollection(_lines([s for s in st if id(s) not in cs_ids]), colors="#a04040",
                                     linewidths=1.0, zorder=8))
    ax.add_collection(LineCollection(_lines([s for s in st if id(s) in cs_ids]), colors="#ff4d4d",
                                     linewidths=2.2, zorder=8))
    # 路口核心（虚线）
    ax.add_collection(LineCollection(_lines([res["core"].exterior]), colors=CORE, linewidths=1.0,
                                     linestyles=(0, (4, 3)), zorder=8))
    # 车道中心线
    segs = [np.asarray(l["geom"].coords) for l in res["lanes"] if not l.get("nonmotor_suspect")]
    ax.add_collection(LineCollection(segs, colors=LANE, linewidths=1.6, zorder=9, capstyle="round"))
    nm = [np.asarray(l["geom"].coords) for l in res["lanes"] if l.get("nonmotor_suspect")]
    if nm:  # 疑似非机动车道：浅色虚线
        ax.add_collection(LineCollection(nm, colors="#9fd8ff", linewidths=1.0, linestyles=(0, (3, 3)), zorder=9))
    segs = segs + nm
    for k, (ls, col, lw) in enumerate(extra_lines):
        if ls:
            ax.add_collection(LineCollection(ls, colors=col, linewidths=lw, zorder=12 + k, capstyle="round"))
    if show_events:
        for l in res["lanes"]:
            g = l["geom"]
            c = np.asarray(g.coords)
            # 方向：中点小箭头
            if l["direction"] == "known" and len(c) >= 3:
                k = len(c) // 2
                ax.annotate("", xy=c[min(k + 1, len(c) - 1)], xytext=c[k - 1],
                            arrowprops=dict(arrowstyle="-|>", color=LANE, lw=0, mutation_scale=9), zorder=10)
            else:
                ax.plot(*c[len(c) // 2], marker="o", ms=4, mfc="none", mec="#ff5555", zorder=10)
            if l["start"] == "lane_add":
                ax.plot(*c[0], marker="o", ms=5, color="#ffd23f", zorder=11)
            if l["end"] == "lane_drop":
                ax.plot(*c[-1], marker="X", ms=6, color="#ffd23f", zorder=11)
            if l["slip"]:
                ax.plot(*c[len(c) // 2], marker="s", ms=5, mfc="none", mec="#7CFC00", zorder=11)
    if extent is None:
        xs = np.concatenate([s[:, 0] for s in segs]) if segs else np.array([-80, 80])
        ys = np.concatenate([s[:, 1] for s in segs]) if segs else np.array([-80, 80])
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        h = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 10
        extent = (cx - h, cx + h, cy - h, cy + h)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.text(0.01, 0.99, title, transform=ax.transAxes, color="#cfd8dc", fontsize=11, va="top",
                family=["PingFang SC", "Arial Unicode MS"])
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_model(res, path, half=80, dpi=100):
    """Step 2 结构叠图：核心面（粉）、障碍（亮绿=转角渠化岛，暗绿=其他岛，暗灰红=外围街区）、路臂口门（黄）、
    物理单向车行道（蓝=驶入，绿=驶出；虚边浅色=无标线物理车行道）、平滑 Frenet 轴线、绕岛渠化右转走廊（橙）。"""
    jm = res["model"]
    B = res["B"]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    dv = B["drivable"]
    _add_polys(ax, [dv] if dv is not None else [], ROAD, 1)
    for o in jm["obstacles"]:
        if o["kind"] == "corner_island":
            _add_polys(ax, [o["geom"]], "#6f9654", 2, alpha=0.95, ec="#b7e394", lw=0.8)
        elif o["kind"] == "island":
            _add_polys(ax, [o["geom"]], ISLAND, 2, alpha=0.85)
        else:
            _add_polys(ax, [o["geom"]], "#2e2226", 2, alpha=0.8)
    ax.add_collection(LineCollection(_lines([l for l, t in B["bounds"] if t in ("white", "dash", "yellow", "bus")]),
                                     colors="#9aa5ad", linewidths=0.4, zorder=3))
    ax.add_collection(LineCollection(_lines([l for l, t in B["bounds"] if t == "yellow"]),
                                     colors="#e5c000", linewidths=0.8, zorder=3))
    _add_polys(ax, [jm["core_poly"]], CORE, 4, alpha=0.35, ec="#f2b8e8", lw=0.8)
    ax.add_collection(LineCollection(_lines(res.get("core_stops", [])), colors="#ff4d4d", linewidths=2.0, zorder=6))
    pal_in = ["#2f7fd8", "#56a6f5", "#1d5aa6", "#7cc0ff"]
    pal_out = ["#2fae5a", "#6bd68a", "#1e7d3f", "#a3e8b6"]
    for c in jm["carriageways"]:
        col = (pal_in if c["role"] == "approach" else pal_out)[c["id"] % 4]
        if c.get("unmarked"):
            _add_polys(ax, [c["polygon"]], col, 5, alpha=0.22, ec=col, lw=1.2)
            ax.plot(*np.asarray(c["axis"].coords).T, color=col, lw=1.4, ls="--", zorder=7)
        else:
            _add_polys(ax, [c["polygon"]], col, 5, alpha=0.42)
            ax.plot(*np.asarray(c["axis"].coords).T, color=col, lw=1.5, zorder=7)
        if c["mouth"] is not None:
            ax.plot(*np.asarray(c["mouth"].coords).T, color="w", lw=1.0, zorder=8)
    for a in jm["arms"]:
        for m in a["mouths"]:
            ax.plot(*np.asarray(m.coords).T, color="#ffd23f", lw=3.0, zorder=8, solid_capstyle="butt")
        p = a["mid"] + a["n"] * 8
        ax.text(p[0], p[1], f"A{a['id']}\n{a['bearing']:.0f}°\n入{len(a['in_carriageways'])} 出{len(a['out_carriageways'])}",
                color="#ffd23f", fontsize=9, ha="center", va="center", zorder=10, family=["PingFang SC", "Arial Unicode MS"])
    for s in jm["slips"]:
        if s.get("corridor_poly") is not None and not s["corridor_poly"].is_empty:
            _add_polys(ax, [s["corridor_poly"]], "#ff9f1c", 6, alpha=0.38, ec="#ffbe55", lw=1.0)
        g = s.get("axis") if s.get("axis") is not None else res["lanes"][s["lane"]]["geom"]
        ax.plot(*np.asarray(g.coords).T, color="#ff9f1c", lw=2.4, zorder=9)
        ax.plot(*s["diverge_pt"], "o", color="#ff9f1c", ms=6, zorder=10)
        ax.plot(*s["merge_pt"], "X", color="#ff9f1c", ms=7, zorder=10)
    for l in res["lanes"]:
        if l.get("role") != "slip":
            ax.plot(*np.asarray(l["geom"].coords).T, color="w", lw=0.6, alpha=0.6, zorder=7)
    cx, cy = jm["core_poly"].centroid.x, jm["core_poly"].centroid.y
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")
    n_unm = sum(1 for c in jm["carriageways"] if c.get("unmarked"))
    ax.text(0.01, 0.99,
            f"路臂 {len(jm['arms'])}  车行道 {len(jm['carriageways'])}(无标线 {n_unm})  渠化右转 {len(jm['slips'])}",
            transform=ax.transAxes, color="#cfd8dc", fontsize=11, va="top", family=["PingFang SC", "Arial Unicode MS"])
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
