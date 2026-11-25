"""叠图：原始证据 + 拟合车道（左），SUMO 成网（右）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

COL = dict(white="#222", dash="#2a6fdb", yellow="#e8a400", bus="#b04fd0", barrier="#2e8b57", sep="#2e8b57")


def _poly(ax, g, fc, ec=None, z=0, alpha=1.0):
    if g is None or g.is_empty:
        return
    for p in getattr(g, "geoms", [g]):
        if p.geom_type != "Polygon":
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, color=fc, zorder=z, lw=0, alpha=alpha)
        if ec:
            ax.plot(x, y, color=ec, lw=0.6, zorder=z + 0.1)
        for h in p.interiors:
            x, y = h.xy
            ax.fill(x, y, color="white", zorder=z + 0.05, lw=0)


def base(ax, ev):
    _poly(ax, ev.D, "#e4e4e4", z=0)
    _poly(ax, ev.nondrive, "#c9e3c0", z=0.2, alpha=0.9)
    for l, t in ev.bounds:
        x, y = l.xy
        ax.plot(x, y, color=COL.get(t, "#666"), lw=0.7, ls="--" if t == "dash" else "-", zorder=1, alpha=0.55)
    for s in ev.stops:
        x, y = s.xy
        ax.plot(x, y, color="#d62728", lw=2.2, zorder=1.5, alpha=0.7)
    for p, v, code in ev.arrows:
        ax.arrow(p[0], p[1], v[0] * 1.8, v[1] * 1.8, width=0.25, color="#7b3fa0", zorder=1.6, alpha=0.6)


def lane_lines(f):
    """拟合车道的边界线（世界坐标）。返回 [(xy数组, 类别列表)]：med / div_ok / div_inf / outer。"""
    out = []
    secs = f["secs"]
    for key in ("in", "out"):
        side = f[key]
        n, w, a, sigma, i0 = side["n"], side["w"], side["a"], side["sigma"], side["i0"]
        if n.max() == 0:
            continue
        for k in range(0, int(n.max()) + 1):
            run, kind_run = [], []
            for i in range(i0, len(secs)):
                sc = secs[i]
                if k > n[i]:
                    if len(run) >= 2:
                        out.append((np.array(run), kind_run))
                    run, kind_run = [], []
                    continue
                u = a[i] + sigma * k * w[i]
                run.append(sc["p"] + sc["n"] * u)
                if k == 0:
                    kd = "med"
                elif k == n[i]:
                    kd = "outer"
                else:
                    rel = sigma * (sc["u"] - a[i])
                    du = rel[np.isin(sc["typ"], ("dash", "white", "bus"))]
                    kd = "div_ok" if len(du) and np.abs(du - k * w[i]).min() <= 0.5 else "div_inf"
                kind_run.append(kd)
            if len(run) >= 2:
                out.append((np.array(run), kind_run))
    return out


def lane_centers(f):
    out = []
    secs = f["secs"]
    for key in ("in", "out"):
        side = f[key]
        n, w, a, sigma, i0 = side["n"], side["w"], side["a"], side["sigma"], side["i0"]
        for k in range(int(n.max()) if n.max() else 0):
            run = []
            for i in range(i0, len(secs)):
                sc = secs[i]
                if k >= n[i]:
                    if len(run) >= 2:
                        out.append((np.array(run), sigma))
                    run = []
                    continue
                run.append(sc["p"] + sc["n"] * (a[i] + sigma * (k + 0.5) * w[i]))
            if len(run) >= 2:
                out.append((np.array(run), sigma))
    return out


def draw(ev, fits, center, path, title="", sumo_lines=None, bad=None, half=95):
    ncol = 2 if sumo_lines is not None else 1
    fig, axs = plt.subplots(1, ncol, figsize=(11 * ncol, 11))
    axs = np.atleast_1d(axs)
    ax = axs[0]
    base(ax, ev)
    for f in fits:
        x, y = f["axis"].xy
        ax.plot(x, y, color="#888", lw=0.6, ls=":", zorder=2)
        for pts, kinds in lane_lines(f):
            for j in range(len(pts) - 1):
                kd = kinds[j]
                c, ls, lw = {"med": ("#ff7f0e", "-", 2.0), "outer": ("#111", "-", 1.6),
                             "div_ok": ("#18a558", "-", 1.6), "div_inf": ("#e0245e", (0, (3, 2)), 1.8)}[kd]
                ax.plot(pts[j:j + 2, 0], pts[j:j + 2, 1], color=c, ls=ls, lw=lw, zorder=3)
        for pts, sg in lane_centers(f):
            ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4" if sg > 0 else "#9467bd", lw=0.8, zorder=2.5, alpha=0.8)
        # 每条进口车道转向标注
        mvs = f.get("lane_mv_final") or f.get("lane_mv")
        if mvs:
            i = min(len(f["secs"]) - 1, f["in"]["i0"] + 3)
            sc = f["secs"][i]
            for k, mv in enumerate(mvs):
                q = sc["p"] + sc["n"] * (f["in"]["a"][i] + (k + 0.5) * f["in"]["w"][i])
                ax.text(q[0], q[1], mv, fontsize=8, color="#7b3fa0", ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        p = np.asarray(f["axis"].coords[-1])
        lab = f"{f['id']}: 进{int(f['in']['n'][f['in']['i0']])} 出{int(f['out']['n'][f['out']['i0']])}"
        ax.text(*(np.asarray(f["axis"].interpolate(min(f["axis"].length, f["s0_in"] + 45)).coords[0])), lab,
                fontsize=10, color="k", zorder=6, bbox=dict(fc="#fffbe6", ec="#999", alpha=0.9))
    ax.set_title(title + "\n绿=有标线支撑的分界  红虚=模型推断（缺线）  橙=中线  黑=外缘  紫字=进口车道转向", fontsize=10)
    if sumo_lines is not None:
        ax2 = axs[1]
        base(ax2, ev)
        normal, internal = sumo_lines
        for s in normal:
            ax2.plot(s[:, 0], s[:, 1], color="#1f77b4", lw=1.3, zorder=3)
        for s in internal:
            ax2.plot(s[:, 0], s[:, 1], color="#f0a500", lw=0.9, zorder=3)
        for s in (bad or []):
            ax2.plot(s[:, 0], s[:, 1], color="#e0245e", lw=2.5, zorder=4)
        ax2.set_title("SUMO 成网：蓝=路段车道  黄=路口内部连接  红=穿越安全岛/导流区的部分", fontsize=10)
    for a in axs:
        a.set_xlim(center[0] - half, center[0] + half)
        a.set_ylim(center[1] - half, center[1] + half)
        a.set_aspect("equal")
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)
