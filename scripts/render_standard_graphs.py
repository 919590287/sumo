"""标准图编译前后叠加渲染（G 车道中心线 / K_conn 黑色虚实线 vs net.xml 车道/内部车道/斑马线彩色）。

用法：/opt/anaconda3/envs/baidu2sumo/bin/python -B -m scripts.render_standard_graphs [name ...]
输出：runs/m0/figures/<name>.png
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common.config import REPO_ROOT, cache_dir  # noqa: E402
from src.data.standard_graphs import STANDARD_GRAPHS  # noqa: E402
from src.sumo_io.pipeline import compile_graph  # noqa: E402
from src.sumo_io.roundtrip import parse_net  # noqa: E402


def render(name: str) -> None:
    g = STANDARD_GRAPHS[name]()
    r = compile_graph(g, cache_dir("m0_render", name), with_eval_sim=False)
    net = parse_net(r.site.net_path)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
    for lid, shp in net.lane_shapes.items():
        if lid.startswith(":") and lid.split("_")[-2].startswith("c"):
            ax.plot(shp[:, 0], shp[:, 1], color="#1f77b4", lw=6, alpha=0.35, solid_capstyle="butt")
        elif lid.startswith(":") and lid.split("_")[-2].startswith("w"):
            continue
        elif lid.startswith(":"):
            ax.plot(shp[:, 0], shp[:, 1], color="#ff7f0e", lw=1.6, alpha=0.9)
        elif net.lane_allow.get(lid) == "pedestrian":
            ax.plot(shp[:, 0], shp[:, 1], color="#bbbbbb", lw=1.0, ls=":")
        else:
            ax.plot(shp[:, 0], shp[:, 1], color="#2ca02c", lw=2.6, alpha=0.8)
    for l in g.lanes:
        p = l.points
        ax.plot(p[:, 0], p[:, 1], color="k", lw=0.8)
        nrm = np.c_[-np.gradient(p[:, 1]), np.gradient(p[:, 0])]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        for s in (1, -1):
            e = p[:, :2] + s * nrm * l.width / 2
            ax.plot(e[:, 0], e[:, 1], color="#888888", lw=0.4)
        ax.annotate("", xy=p[-1, :2], xytext=p[-4, :2], arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    for c in g.connections:
        ax.plot(c.points[:, 0], c.points[:, 1], color="k", lw=0.6, ls="--")
    for n in g.nodes:
        if n.polygon is not None:
            q = np.vstack([n.polygon, n.polygon[:1]])
            ax.plot(q[:, 0], q[:, 1], color="#d62728", lw=0.8)
    rt = r.roundtrip
    ax.set_title(f"{name}: conn {rt.conn_compiled}/{rt.conn_expected}, "
                 f"Hausdorff lane {rt.lane_hausdorff_max:.3f} m / conn {rt.conn_hausdorff_max:.3f} m, "
                 f"crossings {rt.crossings_compiled}/{rt.crossings_expected}", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    out = REPO_ROOT / "runs/m0/figures" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    names = sys.argv[1:] or ["slip_right_turn", "two_stage_crossing", "flare_widening", "six_arm_median",
                             "midblock_uturn", "corridor_two_junctions"]
    for n in names:
        render(n)
