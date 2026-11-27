"""底图叠加可视化（docs/DESIGN.md §7.1-2 强制视觉终审）：

底图 = 输入要素栅格（道路面 / 白黄标线 / 箭头 / 斑马线 / 岛 / 导流 / 停车 / 非机动），叠加：
预测节点与路口轮廓、车道中心线与标量宽度边缘、内部连接轨迹、人行横道，可选真值（虚线）与
SUMO 仿真车辆位置快照（fcd）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.config import load_config
from src.common.geom import offset_polyline
from src.graph.lane_graph import LaneGraph

NODE_COLORS = {"junction_core": "#e41a1c", "split_merge": "#984ea3", "midblock_node": "#00bcd4",
               "road_continuation": "#ffb300", "roi_entry_port": "#222222", "roi_exit_port": "#666666"}


def base_image(bev: np.ndarray) -> np.ndarray:
    names = [c["name"] for c in load_config("data")["channels"]]
    ch = {n: k for k, n in enumerate(names)}
    b = bev.astype(np.float32)
    img = np.full(b.shape[1:] + (3,), 0.10, np.float32)
    img += 0.22 * b[ch["road_surface"]][..., None]
    white = np.max(b[[ch[n] for n in ("white_dashed", "white_solid", "white_double_solid", "white_ldash_rsolid",
                                      "white_lsolid_rdash", "crosswalk_stripe", "road_boundary")]], 0)
    img += 0.65 * white[..., None]
    yel = np.max(b[[ch[n] for n in ("yellow_solid", "yellow_lsolid_rdash", "yellow_double_dashed")]], 0)
    img += yel[..., None] * np.array([0.8, 0.65, 0.0])
    img += b[ch["arrow_mask"]][..., None] * np.array([0.55, 0.35, 0.1])
    img += b[ch["traffic_island"]][..., None] * np.array([0.0, 0.25, 0.05])
    img += b[ch["channelization_area"]][..., None] * np.array([0.35, 0.0, 0.35])
    img += (b[ch["parking"]] + b[ch["nonmotor"]])[..., None] * np.array([0.0, 0.1, 0.35])
    img += b[ch["separation_path"]][..., None] * np.array([0.5, 0.0, 0.0])
    img *= (0.55 + 0.45 * b[ch["roi_mask"]])[..., None]
    return np.clip(img, 0, 1)


def draw_graph(ax, g: LaneGraph, H: int, res: float, style: str = "pred") -> None:
    px = lambda p: (p[..., 0] / res, H - p[..., 1] / res)  # noqa: E731
    ls = "-" if style == "pred" else "--"
    alpha = 0.95 if style == "pred" else 0.6
    lc = "#29b6f6" if style == "pred" else "#ffffff"
    for l in g.lanes:
        x, y = px(l.points[:, :2])
        ax.plot(x, y, ls, color=lc, lw=0.9, alpha=alpha)
        if style == "pred":
            for s in (-1, 1):
                e = offset_polyline(l.points, s * l.width / 2)
                x2, y2 = px(e[:, :2])
                ax.plot(x2, y2, "-", color="#29b6f6", lw=0.3, alpha=0.6)
            ax.annotate("", px(l.points[-1, :2]), px(l.points[-3, :2]),
                        arrowprops=dict(arrowstyle="-|>", color=lc, lw=0.6, mutation_scale=6))
    for c in g.connections:
        x, y = px(c.points[:, :2])
        ax.plot(x, y, ls, color="#66bb6a" if style == "pred" else "#c8e6c9", lw=0.7, alpha=alpha)
    for n in g.nodes:
        x, y = px(n.pos[:2])
        ax.plot(x, y, "o", color=NODE_COLORS[n.node_type], ms=4 if style == "pred" else 3, alpha=alpha)
        if n.polygon is not None and style == "pred":
            p = np.vstack([n.polygon, n.polygon[:1]])
            x, y = px(p)
            ax.plot(x, y, "--", color="#e41a1c", lw=0.7)
    for cw in g.crosswalks:
        x, y = px(cw.endpoints[:, :2])
        ax.plot(x, y, ls, color="#ff4081" if style == "pred" else "#f8bbd0", lw=2.0, alpha=alpha)


def render_overlay(path: Path, bev: np.ndarray, hw, pred: LaneGraph | None, gt: LaneGraph | None = None,
                   title: str = "", vehicles: np.ndarray | None = None, res: float = 0.2) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = int(hw[0]), int(hw[1])
    img = base_image(bev[:, :H, :W])
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.imshow(img, interpolation="nearest")
    if gt is not None:
        draw_graph(ax, gt, H, res, "gt")
    if pred is not None:
        draw_graph(ax, pred, H, res, "pred")
    if vehicles is not None and len(vehicles):
        ax.plot(vehicles[:, 0] / res, H - vehicles[:, 1] / res, "s", color="#ffeb3b", ms=2.5)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=load_config("eval")["overlay_dpi"], bbox_inches="tight")
    plt.close(fig)
