"""按本路口 ROI 的数据现算参数，不用全局固定值。

每个参数都写明来源：
  data      由本 ROI 的标线统计得到
  derived   由 data 参数按几何关系推出
  physical  物理或规范常数（与具体路口无关）
  dataset   数据集自身的约定
  scope     输出范围（用户可调，不影响正确性）
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

# 无量纲系数（敏感性测试时用环境变量 LANEGEN_F='{"w_min_k":0.9}' 覆盖）
F = dict(
    w_min_k=0.75,      # 车道下限 / 众数
    w_max_k=1.35,      # 单车道上限 / 众数
    link_lat_k=0.15,   # 样本串联横向容差 / 众数
    stitch_k=0.25,     # 接回横向、宽度容差 / 众数
    group_k=0.35,      # 箭头成组横向容差 / 众数
    dash_k=1.3,        # 虚线补段上限 / 间隙 P75
    par_deg=25.0,      # 射线命中边界的平行容差（度）
    smooth_m=15.0,     # 中心线平滑窗口（米）
    clear_out_k=2.0,   # 空地判据：连线离开路面的最长连续一段 ≤ clear_out_k × 本地车道宽
    front_tol=3.0,     # 相向判据容差（米）
)
F.update(json.loads(os.environ.get("LANEGEN_F", "{}")))


def lane_width_mode(d, lo=2.0, hi=5.0, bw=0.05):
    """射线间距的众数（车道宽度主峰）。用平滑直方图取峰，比中位数抗非机动车道、双车道走廊的干扰。"""
    d = d[(d >= lo) & (d <= hi)]
    if len(d) < 50:
        return 3.3, "default"
    bins = np.arange(lo, hi + bw, bw)
    h, _ = np.histogram(d, bins)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 3.0) ** 2)
    hs = np.convolve(h, k / k.sum(), mode="same")
    # 只在物理合理区间 [2.5, 4.2] 找峰
    c = (bins[:-1] + bins[1:]) / 2
    ok = (c >= 2.5) & (c <= 4.2)
    return float(c[ok][np.argmax(hs[ok])]), "data"


def dash_gap(dashes, max_search=30.0):
    """同一条虚线上相邻两段之间的间隙：每个端点找前方共线的最近端点，取距离分布。"""
    ends, dirs = [], []
    for l in dashes:
        c = np.asarray(l.coords)
        if len(c) < 2 or l.length < 0.5:
            continue
        t = c[-1] - c[0]
        t = t / np.linalg.norm(t)
        ends += [c[0], c[-1]]
        dirs += [-t, t]
    if len(ends) < 10:
        return None
    P, T = np.array(ends), np.array(dirs)
    tree = cKDTree(P)
    gaps = []
    for i in range(len(P)):
        best = None
        for j in tree.query_ball_point(P[i], max_search):
            if j // 2 == i // 2:
                continue
            v = P[j] - P[i]
            g = np.linalg.norm(v)
            if g < 0.3:
                continue
            u = v / g
            if np.dot(T[i], u) > 0.985 and np.dot(T[j], -u) > 0.985 and abs(T[i][0] * v[1] - T[i][1] * v[0]) < 0.4:
                best = g if best is None else min(best, g)
        if best is not None:
            gaps.append(best)
    return np.array(gaps) if gaps else None


def derive(ray_d, dashes, dash_info=None):
    """dash_info=(dash_max_gap, 来源[, dash_gap_p75])：boundaries.build 已经按同一规则算过时直接复用，不重复统计。"""
    P = {}
    src = {}
    w, s = lane_width_mode(ray_d)
    P["w_mode"], src["w_mode"] = w, s
    # 车道宽度上下限：众数 ±25% 覆盖国标 2.75–3.75 m 的相对跨度；上限必须 < 2 倍下限，
    # 否则一条走廊既可能是 1 条宽车道也可能是 2 条窄车道，判定会自相矛盾。
    P["w_min"] = max(2.2, F["w_min_k"] * w)
    # 单车道上限：国标最宽车道（含路缘偏移）约为典型车道的 1.35 倍；同时必须 < 2 倍下限。
    # 介于单车道上限和 2 倍下限之间的走廊按“宽走廊”处理（贴边候选），不取中点。
    P["w_max"] = min(F["w_max_k"] * w, 2 * P["w_min"] - 0.2)
    P["w_wide"] = 2 * P["w_max"]
    src.update(w_min="derived", w_max="derived", w_wide="derived")
    # 样本串联：横向容差远小于半个车道（相邻车道中心至少相距 w_min）
    P["link_lat"] = F["link_lat_k"] * w
    P["link_r"] = 2.2  # 采样步长 1 m 的 2.2 倍，保证相邻样本可连
    src.update(link_lat="derived", link_r="derived")
    # 虚线补段（上限 = dash_k × 间隙 P75）
    if dash_info is not None:
        P["dash_max_gap"], src["dash_max_gap"] = dash_info[0], dash_info[1]
        P["dash_gap"] = dash_info[2] if len(dash_info) > 2 else None
    else:
        g = dash_gap(dashes)
        if g is not None and len(g) >= 8:
            P["dash_gap"] = float(np.percentile(g, 75))
            P["dash_max_gap"] = float(np.clip(F["dash_k"] * P["dash_gap"], 3.0, 20.0))
            src["dash_max_gap"] = "data"
        else:
            P["dash_gap"] = None
            P["dash_max_gap"] = 16.0
            src["dash_max_gap"] = "default"
    # 接回、成组的横向容差
    P["stitch_lat"] = F["stitch_k"] * w
    P["stitch_dw"] = F["stitch_k"] * w
    P["arrow_group_lat"] = F["group_k"] * w
    P["neighbor_lo"], P["neighbor_hi"] = 0.45 * w, 1.5 * w
    src.update(stitch_lat="derived", stitch_dw="derived", arrow_group_lat="derived")
    # 物理 / 数据集常数
    P["z_tol"], src["z_tol"] = 1.5, "dataset(立交分层阈值)"
    P["close_r"], src["close_r"] = 0.3, "physical(<0.6 m 缝隙不构成通行分隔)"
    P["keep_r"], src["keep_r"] = 160.0, "scope"
    P["_source"] = src
    return P
