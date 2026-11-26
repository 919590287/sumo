"""纯坐标数学工具（AGENTS.md 第 3 节第 6 条允许的确定性基础设施）。

仅包含弧长重采样、法向平移、距离度量等与“裁决”无关的数学运算；
不包含任何阈值化的几何判别或场景特判。
"""
from __future__ import annotations

import numpy as np


def as_xy(points) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return p[..., :2]


def polyline_length(points) -> float:
    p = as_xy(points)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def cumulative_arclength(points) -> np.ndarray:
    p = as_xy(points)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(points, k: int) -> np.ndarray:
    """按等弧长把折线重采样为 k 个点（保留首尾点，保留输入的全部坐标维度）。"""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        raise ValueError("折线至少需要 2 个点")
    s = cumulative_arclength(p)
    if s[-1] <= 0:
        raise ValueError("折线长度为 0")
    keep = np.concatenate([[True], np.diff(s) > 0])
    p, s = p[keep], s[keep]
    t = np.linspace(0.0, s[-1], k)
    out = np.stack([np.interp(t, s, p[:, d]) for d in range(p.shape[1])], axis=1)
    out[0], out[-1] = p[0], p[-1]
    return out


def tangents(points) -> np.ndarray:
    """逐点单位切向（中心差分，端点单侧差分）。"""
    p = as_xy(points)
    g = np.gradient(p, axis=0)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-12)


def left_normals(points) -> np.ndarray:
    t = tangents(points)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def offset_polyline(points, d: float) -> np.ndarray:
    """沿左法向平移 d 米（d<0 为向右），保留 z 维。"""
    p = np.asarray(points, dtype=np.float64).copy()
    p[:, :2] = p[:, :2] + d * left_normals(p)
    return p


def densify(points, step: float = 0.05) -> np.ndarray:
    p = as_xy(points)
    L = polyline_length(p)
    k = max(2, int(np.ceil(L / step)) + 1)
    return resample_polyline(p, k)


def point_to_polyline_distance(q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """点集 q[M,2] 到折线 poly[K,2] 的最近距离。"""
    q = as_xy(q)
    a = as_xy(poly)[:-1]
    b = as_xy(poly)[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-18)
    aq = q[:, None, :] - a[None, :, :]
    t = np.clip((aq * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(q[:, None, :] - proj, axis=-1).min(axis=1)


def hausdorff(a, b, step: float = 0.05) -> float:
    """两条折线的对称 Hausdorff 距离（一侧加密采样 + 另一侧精确点到折线距离）。"""
    a = as_xy(a)
    b = as_xy(b)
    da = point_to_polyline_distance(densify(a, step), b).max()
    db = point_to_polyline_distance(densify(b, step), a).max()
    return float(max(da, db))


def endpoint_drift(a, b) -> float:
    a = as_xy(a)
    b = as_xy(b)
    return float(max(np.linalg.norm(a[0] - b[0]), np.linalg.norm(a[-1] - b[-1])))


def signed_turn_angle(t_in: np.ndarray, t_out: np.ndarray) -> float:
    """有符号偏转角 atan2(t_in × t_out, t_in · t_out)。"""
    cross = t_in[0] * t_out[1] - t_in[1] * t_out[0]
    dot = t_in[0] * t_out[0] + t_in[1] * t_out[1]
    return float(np.arctan2(cross, dot))


def hermite_curve(p0, t0, p1, t1, k: int, tension: float | None = None) -> np.ndarray:
    """三次 Hermite 曲线采样 k 点（端点位置与切向锁定），仅供手工标准图/合成真值构造使用。"""
    p0 = np.asarray(p0, float)[:2]
    p1 = np.asarray(p1, float)[:2]
    t0 = np.asarray(t0, float)[:2]
    t1 = np.asarray(t1, float)[:2]
    t0 = t0 / np.linalg.norm(t0)
    t1 = t1 / np.linalg.norm(t1)
    chord = np.linalg.norm(p1 - p0)
    m = chord if tension is None else tension
    s = np.linspace(0, 1, 4 * k + 1)[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    dense = h00 * p0 + h10 * m * t0 + h01 * p1 + h11 * m * t1
    return resample_polyline(dense, k)


def polygon_area(poly) -> float:
    p = as_xy(poly)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
