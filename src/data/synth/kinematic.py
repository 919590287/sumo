"""独立运动学内部轨迹生成器（docs/DESIGN.md §5 Layer 1-1）。

合成真值 K_conn* 不用 netconvert 默认内部曲线，而由 C² 连续的七次 Bézier 样条直接生成：
- 端点位置与切向严格锁定进出口断面；P0-P1-P2 与 P5-P6-P7 分别共线 ⇒ 两端曲率为 0，
  与恒宽（端部近直线）车道段 C² 衔接；
- 显式约束：最小转弯半径（普通 R_min = 6.0 m、掉头 4.5 m）、对安全岛 / 导流区的车体包络避障
  间隙、同 (进口 edge, 出口 edge) 并列转向的保序间隔；
- 罚函数形式的 L-BFGS-B 优化（8 个自由度：两端切向控制点距离 + 两个内部控制点），
  不满足约束即抛出 ``KinematicFail``，由场景采样器整体拒绝重采（合成数据侧的拒绝采样）。

本模块只服务于训练数据生成，不参与任何推理期裁决。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from src.common.geom import resample_polyline


class KinematicFail(RuntimeError):
    pass


def _bernstein(n: int, t: np.ndarray, d: int = 0) -> np.ndarray:
    """n 次 Bernstein 基（或其 d 阶导数）在 t 处的取值矩阵 [len(t), n+1]。"""
    from math import comb
    if d == 0:
        return np.stack([comb(n, k) * t ** k * (1 - t) ** (n - k) for k in range(n + 1)], 1)
    lower = _bernstein(n - 1, t, d - 1)
    out = np.zeros((len(t), n + 1))
    for k in range(n + 1):
        a = lower[:, k - 1] if k - 1 >= 0 else 0.0
        b = lower[:, k] if k <= n - 1 else 0.0
        out[:, k] = n * (a - b)
    return out


@dataclass
class Obstacles:
    """禁入多边形集合（安全岛 / 导流区）：KD 树加速的边界距离 + 点内判定。"""
    polys: list[np.ndarray]

    def __post_init__(self):
        pts = []
        for p in self.polys:
            p = np.asarray(p, float)
            ring = np.vstack([p, p[:1]])
            seg = np.diff(ring, axis=0)
            L = np.linalg.norm(seg, axis=1)
            for a, s, l in zip(ring[:-1], seg, L):
                k = max(2, int(math.ceil(l / 0.2)) + 1)
                pts.append(a + np.linspace(0, 1, k)[:, None] * s)
        self.tree = cKDTree(np.vstack(pts)) if pts else None
        self.paths = [MplPath(np.asarray(p, float)) for p in self.polys]

    def signed_distance(self, q: np.ndarray) -> np.ndarray:
        if self.tree is None:
            return np.full(len(q), np.inf)
        d, _ = self.tree.query(q)
        inside = np.zeros(len(q), bool)
        for p in self.paths:
            inside |= p.contains_points(q)
        return np.where(inside, -d, d)


class TrajectoryGenerator:
    DEG = 7

    def __init__(self, cfg: dict):
        self.cfg = cfg
        n = cfg["samples"]
        self.t = np.linspace(0.0, 1.0, n)
        self.B = _bernstein(self.DEG, self.t)
        self.dB = _bernstein(self.DEG, self.t, 1)
        self.ddB = _bernstein(self.DEG, self.t, 2)
        mid = (self.t > 0.15) & (self.t < 0.85)
        self.mid = mid

    # ------------------------------------------------------------------ 参数化
    @staticmethod
    def _ctrl(x, p0, t0, p1, t1):
        a0, b0, a1, b1 = np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), np.exp(x[3])
        P = np.empty((8, 2))
        P[0] = p0
        P[1] = p0 + a0 * t0
        P[2] = p0 + (a0 + b0) * t0
        P[3] = x[4:6]
        P[4] = x[6:8]
        P[5] = p1 - (a1 + b1) * t1
        P[6] = p1 - a1 * t1
        P[7] = p1
        return P

    def _eval(self, P):
        xy = self.B @ P
        v = self.dB @ P
        a = self.ddB @ P
        sp = np.linalg.norm(v, axis=1)
        kappa = (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.maximum(sp, 1e-9) ** 3
        ds = sp / (len(self.t) - 1)
        return xy, kappa, ds

    def _cost(self, x, p0, t0, p1, t1, r_min, obs, sib, need):
        c = self.cfg
        P = self._ctrl(x, p0, t0, p1, t1)
        xy, kappa, ds = self._eval(P)
        bend = float(np.sum(kappa ** 2 * ds))
        length = float(ds.sum())
        pen = float(np.sum(np.maximum(np.abs(kappa) - 1.0 / r_min, 0.0) ** 2 * ds)) * 100.0
        if obs is not None:
            sd = obs.signed_distance(xy)
            pen += float(np.sum(np.maximum(need - sd, 0.0) ** 2))
        if sib is not None:
            d, _ = sib.query(xy[self.mid])
            pen += float(np.sum(np.maximum(self.gap - d, 0.0) ** 2))
        return c["w_bend"] * bend + c["w_len"] * length + c["w_penalty"] * pen

    def _check(self, P, r_min, obs, sib, need) -> list[str]:
        xy, kappa, ds = self._eval(P)
        bad = []
        # 拒绝绕圈解：总转角不得显著超过端点切向净偏转，弧长不得远超弦长
        turn_total = float(np.sum(np.abs(kappa) * ds))
        if turn_total > abs(self._psi) + math.pi / 2 or float(ds.sum()) > max(3.0 * self._chord, self._chord + 45.0):
            bad.append("绕圈 / 过长轨迹")
        if np.abs(kappa).max() > 1.0 / r_min * 1.02:
            bad.append(f"最小转弯半径 {1.0 / max(np.abs(kappa).max(), 1e-9):.2f} m < {r_min} m")
        if obs is not None and obs.signed_distance(xy).min() < need - 0.05:
            bad.append("侵入安全岛 / 导流区")
        if sib is not None and sib.query(xy[self.mid])[0].min() < self.gap - 0.1:
            bad.append("并列转向间隔不足")
        return bad

    @staticmethod
    def _inits(p0, t0, p1, t1, L, uturn, r_min):
        """多起点初值：Hermite 型 + （掉头）前探外摆的灯泡形。"""
        outs = []
        n0 = np.array([-t0[1], t0[0]])
        side = 1.0 if float((p1 - p0) @ n0) >= 0 else -1.0
        scales = [1.0, 0.7, 1.4] if not uturn else []
        for k in scales:
            m = max(L, 1.0) * k
            s = np.array([3 / 7, 4 / 7])[:, None]
            h = ((2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * m * t0
                 + (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * m * t1)
            outs.append(np.r_[[math.log(m / 7)] * 4, h[0], h[1]])
        if uturn:
            for F in (2.2 * r_min, 3.0 * r_min, 1.6 * r_min):
                for sw in (0.6 * r_min, 1.2 * r_min, 0.0):
                    P3 = p0 + t0 * F - n0 * side * sw
                    P4 = p1 - t1 * F + n0 * side * sw
                    outs.append(np.r_[[math.log(F / 3)] * 4, P3, P4])
        return outs

    # ------------------------------------------------------------------ 主入口
    def generate(self, p0, t0, p1, t1, k: int, uturn: bool = False, obstacles: Obstacles | None = None,
                 siblings: list[np.ndarray] | None = None, lane_width: float = 3.2) -> tuple[np.ndarray, np.ndarray]:
        """返回 (k 点等弧长轨迹, 稠密轨迹)。"""
        self.gap = self.cfg["parallel_gap_frac"] * lane_width
        p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
        t0 = np.asarray(t0, float)[:2]
        t1 = np.asarray(t1, float)[:2]
        t0, t1 = t0 / np.linalg.norm(t0), t1 / np.linalg.norm(t1)
        chord = p1 - p0
        L = float(np.linalg.norm(chord))
        if L < 1e-6:
            raise KinematicFail("连接首尾重合")
        # 直线续接（属性切换 / 掉头开口外的同车道续行）
        lat = abs(t0[0] * chord[1] - t0[1] * chord[0])
        if float(t0 @ t1) > 0.99999 and lat < 1e-6 and float(t0 @ chord) > 0:
            dense = p0 + np.linspace(0, 1, 32)[:, None] * chord
            return resample_polyline(dense, k), dense
        c = self.cfg
        r_min = c["r_min_uturn"] if uturn else c["r_min_turn"]
        need = c["vehicle_half_width_m"] + c["obstacle_clearance_m"]
        sib = None
        if siblings:
            sib = cKDTree(np.vstack(siblings))
        obs = obstacles if (obstacles is not None and obstacles.polys) else None
        self._chord = L
        self._psi = math.atan2(t0[0] * t1[1] - t0[1] * t1[0], float(t0 @ t1))
        if uturn:
            self._psi = math.pi
        best, best_cost = None, np.inf
        for x0 in self._inits(p0, t0, p1, t1, L, uturn, r_min):
            P = self._ctrl(x0, p0, t0, p1, t1)
            if not uturn and not self._check(P, r_min, obs, sib, need):
                best = P
                break
            lo = np.r_[[math.log(0.2)] * 4, np.minimum(p0, p1) - 3 * L - 30, np.minimum(p0, p1) - 3 * L - 30]
            hi = np.r_[[math.log(4 * L + 40)] * 4, np.maximum(p0, p1) + 3 * L + 30, np.maximum(p0, p1) + 3 * L + 30]
            x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
            with np.errstate(all="ignore"):
                res = minimize(self._cost, x0, args=(p0, t0, p1, t1, r_min, obs, sib, need), method="L-BFGS-B",
                               bounds=list(zip(lo, hi)), options={"maxiter": c["max_iter"]})
            if not np.all(np.isfinite(res.x)):
                continue
            P = self._ctrl(res.x, p0, t0, p1, t1)
            if not self._check(P, r_min, obs, sib, need):
                if not uturn:
                    best = P
                    break
                best = P                          # 掉头：首个可行的优化解（绕圈 / 过长已由校验排除）
                break
        if best is None:
            raise KinematicFail("; ".join(self._check(P, r_min, obs, sib, need)) or "优化失败")
        P = best
        tt = np.linspace(0, 1, 200)
        dense = _bernstein(self.DEG, tt) @ P
        out = resample_polyline(dense, k)
        out[0], out[-1] = p0, p1
        return out, dense


def min_radius(points: np.ndarray) -> float:
    """折线离散最小曲率半径（三点外接圆），供单测与质检。"""
    p = np.asarray(points, float)[:, :2]
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / np.maximum(2 * cross, 1e-12)
    return float(np.min(r)) if len(r) else float("inf")
