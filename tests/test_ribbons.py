"""Phase 3 切片：标线导向参考轴与标线链车道带（合成几何，不依赖数据）。"""
import math
import unittest

import numpy as np
from shapely.geometry import LineString, box

from lanegen import ribbons


def _F(M=40, w=3.5, marks=((-1.75, "dash"), (1.75, "dash")), extra=None):
    ss = np.arange(M) * 2.0
    obs = []
    for i in range(M):
        o = [(d, t) for d, t in marks]
        if extra:
            o += [(d, t) for d, t in extra(i)]
        obs.append(sorted(o))
    return dict(M=M, ds=2.0, ss=ss, d_R=np.full(M, -5.25), d_L=np.full(M, 5.25), w_cw=w,
                obs_all=obs, arrows=[])


class MarkAxis(unittest.TestCase):
    def test_wiggly_axis_replaced_by_straight_marking(self):
        poly = box(0, -10, 120, 10)
        xs = np.linspace(1, 119, 60)
        wiggly = LineString(np.c_[xs, 3.0 * np.sin(xs / 8.0)])
        B = dict(bounds=[(LineString([(0, 1.75), (120, 1.75)]), "dash"), (LineString([(0, -1.75), (120, -1.75)]), "dash")])
        ax = ribbons.mark_axis(wiggly, poly, B)
        self.assertIsNotNone(ax)
        c = np.asarray(ax.coords)
        hd = np.degrees(np.arctan2(np.diff(c[:, 1]), np.diff(c[:, 0])))
        self.assertLess(float(np.max(np.abs(hd))), 1.0)
        self.assertGreater(c[-1, 0] - c[0, 0], 110)  # 两端外延到原轴范围

    def test_no_marking_keeps_axis(self):
        poly = box(0, -10, 120, 10)
        self.assertIsNone(ribbons.mark_axis(LineString([(1, 0), (119, 0)]), poly, dict(bounds=[])))


class RibbonPartition(unittest.TestCase):
    def test_marked_three_lanes(self):
        K, B, SN, info = ribbons.ribbon_partition(_F(), "exit")
        self.assertTrue(np.all(K == 3))
        np.testing.assert_allclose(B[10][3], [-5.25, -1.75, 1.75, 5.25], atol=1e-6)

    def test_gore_band_gets_no_lane(self):
        # 第 10–30 断面右侧分出一条白实线（离虚线 1 m）：楔形区不放车道，车道中心不压这条白线
        F = _F(extra=lambda i: [(-2.75, "white")] if 10 <= i <= 30 else [])
        K, B, SN, info = ribbons.ribbon_partition(F, "exit")
        for i in range(12, 29):
            b = B[i][K[i]]
            centres = 0.5 * (b[:-1] + b[1:])
            self.assertTrue(np.all(np.abs(centres + 2.75) > 0.6), (i, b))
            self.assertTrue(np.all(np.abs(centres + 1.75) > 0.6), (i, b))

    def test_wide_unmarked_band_subdivided(self):
        F = _F(marks=((1.75, "dash"),))  # 右侧 7 m 宽无标线带
        K, B, SN, info = ribbons.ribbon_partition(F, "exit")
        self.assertTrue(np.all(K == 3))

    def test_unmarked_carriageway_returns_none(self):
        self.assertIsNone(ribbons.ribbon_partition(_F(marks=()), "exit"))


if __name__ == "__main__":
    unittest.main()
