"""标线语义与进口箭头的通用规则（v18b，开创大道×荔红二路复盘时补的）。

运行（项目根目录）：
    ~/.microsim-platform/venv/bin/python -B -W ignore -m unittest tests.test_marking_semantics -v
不依赖数据。行驶方向 +x，y 为横向。
"""
import unittest

import numpy as np
from shapely.geometry import LineString

from lanegen import boundaries as bd
from lanegen import topology


def _h(y, x0=0.0, x1=60.0):
    return LineString([(x0, y), (x1, y)])


class BusYellowTest(unittest.TestCase):
    def test_bus_lane_pair_is_not_centerline(self):
        # 公交道夹在两条黄线之间（y=0 与 y=3.5），字样锚线在车道中线 y=1.75；外侧还有路面（不贴路缘也要认出）
        lo, hi = _h(0.0), _h(3.5)
        anchor = [_h(1.75, 10, 40)]
        ys = [lo, hi]
        self.assertTrue(bd._is_bus_edge_yellow(lo, ys, anchor))
        self.assertTrue(bd._is_bus_edge_yellow(hi, ys, anchor))

    def test_double_yellow_centerline_kept(self):
        # 双黄中心线旁边就是内侧公交道：仍是中心线
        a, b = _h(0.0), _h(0.3)
        anchor = [_h(2.0, 10, 40)]
        self.assertFalse(bd._is_bus_edge_yellow(a, [a, b], anchor))
        self.assertFalse(bd._is_bus_edge_yellow(b, [a, b], anchor))

    def test_plain_yellow_kept(self):
        y = _h(0.0)
        self.assertFalse(bd._is_bus_edge_yellow(y, [y], []))
        self.assertFalse(bd._is_bus_edge_yellow(y, [y], [_h(8.0, 10, 40)]))  # 公交标记在一个车道宽之外

    def test_duplicate_copy_is_not_double(self):
        # 同一条黄线重复绘制（重合）不算双黄
        a, b = _h(0.0), _h(0.0)
        self.assertTrue(bd._is_bus_edge_yellow(a, [a, b], [_h(1.75, 10, 40)]))

    def test_yellow_on_anchor_is_lettering(self):
        y = _h(1.75, 10, 30)
        self.assertEqual(bd._is_bus_edge_yellow(y, [y], [_h(1.75, 5, 40)]), "center")


class FreeRightArrowTest(unittest.TestCase):
    """停止线只画到右转车道左侧（渠化右转不停车）：右转箭头仍是本路口进口箭头；离得远的不是。"""

    def setUp(self):
        # 停止线在 x=50，横跨 y∈[0, 10.5]（3 条车道）；车辆 +x 驶向路口
        self.stops = [LineString([(50, 0), (50, 10.5)])]

    def _arrows(self, y):
        return [(np.array([20.0, y]), np.array([1.0, 0.0]), "arrow_right")]

    def test_just_outside_stop_end(self):
        out = topology.approach_arrows(self._arrows(-1.75), self.stops, [0])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], 0)

    def test_far_outside(self):
        self.assertEqual(topology.approach_arrows(self._arrows(-6.0), self.stops, [0]), [])

    def test_normal_hit_unchanged(self):
        out = topology.approach_arrows(self._arrows(5.0), self.stops, [0])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
