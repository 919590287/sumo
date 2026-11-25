"""SUMO 编译层的段间连接与渠化右转消重（v19 Phase 1）。"""
import unittest

from lanegen import sumo_check


class SectionLinks(unittest.TestCase):
    def _in_out(self, cons, n1, n2):
        indeg = {j: 0 for j in range(n2)}
        outdeg = {i: 0 for i in range(n1)}
        for _, i, _, j in cons:
            indeg[j] += 1
            outdeg[i] += 1
        return indeg, outdeg

    def test_added_lane_gets_upstream_feed(self):
        # 上游 2 车道 [10, 11]，下游左侧展宽出左转湾 [10, 11, 12]
        cons = sumo_check._section_links("a", [10, 11], "b", [10, 11, 12])
        indeg, outdeg = self._in_out(cons, 2, 3)
        self.assertTrue(all(v >= 1 for v in indeg.values()), cons)
        self.assertIn(("a", 1, "b", 2), cons)  # 新增车道从相邻的左侧车道分流

    def test_added_right_lane(self):
        cons = sumo_check._section_links("a", [10, 11], "b", [9, 10, 11])
        self.assertIn(("a", 0, "b", 0), cons)
        self.assertIn(("a", 0, "b", 1), cons)

    def test_dropped_lane_merges(self):
        cons = sumo_check._section_links("a", [9, 10, 11], "b", [10, 11])
        indeg, outdeg = self._in_out(cons, 3, 2)
        self.assertTrue(all(v >= 1 for v in outdeg.values()), cons)

    def test_disjoint_ids(self):
        cons = sumo_check._section_links("a", [1, 2], "b", [5, 6, 7])
        indeg, outdeg = self._in_out(cons, 2, 3)
        self.assertTrue(all(v >= 1 for v in indeg.values()))
        self.assertTrue(all(v >= 1 for v in outdeg.values()))


class SlipBypass(unittest.TestCase):
    def test_bypass_drops_right_from_options(self):
        jm = dict(arms=[dict(id=0, bearing=0.0, out_carriageways=[1]), dict(id=1, bearing=90.0, out_carriageways=[2]),
                        dict(id=2, bearing=180.0, out_carriageways=[3]), dict(id=3, bearing=270.0, out_carriageways=[4])],
                  carriageways=[dict(id=i, arm=i - 1, role="exit") for i in range(1, 5)],
                  slips=[dict(from_arm=2, to_arm=1, lanes=[99])])
        netd = dict(exits={i: f"e{i}" for i in range(1, 5)}, edges={f"e{i}": dict(lanes=[0, 1]) for i in range(1, 5)})
        # 从南臂(arm 2)北行（heading=0），右转目标是东臂 arm 1，已由渠化右转承担
        cw = dict(arm=2)
        mv_no_arrow = [(0, {"S"}, False), (1, {"L", "S"}, False)]
        opts = sumo_check.approach_options(jm, netd, cw, mv_no_arrow, 0.0)
        self.assertFalse(any("R" in o.movements for o in opts))
        mv_arrow = [(0, {"R"}, True), (1, {"S"}, True)]
        opts = sumo_check.approach_options(jm, netd, cw, mv_arrow, 0.0)
        self.assertTrue(any("R" in o.movements for o in opts))



class LaneAlignment(unittest.TestCase):
    def test_interior_split_keeps_neighbours(self):
        from lanegen.frenet import _align_lanes
        # 中间车道分叉成两条：外侧两条车道不应被嫁接到隔壁
        self.assertEqual(_align_lanes([0.0, 3.5, 7.0], [0.0, 3.0, 4.2, 7.0]), [(0, 0), (1, 1), (2, 3)])
        # 右侧新增
        self.assertEqual(_align_lanes([3.5, 7.0], [0.2, 3.5, 7.0]), [(0, 1), (1, 2)])
        # 左侧消失
        self.assertEqual(_align_lanes([0.0, 3.5, 7.0], [0.0, 3.5]), [(0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
