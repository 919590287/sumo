"""Step 3 → Step 4 数据契约测试。

运行（项目根目录）：
    ~/.microsim-platform/venv/bin/python -B -W ignore -m unittest tests.test_contract -v
单元测试不依赖数据；集成测试需要 GPKG（环境变量 LANEGEN_GPKG 或 roi.GPKG 默认路径），找不到就跳过。
"""
import json
import os
import tempfile
import unittest

import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import transform as shp_transform

from lanegen import frenet, model


def _lane(coords, cw=None, role="approach", source="frenet_dp"):
    return dict(geom=LineString(coords), width=3.5, carriageway=cw, role=role, source=source)


class BuildSectionsTest(unittest.TestCase):
    """build_sections：右侧加一条车道时，各段序号严格右→左 0..K-1，links 记录延续关系（旧 v15 会出现两条 lane_index=0）。"""

    def setUp(self):
        # 行驶方向 +x，左法向 +y：y 越小越靠右
        self.cw = dict(id=0, role="approach", axis=LineString([(0, 0), (100, 0)]))
        self.lanes = [
            _lane([(0, 1.75), (100, 1.75)], 0),     # 左车道（全程）
            _lane([(0, -1.75), (100, -1.75)], 0),   # 中车道（全程，加道后从序号 0 变 1）
            _lane([(50, -5.25), (100, -5.25)], 0),  # 右侧新增车道（s=50 起）
        ]

    def test_right_add(self):
        secs = frenet.build_sections(self.cw, self.lanes, [0, 1, 2], cuts=[50.0])
        self.assertEqual([s["n_lanes"] for s in secs], [2, 3])
        self.assertEqual(secs[0]["lane_ids"], [1, 0])          # 右→左
        self.assertEqual(secs[1]["lane_ids"], [2, 1, 0])
        self.assertEqual(secs[1]["transition"], "lane_add_right")
        self.assertEqual(sorted(secs[1]["links"]), [[0, 1], [1, 2]])
        # 进口道 lane_index 取靠路口一端（最后一段）的序号
        self.assertEqual([self.lanes[i]["lane_index"] for i in range(3)], [2, 1, 0])
        self.assertEqual(self.lanes[1]["sec_idx"], {0: 0, 1: 1})
        self.assertEqual(self.cw["lanes"], [2, 1, 0])
        for s in secs:
            self.assertEqual(sorted(self.lanes[i]["sec_idx"][s["section_id"]] for i in s["lane_ids"]),
                             list(range(s["n_lanes"])))

    def test_merge_identical_sections(self):
        # 多余切点（被删车道留下的）两侧车道集合相同 → 合并成一段
        secs = frenet.build_sections(self.cw, self.lanes, [0, 1], cuts=[30.0, 60.0])
        self.assertEqual(len(secs), 1)
        self.assertEqual(secs[0]["n_lanes"], 2)


class ValidateContractTest(unittest.TestCase):
    def _jm(self):
        cw = dict(id=0, role="approach", axis=LineString([(0, 0), (100, 0)]))
        lanes = [_lane([(0, 1.75), (100, 1.75)], 0), _lane([(0, -1.75), (100, -1.75)], 0)]
        frenet.build_sections(cw, lanes, [0, 1])
        slip_lane = _lane([(0, -20), (20, -30)], None, role="slip", source="bypass_slip")
        lanes.append(slip_lane)
        jm = dict(carriageways=[cw], slips=[dict(id=0, lanes=[2])], extra_lanes=[])
        return jm, lanes

    def test_clean(self):
        jm, lanes = self._jm()
        self.assertEqual(frenet.validate_model_contract(jm, lanes), [])

    def test_orphan_and_dangling(self):
        jm, lanes = self._jm()
        lanes.append(_lane([(0, 10), (50, 10)], cw=7, role="exit"))  # 孤儿 + 悬挂 carriageway=7
        errs = frenet.validate_model_contract(jm, lanes)
        self.assertTrue(any("孤儿" in e for e in errs))
        self.assertTrue(any("悬挂" in e for e in errs))

    def test_section_mismatch(self):
        jm, lanes = self._jm()
        jm["carriageways"][0]["sections"][0]["n_lanes"] = 3
        self.assertTrue(any("n_lanes" in e for e in frenet.validate_model_contract(jm, lanes)))

    def test_out_of_range_and_double_owner(self):
        jm, lanes = self._jm()
        jm["slips"][0]["lanes"] = [2, 9]
        jm["extra_lanes"] = [0]
        errs = frenet.validate_model_contract(jm, lanes)
        self.assertTrue(any("越界" in e for e in errs))
        self.assertTrue(any("已归属" in e for e in errs))


class SlipPairingTest(unittest.TestCase):
    """_slip_lanes 与 slip 严格成对：中间一条生成失败时，后面的车道不会错配给前面的 slip（v15 的 zip 错位）。"""

    def test_pairs(self):
        drivable = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
        core = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        B = dict(nondrive=None, drivable=drivable)
        slips = [
            dict(id=0, axis=LineString([(20, 20), (40, 50)]), width_m=3.5, from_arm=0, to_arm=1),
            dict(id=1, axis=LineString([(20, -20), (22, -21)]), width_m=3.5, from_arm=1, to_arm=2),  # 太短 → 失败
            dict(id=2, axis=LineString([(-20, -20), (-40, -50)]), width_m=3.5, from_arm=2, to_arm=3),
        ]
        out = frenet._slip_lanes(slips, B, core, 3.5)
        self.assertEqual([s["id"] for s, _ in out], [0, 2])
        for s, ln in out:
            self.assertEqual((ln["slip_from_arm"], ln["slip_to_arm"]), (s["from_arm"], s["to_arm"]))
            self.assertLess(ln["geom"].distance(s["axis"]), 1e-6)


class ToJsonTest(unittest.TestCase):
    """to_json：分段切点同时给局部米坐标与 WGS84 点；其余几何全部经 to_wgs 转换。"""

    def test_cut_point_projected(self):
        from pyproj import Transformer
        from lanegen.roi import local_crs
        lon0, lat0 = 113.3, 23.1
        tf = Transformer.from_crs(local_crs(lon0, lat0), 4326, always_xy=True).transform
        to_wgs = lambda g: shp_transform(tf, g)
        core = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        cw = dict(id=0, arm=0, role="approach", lanes=[0], n_lanes=1, width_m=3.5, est_lanes=1, arrows=[],
                  polygon=core.buffer(2), axis=LineString([(0, 10), (0, 60)]), mouth=LineString([(-2, 5), (2, 5)]),
                  sections=[dict(section_id=0, s_start=0.0, s_end=20.0, n_lanes=1, lane_ids=[0], cut_xy=(0.0, 10.0),
                                 transition=None, links=[]),
                            dict(section_id=1, s_start=20.0, s_end=50.0, n_lanes=1, lane_ids=[0], cut_xy=(0.0, 30.0),
                                 transition="reconfig", links=[[0, 0]])])
        jm = dict(core_poly=core, arms=[dict(id=0, bearing=0.0, mouths=[LineString([(-2, 5), (2, 5)])],
                                              in_carriageways=[0], out_carriageways=[])],
                  carriageways=[cw], slips=[], obstacles=[], extra_lanes=[], guides=[])
        js = json.loads(json.dumps(model.to_json(jm, to_wgs)))
        for sec in js["carriageways"][0]["sections"]:
            self.assertNotIn("cut_xy", sec)
            lon, lat = sec["cut_pt"]["coordinates"]
            self.assertAlmostEqual(lon, lon0, delta=0.01)
            self.assertAlmostEqual(lat, lat0, delta=0.01)
            self.assertEqual(len(sec["cut_xy_local"]), 2)


class TurnSolverTest(unittest.TestCase):
    """CP-SAT 转向分配：箭头一致时照箭头连；箭头串位（右侧车道标左转）时不允许交叉。"""

    def _run(self, arrows):
        from lanegen.turns import _solve_approach, RANK
        from lanegen.sumo_check import ExitOption
        options = [ExitOption("e" + ch, angle, 0, n, frozenset(ch))
                   for ch, angle, n in [("R", 90, 2), ("S", 0, 3), ("L", -90, 2)]]
        sol = _solve_approach([(set(a), True) for a in arrows], options)
        self.assertIsNotNone(sol)
        by_lane = {}
        for k, ch, edge, j in sol:
            by_lane.setdefault(k, set()).add(ch)
        for k1 in by_lane:
            for k2 in by_lane:
                if k1 < k2:
                    self.assertLessEqual(max(RANK[c] for c in by_lane[k1]), min(RANK[c] for c in by_lane[k2]))
        return by_lane

    def test_consistent(self):
        self.assertEqual(self._run(["R", "S", "S", "L"]), {0: {"R"}, 1: {"S"}, 2: {"S"}, 3: {"L"}})

    def test_swapped_arrows_no_cross(self):
        self._run(["L", "S", "R"])


class ForeignCutTest(unittest.TestCase):
    def test_cut_keeps_core_side(self):
        core = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        lane = LineString([(0, 6), (0, 80)])
        stop = LineString([(-4, 50), (4, 50)])
        g = frenet._cut_at_foreign(lane, [stop], core)
        self.assertLess(max(y for _, y in g.coords), 50.0)


def _gpkg_available():
    from lanegen import roi
    return os.path.exists(roi.GPKG)


@unittest.skipUnless(_gpkg_available(), "没有 GPKG 数据源")
class PipelineContractTest(unittest.TestCase):
    """真实路口跑全流程：契约零错误、导出字段齐全、坐标全是 WGS84。"""

    CASES = [("样例2_标准十字", 113.32536, 23.12337), ("随机2_南沙", 113.49683, 22.78056)]

    def test_cases(self):
        from lanegen import pipeline
        for name, lon, lat in self.CASES:
            with self.subTest(name=name):
                res = pipeline.run(lon, lat, log=lambda *a: None)
                jm = res["model"]
                self.assertEqual(jm.get("contract_errors"), [], jm.get("contract_errors"))
                lanes = res["lanes"]
                for c in jm["carriageways"]:
                    for sec in c.get("sections", []):
                        idx = [lanes[i]["sec_idx"][sec["section_id"]] for i in sec["lane_ids"]]
                        self.assertEqual(sorted(idx), list(range(sec["n_lanes"])))
                with tempfile.TemporaryDirectory() as d:
                    pipeline.export(res, d)
                    js = json.load(open(os.path.join(d, "junction_model.json"), encoding="utf-8"))
                    gj = json.load(open(os.path.join(d, "lanes.geojson"), encoding="utf-8"))
                for f in gj["features"]:
                    for k in ("lane_index", "source", "allow", "slip_from_arm", "slip_to_arm", "carriageway"):
                        self.assertIn(k, f["properties"])
                self.assertEqual(js["contract_errors"], [])
                pts = []
                for c in js["carriageways"]:
                    pts += [Point(s["cut_pt"]["coordinates"]) for s in c["sections"] if s.get("cut_pt")]
                    pts.append(shape(c["polygon"]).centroid)
                for p in pts:
                    self.assertTrue(100 < p.x < 130 and 10 < p.y < 40, (p.x, p.y))
                owned = sorted(i for c in js["carriageways"] for i in c["lanes"]) + \
                    sorted(i for s in js["slips"] for i in s["lanes"]) + sorted(js["extra_lanes"])
                self.assertEqual(sorted(owned), list(range(len(gj["features"]))))


if __name__ == "__main__":
    unittest.main()
