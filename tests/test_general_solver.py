"""独立于真实坐标和留出集的几何反例。"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, box

from lanegen import frenet, model, pipeline, roi, sumo_check, turns


def marked_cross(n):
    """四臂双向路，各进口和出口均有 n 条等宽车道。"""
    width = 3.5
    half = width * n
    crs = roi.local_crs(0, 0)
    road = box(-half, -half, half, half)
    lines, symbols = [], []
    for bearing in (0, 90, 180, 270):
        def rot(g):
            return rotate(g, -bearing, origin=(0, 0))

        def mark(coords, style):
            lines.append(dict(geometry=rot(LineString(coords)), 源样式ID=style, 原始属性JSON="{}",
                              源显示Z最小=0., 源显示Z最大=0.))

        road = road.union(rot(box(-half, half, half, half + 100)))
        mark([(-half, half), (0, half)], 501775)
        for i in range(1 - n, n):
            mark([(i * width, half), (i * width, half + 100)], 503970 if i == 0 else 503969)
        for i in range(n):
            symbols.append(dict(geometry=rot(Point(-(i + 0.5) * width, half + 18).buffer(0.25)),
                                分类代码="arrow_straight", 要素类型="符号轮廓", 旋转角度=(bearing + 180) % 360, 原始类别=0))
    symbols.append(dict(geometry=road, 分类代码="road_surface", 要素类型="路面", 旋转角度=np.nan, 原始类别=15))
    main = gpd.GeoDataFrame(symbols, crs=crs)
    main["源显示Z最小"] = main["源显示Z最大"] = main["源显示Z代表"] = 0.
    marks = gpd.GeoDataFrame(lines, crs=crs)
    return {k: main if k == "main" else marks if k == "lines" else marks.iloc[:0] for k in roi.LAYERS}, crs


class WideApproachTest(unittest.TestCase):
    def test_complete_four_arm_intersection(self):
        for count in (2, 9, 12):
            with self.subTest(count=count), patch.object(roi, "read_roi", return_value=marked_cross(count)):
                res = pipeline.run(0, 0, log=lambda *args: None)
                jm = res["model"]
                self.assertEqual(res["status"], "identified")
                self.assertEqual(len(jm["arms"]), 4)
                self.assertEqual(jm["contract_errors"], [])
                for arm in jm["arms"]:
                    for role, end in (("approach", -1), ("exit", 0)):
                        total = sum(c["sections"][end]["n_lanes"] for c in jm["carriageways"]
                                    if c["arm"] == arm["id"] and c["role"] == role and c.get("sections"))
                        self.assertEqual(total, count)

    def test_polygon_to_lane_centres(self):
        # 涵盖原 8 车道上限、原横断面射线长度和不同的局部车道宽。
        cases = [(n, w, (19 * n + 37 * j) % 360) for n in range(1, 17) for j, w in enumerate((3.0, 3.5, 4.0))]
        for count, width, angle in cases:
            with self.subTest(count=count, width=width, angle=angle):
                def rot(geom):
                    return rotate(geom, angle, origin=(0, 0))

                half = count * width / 2
                poly = rot(box(-half, 0, half, 54))
                mouth = rot(LineString([(-half, 0), (half, 0)]))
                normal = np.array(rot(LineString([(0, 0), (0, 1)])).coords[1])
                axis, measured = model._trace_poly_axis(poly, mouth, normal, "approach")
                self.assertAlmostEqual(measured, count * width)
                marks = [rot(LineString([(i * width - half, 0), (i * width - half, 54)])) for i in range(1, count)]
                cw = dict(id=0, arm=0, role="approach", polygon=poly, axis=axis)
                bounds = dict(bounds=[(g, "dash") for g in marks], drivable=poly, nondrive=None)
                lanes, cuts = frenet.partition_carriageway(cw, [], bounds, rot(box(-half, -8, half, 0)),
                                                          [mouth], [], [], None, width)
                self.assertEqual(cuts, [])
                self.assertEqual(len(lanes), count)
                xs = sorted(rotate(lane["geom"], -angle, origin=(0, 0)).interpolate(0.5, normalized=True).x
                            for lane in lanes)
                np.testing.assert_allclose(xs, (np.arange(count) + 0.5) * width - half, atol=1e-6)


class ForkTest(unittest.TestCase):
    def test_one_arrow_does_not_confirm_two_unmarked_mouths(self):
        core = box(-5, -15, 5, 15)
        road = box(5, -11, 50, -7).union(box(5, 7, 50, 11))
        arms = [dict(id=i, bearing=90, n=np.array([1., 0.]), width=4, mid=np.array([5., y]),
                     mouths=[LineString([(5, y - 2), (5, y + 2)])]) for i, y in enumerate([-9, 9])]
        one_arrow = [(0, "R")]
        for order in (arms, arms[::-1]):
            confirmed, uncertain = model._minor_exits(order, [], one_arrow, road, core, [])
            self.assertFalse(confirmed)
            self.assertEqual(len(uncertain), 2)
            ref = LineString([(0, -9), (45, -9)])
            confirmed, uncertain = model._minor_exits(order, [], one_arrow, road, core, [ref])
            self.assertEqual([a["id"] for a in confirmed], [0])
            self.assertEqual(len(uncertain), 1)

    def _fork(self, angles, arrows, heading=0, reverse=False, arms=None, lateral=None):
        # 进口朝北，再把整个路口旋转；出口边 ID 保持物理身份。
        axis = rotate(LineString([(0, -60), (0, 0)]), -heading, origin=(0, 0))
        cw = dict(id=0, arm=0, role="approach", axis=axis)
        lanes = [dict(arrows=[(0, ch)] if ch else []) for ch in arrows]
        edges = {"in": dict(lanes=list(range(len(lanes))))}
        cws, exits, bearings = [cw], {}, {0: (heading + 180) % 360}
        for i, angle in enumerate(angles, start=1):
            arm = arms[i-1] if arms else i
            edge = str(i)
            exits[i] = edge
            edges[edge] = dict(lanes=[len(lanes)])
            lanes.append(dict(arrows=[]))
            bearings[arm] = (heading + angle) % 360
            cws.append(dict(id=i, role="exit", arm=arm, d_lat=lateral[i-1] if lateral else 0))
        arm_list = [dict(id=a, bearing=b) for a, b in bearings.items()]
        jm = dict(carriageways=cws[::-1] if reverse else cws, arms=arm_list[::-1] if reverse else arm_list)
        netd = dict(exits=exits, edges=edges, approach={0: "in"})
        return dict(model=jm, lanes=lanes), netd

    def test_two_right_turns_rotation_and_input_order(self):
        expected = [("in", 0, "2", 0), ("in", 1, "1", 0)]
        for heading in (0, 73, 270):
            for reverse in (False, True):
                res, netd = self._fork([80, 100], ["R", "R"], heading, reverse)
                self.assertEqual(turns.cpsat_connections(res, netd), expected)
                self.assertEqual(sumo_check.heuristic_connections(res, netd), expected)

    def test_more_exits_than_turn_lanes(self):
        res, netd = self._fork([80, 100], ["R"])
        self.assertEqual(turns.cpsat_connections(res, netd), [("in", 0, "1", 0), ("in", 0, "2", 0)])

    def test_parallel_carriageways_and_unrelated_left(self):
        res, netd = self._fork([90, 90, -90], ["R", "R"], arms=[1, 1, 2], lateral=[4, -4, 0])
        expected = [("in", 0, "2", 0), ("in", 1, "1", 0)]
        self.assertEqual(turns.cpsat_connections(res, netd), expected)
        with patch.object(turns, "_solve_approach", return_value=None):
            self.assertEqual(turns.cpsat_connections(res, netd), expected)

    def test_unmarked_five_arm_junction(self):
        res, netd = self._fork([-90, 0, 80, 100], ["", "", "", ""])
        connections = turns.cpsat_connections(res, netd)
        self.assertEqual({edge for _, _, edge, _ in connections}, {"1", "2", "3", "4"})
        self.assertEqual({k for _, k, _, _ in connections}, set(range(4)))
        ranks = {"4": 0, "3": 1, "2": 2, "1": 3}
        for _, k, a, _ in connections:
            for _, h, b, _ in connections:
                if k < h:
                    self.assertLessEqual(ranks[a], ranks[b])

    @unittest.skipUnless(all(shutil.which(sumo_check._tool(name)) for name in ("netconvert", "sumo")),
                         "需要 netconvert 和 sumo")
    def test_compiled_fork_reaches_both_exits(self):
        from shapely.affinity import translate
        axis = LineString([(0, -60), (0, -10)])
        lanes = [dict(geom=translate(axis, xoff=x), arrows=[(0, "R")], width=3.5) for x in (1.75, -1.75)]
        cws, arms = [], []
        for i, bearing in enumerate([180, 80, 100]):
            ids = [0, 1]
            line = axis
            if i:
                line = rotate(LineString([(0, 10), (0, 60)]), -bearing, origin=(0, 0))
                ids = [len(lanes)]
                lanes.append(dict(geom=line, width=3.5, arrows=[]))
            section = dict(s_start=0, s_end=line.length, lane_ids=ids, section_id=0)
            cws.append(dict(id=i, arm=i, axis=line, role="exit" if i else "approach", lanes=ids, sections=[section]))
            arms.append(dict(id=i, bearing=bearing, out_carriageways=[i] if i else []))
        jm = dict(core_poly=box(-10, -10, 10, 10), carriageways=cws, arms=arms, slips=[])
        report = sumo_check.check(dict(model=jm, lanes=lanes, status="identified"))
        self.assertTrue(report["net_ok"])
        self.assertEqual(report["fallback_approaches"], 0)
        self.assertEqual(report["route_pairs"], 2)
        self.assertEqual(report["route_arrived"], 2)
        self.assertEqual(report["turn_cross"], 0)
        self.assertEqual(report["candidate_mv_cover"], 1)


class RecognitionTest(unittest.TestCase):
    def test_identity_and_incomplete_models(self):
        def status(roles, stops):
            cws = [dict(arm=arm, role=role, lanes=[i]) for i, (arm, role) in enumerate(roles)]
            return pipeline._recognition(dict(carriageways=cws, extra_lanes=[0]), stops, True)[0]
        self.assertEqual(status([], []), "unresolved")
        self.assertEqual(status([(0, "approach"), (1, "exit")], []), "unresolved")
        self.assertEqual(status([(0, "approach")], [object()]), "partial")
        self.assertEqual(status([(0, "approach"), (1, "exit"), (2, "exit")], []), "identified")

    def test_empty_input_export(self):
        crs = roi.local_crs(0, 0)
        cols = ["源样式ID", "原始属性JSON", "源显示Z最小", "源显示Z最大", "源显示Z代表", "分类代码",
                "原始类别", "要素类型", "旋转角度", "geometry"]
        layers = {k: gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs) for k in roi.LAYERS}
        with patch.object(roi, "read_roi", return_value=(layers, crs)):
            res = pipeline.run(0, 0, log=lambda *args: None)
        self.assertEqual(res["status"], "no_data")
        self.assertFalse(sumo_check.check(res)["net_ok"])
        with tempfile.TemporaryDirectory() as out:
            report = pipeline.export(res, out)
            self.assertEqual(report["status_reason"], "empty_roi")
            for filename in ("report.json", "junction_model.json"):
                self.assertEqual(json.loads((Path(out) / filename).read_text())["status"], "no_data")
            self.assertIsNone(json.loads((Path(out) / "junction_model.json").read_text())["core_poly"])
            self.assertEqual(json.loads((Path(out) / "lanes.geojson").read_text())["features"], [])
