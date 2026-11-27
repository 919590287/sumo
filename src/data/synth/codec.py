"""合成样本紧凑编码（docs/DESIGN.md §5 Layer 1-3「离线存轻量矢量与真值图」）。

- 要素几何：按图层 / 几何类型分组 ``shapely.to_ragged_array``，坐标量化为 int32 厘米
  （相对场景 UTM 偏移），偏移数组 int32；分类代码以词表下标存储；
- 真值图 G*：float32 定长数组（车道 32 点、连接 8 点、轮廓 32 点）；
- 整体 pickle + zlib。单样本典型 15–40 KB。
"""
from __future__ import annotations

import pickle
import zlib
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import MultiLineString, MultiPolygon

from src.data.synth.render import FeatureSet
from src.graph.lane_graph import (LANE_ROLES, NODE_TYPES, PERMISSIONS, Connection, Crosswalk, Edge, Lane,
                                  LaneGraph, Node)

FORMAT_VERSION = 1


def _as_multi(g):
    if g.geom_type == "Polygon":
        return MultiPolygon([g])
    if g.geom_type == "LineString":
        return MultiLineString([g])
    return g


def _pack_geoms(geoms: list) -> dict | None:
    if not geoms:
        return None
    geoms = np.asarray([_as_multi(g) for g in geoms], dtype=object)
    typ, coords, offsets = shapely.to_ragged_array(geoms)
    return {"type": int(typ), "xy": np.round(coords * 100).astype(np.int32),
            "off": [np.asarray(o, np.int32) for o in offsets]}


def _unpack_geoms(d: dict | None) -> list:
    if d is None:
        return []
    return list(shapely.from_ragged_array(shapely.GeometryType(d["type"]), d["xy"].astype(np.float64) / 100.0,
                                          tuple(np.asarray(o, np.int64) for o in d["off"])))


@dataclass
class SampleRecord:
    graph: LaneGraph          # 场景局部坐标，origin = 场景 UTM 偏移
    features: FeatureSet      # 场景局部坐标
    meta: dict


# ====================================================================== 图
def pack_graph(g: LaneGraph) -> dict:
    nodes = g.nodes
    lanes = g.lanes
    return {
        "origin": tuple(g.origin),
        "node_id": np.array([n.id for n in nodes], np.int32),
        "node_pos": np.array([n.pos[:2] for n in nodes], np.float32).reshape(-1, 2),
        "node_type": np.array([NODE_TYPES.index(n.node_type) for n in nodes], np.int8),
        "node_poly": {n.id: n.polygon.astype(np.float32) for n in nodes if n.polygon is not None},
        "lane_id": np.array([l.id for l in lanes], np.int32),
        "lane_pts": np.array([l.points[:, :2] for l in lanes], np.float32).reshape(-1, lanes[0].points.shape[0], 2),
        "lane_w": np.array([l.width for l in lanes], np.float32),
        "lane_role": np.array([LANE_ROLES.index(l.role) for l in lanes], np.int8),
        "lane_perm": np.array([PERMISSIONS.index(l.permission) for l in lanes], np.int8),
        "lane_flags": np.array([[l.allow_change_left, l.allow_change_right, l.is_stop_line_end] for l in lanes],
                               np.bool_),
        "edges": [(e.id, list(e.lanes), e.from_node, e.to_node) for e in g.edges],
        "conn_ij": np.array([[c.from_lane, c.to_lane] for c in g.connections], np.int32).reshape(-1, 2),
        "conn_pts": np.array([c.points[:, :2] for c in g.connections], np.float32).reshape(-1, 8, 2),
        "cw": [(c.id, c.endpoints[:, :2].astype(np.float32), float(c.width), c.node, list(c.edges))
               for c in g.crosswalks],
        "meta": dict(g.meta),
    }


def unpack_graph(d: dict) -> LaneGraph:
    nodes = []
    for k, nid in enumerate(d["node_id"]):
        nid = int(nid)
        poly = d["node_poly"].get(nid)
        nodes.append(Node(nid, d["node_pos"][k].astype(np.float64), NODE_TYPES[int(d["node_type"][k])],
                          None if poly is None else poly.astype(np.float64)))
    lanes = []
    for k, lid in enumerate(d["lane_id"]):
        f = d["lane_flags"][k]
        lanes.append(Lane(int(lid), d["lane_pts"][k].astype(np.float64), float(d["lane_w"][k]),
                          LANE_ROLES[int(d["lane_role"][k])], PERMISSIONS[int(d["lane_perm"][k])],
                          bool(f[0]), bool(f[1]), bool(f[2])))
    edges = [Edge(int(i), [int(x) for x in ls], int(a), int(b)) for i, ls, a, b in d["edges"]]
    conns = [Connection(int(i), int(j), d["conn_pts"][k].astype(np.float64)) for k, (i, j) in enumerate(d["conn_ij"])]
    cws = [Crosswalk(int(i), ep.astype(np.float64), w, int(n), [int(x) for x in es]) for i, ep, w, n, es in d["cw"]]
    return LaneGraph(nodes, lanes, edges, conns, cws, origin=tuple(d["origin"]), meta=dict(d["meta"]))


# ====================================================================== 要素
def pack_features(fs: FeatureSet) -> dict:
    vocab = sorted({r[0] for r in fs.road} | {c for c, _ in fs.other_area})
    vi = {c: k for k, c in enumerate(vocab)}
    return {
        "vocab": vocab,
        "road_code": np.array([vi[r[0]] for r in fs.road], np.int16),
        "road_attr": np.array([[r[2], r[3], r[4]] for r in fs.road], np.float32).reshape(-1, 3),
        "road": _pack_geoms([r[1] for r in fs.road]),
        "stripes": _pack_geoms(fs.stripes),
        "separation": _pack_geoms(fs.separation),
        "other_code": np.array([vi[c] for c, _ in fs.other_area], np.int16),
        "other": _pack_geoms([g for _, g in fs.other_area]),
        "bus": np.array([[p.x, p.y] for p in fs.bus_stop], np.float32).reshape(-1, 2),
    }


def unpack_features(d: dict) -> FeatureSet:
    from shapely.geometry import Point
    fs = FeatureSet()
    vocab = d["vocab"]
    road = _unpack_geoms(d["road"])
    for k, g in enumerate(road):
        a = d["road_attr"][k]
        fs.road.append((vocab[int(d["road_code"][k])], g, float(a[0]), float(a[1]), float(a[2])))
    fs.stripes = _unpack_geoms(d["stripes"])
    fs.separation = _unpack_geoms(d["separation"])
    fs.other_area = [(vocab[int(c)], g) for c, g in zip(d["other_code"], _unpack_geoms(d["other"]))]
    fs.bus_stop = [Point(float(x), float(y)) for x, y in d["bus"]]
    return fs


def encode_sample(g: LaneGraph, fs: FeatureSet, meta: dict) -> bytes:
    payload = {"v": FORMAT_VERSION, "graph": pack_graph(g), "feat": pack_features(fs), "meta": meta}
    return zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), 6)


def decode_sample(b: bytes) -> SampleRecord:
    d = pickle.loads(zlib.decompress(b))
    if d["v"] != FORMAT_VERSION:
        raise ValueError(f"合成样本格式版本不符：{d['v']}")
    return SampleRecord(unpack_graph(d["graph"]), unpack_features(d["feat"]), d["meta"])
