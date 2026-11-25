"""在指定种子周围测量百度道路参考线支持的路臂方位。

从项目根目录运行 python -B -m tools.probe_reference_ports 经度 纬度。
输出物理方位候选和源 FID，用于与路口模型逐臂核对。
"""

import argparse
from collections import defaultdict
import json
import math

import numpy as np
import shapely
from shapely.geometry import Point

from lanegen import roi
from lanegen.boundaries import explode_lines
from lanegen.junction import _tangent_at


RADII_M = (20, 30, 40)
GROUP_SPAN_DEG = 24


def _angular_groups(hits):
    """在最大空隙处展开圆周，使跨正北方向的观测也能归入同一组。"""
    if not hits:
        return []
    hits = sorted(hits)
    bearings = [angle for angle, radius, fid in hits]
    gaps = np.diff([*bearings, bearings[0] + 360])
    start = (int(np.argmax(gaps)) + 1) % len(hits)
    ordered = hits[start:] + [(angle + 360, radius, fid) for angle, radius, fid in hits[:start]]
    groups = []
    for hit in ordered:
        if not groups or hit[0] - groups[-1][0][0] > GROUP_SPAN_DEG:
            groups.append([])
        groups[-1].append(hit)
    return groups


def reference_ports(refs):
    """refs 是已按源显示 Z 筛选的局部米制参考线表，坐标原点是输入种子。"""
    rings = [(radius, Point(0, 0).buffer(radius, quad_segs=64).boundary) for radius in RADII_M]
    hits = []
    for fid, geometry in zip(refs["fid"], refs.geometry):
        for line in explode_lines([geometry]):
            for radius, ring in rings:
                for p in shapely.get_parts(line.intersection(ring)):
                    if p.geom_type != "Point" or p.is_empty:
                        continue
                    ray = np.array([p.x, p.y])
                    tangent = _tangent_at(line, line.project(p), h=2)
                    if abs(np.dot(tangent, ray)) < 0.6 * np.linalg.norm(ray):
                        continue
                    angle = math.degrees(math.atan2(p.x, p.y)) % 360
                    hits.append((angle, radius, int(fid)))

    ports = []
    for group in _angular_groups(hits):
        per_ring = defaultdict(list)
        for angle, radius, fid in group:
            per_ring[radius].append(angle)
        if len(per_ring) < 2:
            continue
        # 每个圆环占一票，减少同一处多条渲染线造成的偏移。
        bearing = float(np.mean([np.median(angles) for angles in per_ring.values()])) % 360
        ports.append(dict(bearing_deg=round(bearing, 1) % 360,
                          radii_m=sorted(per_ring),
                          ref_fids=sorted({fid for angle, radius, fid in group})))
    return sorted(ports, key=lambda port: port["bearing_deg"])


def probe(lon, lat):
    layers, _ = roi.read_roi(lon, lat, radius_m=120)
    z0 = roi.estimate_z0(layers["lines"], (0, 0))
    refs = roi.zfilter(layers, z0)["ref"]
    return dict(seed=[lon, lat], z0=z0, ports=reference_ports(refs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("longitude", type=float)
    parser.add_argument("latitude", type=float)
    args = parser.parse_args()
    print(json.dumps(probe(args.longitude, args.latitude), ensure_ascii=False, indent=2))
