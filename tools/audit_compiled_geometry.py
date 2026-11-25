"""重跑路口，回读 SUMO 实际车道几何，与识别车道和同层原始岛区对照。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile

import numpy as np
import sumolib
from shapely.geometry import LineString, Point

from lanegen import evaluate, pipeline, sumo_check
from lanegen.turns import cpsat_connections


def audit(case):
    result = dict(fid=case["fid"], district=case["district"])
    res = pipeline.run(case["lon"], case["lat"], log=lambda *args: None)
    road = sumo_check.build_network(res)
    result.update(status=res["status"], compiled=False, source_on_mark_m=evaluate.on_raw_marking_m(res))
    if not road["edges"]:
        return dict(result, reason="no_edges")
    with tempfile.TemporaryDirectory() as directory:
        turns = cpsat_connections(res, road)
        ok, warnings, details = sumo_check.compile_net(road, turns, directory, res["lanes"])
        if not ok:
            return dict(result, reason=details.strip())
        net = sumolib.net.readNet(str(Path(directory) / "net.net.xml"), withInternal=True)

    zone = evaluate._nondrive_zone(res["roi"]["main"])
    inner = zone.buffer(-0.5) if zone is not None else None
    has_zone = inner is not None and not inner.is_empty
    normal, penetration, end_errors = [], [], []
    for edge in net.getEdges(withInternal=True):
        eid = edge.getID()
        if edge.getFunction() == "internal":
            # v20 的主冲突节点命名为 J。分流、汇流和车道数变化节点另有名称。
            if eid.startswith(":J_"):
                for lane in edge.getLanes():
                    line = LineString(lane.getShape())
                    penetration.append(line.intersection(inner).length if has_zone else 0.0)
            continue
        lane_ids = road["edges"][eid]["lanes"]
        for lane in edge.getLanes():
            line = LineString(lane.getShape())
            normal.append(dict(geom=line))
            source = res["lanes"][lane_ids[lane.getIndex()]]["geom"]
            if eid in road["approach"].values():
                end_errors.append(Point(line.coords[-1]).distance(Point(source.coords[-1])))
            if eid in road["exits"].values():
                end_errors.append(Point(line.coords[0]).distance(Point(source.coords[0])))
    return dict(result, compiled=True, warnings=len(warnings),
                net_on_mark_m=evaluate.on_raw_marking_m(dict(res, lanes=normal)),
                has_obstacle_in_roi=has_zone, main_internal_segments=len(penetration),
                penetrating_segments=sum(d > 1.0 for d in penetration),
                penetration_m=round(sum(penetration), 2), mouth_lane_ends=len(end_errors),
                mouth_p95_m=round(float(np.percentile(end_errors, 95)), 2) if end_errors else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("data/评测/留出集.json"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.workers == 1:
        for case in cases:
            print(json.dumps(audit(case), ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as workers:
            for record in workers.map(audit, cases):
                print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
