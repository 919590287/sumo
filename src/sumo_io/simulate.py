"""确定性 SUMO 仿真测试台：需求生成 + sumo 运行 + 统计解析。

- probe：稀疏全可达 OD 探测流（每个合法 (进口 edge → 出口 edge) 转向至少有车跑通）；
- conflict：中等稳态冲突车流；
- sparse_single：交付版 site.net.xml 上逐车间隔放行的无冲突全可达探测。
需求完全由 G 的拓扑（E_succ 与车道权限）枚举得到，不依赖任何几何规则。
"""
from __future__ import annotations

import random
import subprocess
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import load_config, sumo_tool
from src.graph.lane_graph import LaneGraph
from src.sumo_io.serializer import SumoIds

PERM_CLASSES = {"general": {"car", "bus"}, "variable": {"car", "bus"}, "reversible": {"car", "bus"},
                "bus_only": {"bus"}}


@dataclass
class SimStats:
    loaded: int = 0
    inserted: int = 0
    arrived: int = 0
    running: int = 0
    waiting: int = 0
    teleports: int = 0
    collisions: int = 0
    emergency_braking: int = 0
    lane_changes: int = 0
    lane_change_from: list[str] = field(default_factory=list)
    returncode: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def arrival_rate(self) -> float:
        return self.arrived / self.loaded if self.loaded else 0.0


def _edge_graph(g: LaneGraph, vclass: str) -> dict[int, set[int]]:
    """vClass 可行的 edge 级后继图（连接两端车道均允许该类车辆）。"""
    e_of = g.edge_of_lane()
    lanes = {l.id: l for l in g.lanes}
    adj: dict[int, set[int]] = {e.id: set() for e in g.edges}
    for c in g.connections:
        if vclass in PERM_CLASSES[lanes[c.from_lane].permission] and \
                vclass in PERM_CLASSES[lanes[c.to_lane].permission]:
            adj[e_of[c.from_lane].id].add(e_of[c.to_lane].id)
    return adj


def _bfs(adj, src: int) -> dict[int, int | None]:
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in prev:
                prev[v] = u
                q.append(v)
    return prev


def _path(prev, dst) -> list[int]:
    out = [dst]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return out[::-1]


def movement_routes(g: LaneGraph) -> list[tuple[str, list[int]]]:
    """为每个 edge 级转向 (e_in → e_out) 生成一条 入口端口 → … → 出口端口 的完整路径。"""
    nodes = {n.id: n for n in g.nodes}
    entries = [e.id for e in g.edges if nodes[e.from_node].node_type == "roi_entry_port"]
    exits = {e.id for e in g.edges if nodes[e.to_node].node_type == "roi_exit_port"}
    routes, seen = [], set()
    for vclass in ("car", "bus"):
        adj = _edge_graph(g, vclass)
        reach = {s: _bfs(adj, s) for s in entries}
        for a, outs in sorted(adj.items()):
            for b in sorted(outs):
                if (a, b) in seen:
                    continue
                src = next((s for s in entries if a in reach[s]), None)
                if src is None:
                    continue
                tail = _bfs(adj, b)
                dst = next((x for x in sorted(exits) if x in tail), None)
                if dst is None:
                    continue
                routes.append((vclass, _path(reach[src], a) + _path(tail, dst)))
                seen.add((a, b))
    return routes


def od_pairs(g: LaneGraph) -> list[tuple[str, list[int]]]:
    """全部可达 (入口端口 edge, 出口端口 edge) OD 的一条路径。"""
    nodes = {n.id: n for n in g.nodes}
    entries = [e.id for e in g.edges if nodes[e.from_node].node_type == "roi_entry_port"]
    exits = sorted(e.id for e in g.edges if nodes[e.to_node].node_type == "roi_exit_port")
    out, seen = [], set()
    for vclass in ("car", "bus"):
        adj = _edge_graph(g, vclass)
        for s in entries:
            prev = _bfs(adj, s)
            for t in exits:
                if t in prev and (s, t) not in seen:
                    out.append((vclass, _path(prev, t)))
                    seen.add((s, t))
    return out


def _vtypes_xml() -> str:
    vt = load_config("sim")["vtypes"]
    return "".join(f'    <vType id="{k}" ' + " ".join(f'{a}="{b}"' for a, b in v.items()) + "/>\n"
                   for k, v in vt.items())


def write_demand(g: LaneGraph, ids: SumoIds, out: Path, mode: str, seed: int) -> Path:
    cfg = load_config("sim")
    rnd = random.Random(seed)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n', _vtypes_xml()]
    veh = []
    if mode in ("probe", "sparse_single"):
        routes = movement_routes(g)
        reps = cfg["probe"]["vehicles_per_movement"] if mode == "probe" else 1
        hw = cfg["probe"]["headway_s"] if mode == "probe" else cfg["sparse_single"]["headway_s"]
        order = [(vc, r) for vc, r in routes for _ in range(reps)]
        if mode == "probe":
            rnd.shuffle(order)
        for k, (vc, r) in enumerate(order):
            veh.append((k * hw, vc, r))
    elif mode == "conflict":
        c = cfg["conflict"]
        ods = od_pairs(g)
        lanes_per_entry = {e.id: len(e.lanes) for e in g.edges}
        by_entry: dict[int, list] = {}
        for vc, r in ods:
            by_entry.setdefault(r[0], []).append((vc, r))
        for src, lst in sorted(by_entry.items()):
            rate = c["veh_per_hour_per_entry_lane"] * lanes_per_entry[src] / 3600.0
            t = rnd.expovariate(rate)
            while t < c["duration_s"]:
                vc, r = lst[rnd.randrange(len(lst))]
                veh.append((t, vc, r))
                t += rnd.expovariate(rate)
    else:
        raise ValueError(mode)
    veh.sort(key=lambda x: x[0])
    for k, (t, vc, r) in enumerate(veh):
        edges = " ".join(ids.edge[e] for e in r)
        lines.append(f'    <vehicle id="v{k}" type="{vc}" depart="{t:.2f}" departLane="best" departSpeed="max">\n'
                     f'        <route edges="{edges}"/>\n    </vehicle>\n')
    lines.append("</routes>\n")
    out.write_text("".join(lines), encoding="utf-8")
    return out


def parse_statistics(path: Path) -> SimStats:
    s = SimStats()
    root = ET.parse(path).getroot()
    v = root.find("vehicles")
    if v is not None:
        s.loaded = int(v.get("loaded", 0))
        s.inserted = int(v.get("inserted", 0))
        s.running = int(v.get("running", 0))
        s.waiting = int(v.get("waiting", 0))
    t = root.find("teleports")
    if t is not None:
        s.teleports = int(t.get("total", 0))
    sf = root.find("safety")
    if sf is not None:
        s.collisions = int(sf.get("collisions", 0))
        s.emergency_braking = int(sf.get("emergencyBraking", 0))
    ts = root.find("vehicleTripStatistics")
    if ts is not None:
        s.arrived = int(ts.get("count", 0))
    return s


def run_sumo(net: Path, routes: Path, out_dir: Path, seed: int, tag: str,
             lanechange: bool = False, extra: list[str] | None = None, timeout: float = 600) -> SimStats:
    cfg = load_config("sim")
    out_dir = Path(out_dir)
    stat = out_dir / f"stats_{tag}_{seed}.xml"
    cmd = [sumo_tool("sumo"), "-n", str(net), "-r", str(routes), "--seed", str(seed),
           "--end", str(cfg["end_s"]), "--statistic-output", str(stat), *cfg["sumo_args"]]
    lc = out_dir / f"lanechange_{tag}_{seed}.xml"
    if lanechange:
        cmd += ["--lanechange-output", str(lc)]
    if extra:
        cmd += extra
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    s = parse_statistics(stat) if stat.exists() else SimStats()
    s.returncode = p.returncode
    s.errors = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith("Error")]
    if lanechange and lc.exists():
        s.lane_change_from = [c.get("from", "") for c in ET.parse(lc).getroot().iter("change")]
        s.lane_changes = len(s.lane_change_from)
    return s
