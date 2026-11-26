"""LaneGraph → SUMO plain XML（.nod.xml / .edg.xml / .con.xml）的 1:1 确定性转写。

docs/DESIGN.md §4.3：纯语法转写，严禁几何补猜与拓扑修补。唯一的“构造”行为均为
SUMO 编译器语法接口例外（AGENTS.md 第 3 节第 6 条）：
- 非 junction_core 内部节点的闭合 shape 由其接入车道的端点横断面角点按角序装配；
- 被人行横道跨越的 edge 写出 index 0 辅助人行道；单向跨越在远端挂载岛侧辅助锚点边；
- 出度为 0 的非 roi_exit_port edge 写出 ``<connection from="e"/>`` 禁止自动补连。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from src.common.config import load_config
from src.common.geom import left_normals, offset_polyline
from src.graph.lane_graph import LaneGraph


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _shape(points) -> str:
    p = np.asarray(points, dtype=np.float64)
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in p[:, :2])


@dataclass
class SumoIds:
    """G 实体 ↔ SUMO id 的确定性映射（往返核验使用）。"""
    node: dict[int, str] = field(default_factory=dict)
    edge: dict[int, str] = field(default_factory=dict)
    lane_offset: dict[int, int] = field(default_factory=dict)      # edge id → 机动车道索引偏移（辅助人行道）
    aux_edges: list[str] = field(default_factory=list)
    aux_nodes: list[str] = field(default_factory=list)
    junction_core_nodes: list[str] = field(default_factory=list)

    def lane_sumo_id(self, g: LaneGraph, lid: int) -> str:
        e = g.edge_of_lane()[lid]
        return f"{self.edge[e.id]}_{g.lane_index()[lid] + self.lane_offset[e.id]}"


@dataclass
class PlainXml:
    nod: str
    edg: str
    con: str
    ids: SumoIds

    def write(self, out_dir: Path, stem: str = "site") -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {k: out_dir / f"{stem}.{k}.xml" for k in ("nod", "edg", "con")}
        for k, p in paths.items():
            p.write_text(getattr(self, k), encoding="utf-8")
        return paths


class SumoSerializer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config("sumo_export")

    # ------------------------------------------------------------------
    def serialize(self, g: LaneGraph) -> PlainXml:
        errs = g.validate()
        if errs:
            raise ValueError("输入图不合法（须先经约束图解码器）：" + "; ".join(errs[:5]))
        c = self.cfg
        pre = c["id_prefix"]
        ids = SumoIds()
        for n in g.nodes:
            ids.node[n.id] = f"{pre['node']}{n.id}"
            if n.node_type == "junction_core":
                ids.junction_core_nodes.append(ids.node[n.id])
        for e in g.edges:
            ids.edge[e.id] = f"{pre['edge']}{e.id}"

        lanes = {l.id: l for l in g.lanes}
        crossed_edges = sorted({eid for cw in g.crosswalks for eid in cw.edges})
        for e in g.edges:
            ids.lane_offset[e.id] = 1 if e.id in crossed_edges else 0

        sidewalk_shapes = {eid: self._aux_sidewalk_shape(g, g.edge(eid), lanes) for eid in crossed_edges}
        anchors = self._anchor_edges(g, lanes, sidewalk_shapes, ids)
        node_shapes = self._node_shapes(g, lanes, sidewalk_shapes, anchors)

        nod = self._write_nodes(g, ids, node_shapes, anchors)
        edg = self._write_edges(g, ids, lanes, sidewalk_shapes, anchors)
        con = self._write_connections(g, ids)
        return PlainXml(nod=nod, edg=edg, con=con, ids=ids)

    # ------------------------------------------------------------------ 几何装配（纯 I/O）
    @staticmethod
    def _lane_end(lane, at_end: bool):
        """车道在端点处的 (中心点, 左法向)。"""
        pts = lane.points[:, :2]
        nrm = left_normals(pts)
        k = -1 if at_end else 0
        return pts[k], nrm[k]

    def _aux_sidewalk_shape(self, g: LaneGraph, e, lanes) -> np.ndarray:
        l0 = lanes[e.lanes[0]]
        w = self.cfg["aux_sidewalk"]["width"]
        return offset_polyline(l0.points, -(l0.width / 2 + w / 2))[:, :2]

    def _edge_corners(self, g, e, lanes, at_node: int, sidewalk_shapes) -> list[np.ndarray]:
        """edge 在节点 at_node 处横断面的两侧外角点（含辅助人行道外缘）。"""
        at_end = e.to_node == at_node
        r_lane = lanes[e.lanes[0]]
        l_lane = lanes[e.lanes[-1]]
        pr, nr = self._lane_end(r_lane, at_end)
        pl, nl = self._lane_end(l_lane, at_end)
        right = pr - nr * r_lane.width / 2
        if e.id in sidewalk_shapes:
            right = right - nr * self.cfg["aux_sidewalk"]["width"]
        left = pl + nl * l_lane.width / 2
        return [right, left]

    def _node_shapes(self, g, lanes, sidewalk_shapes, anchors) -> dict[int, np.ndarray | None]:
        """junction_core 写模型预测轮廓；其余节点（含端口）由接入 edge 端点横断面角点按角序装配。"""
        shapes: dict[int, np.ndarray | None] = {}
        for n in g.nodes:
            if n.node_type == "junction_core":
                shapes[n.id] = np.asarray(n.polygon)[:, :2]
                continue
            pts = []
            for e in g.edges:
                if n.id in (e.from_node, e.to_node):
                    pts.extend(self._edge_corners(g, e, lanes, n.id, sidewalk_shapes))
            pts = np.asarray(pts)
            if len(pts) == 2:                  # 端口：单一横断面线段
                shapes[n.id] = pts
                continue
            ctr = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
            order = np.argsort(ang, kind="stable")
            ring = pts[order]
            keep = [0]
            for k in range(1, len(ring)):
                if np.linalg.norm(ring[k] - ring[keep[-1]]) > 1e-3:
                    keep.append(k)
            shapes[n.id] = ring[keep]
        return shapes

    def _anchor_edges(self, g, lanes, sidewalk_shapes, ids) -> list[dict]:
        """单向跨越的人行横道在远端（岛侧）挂载辅助行人锚点边。

        远端落脚点所在的节点角扇区（相邻两条接入 edge 之间）即岛侧落脚区；
        同一扇区内的全部远端落脚点共用一条锚点边，沿扇区角平分线向外挂载。
        """
        cfg = self.cfg["aux_anchor_edge"]
        pre = self.cfg["id_prefix"]
        groups: dict[tuple[int, int], list[np.ndarray]] = {}
        sectors: dict[tuple[int, int], float] = {}
        for cw in g.crosswalks:
            es = [g.edge(eid) for eid in cw.edges]
            inc = [e for e in es if e.to_node == cw.node]
            outg = [e for e in es if e.from_node == cw.node]
            if inc and outg:
                continue                      # 双向跨越：两端均落在各自 edge 的辅助人行道上
            e = (inc or outg)[0]
            at_end = e.to_node == cw.node
            r_lane = lanes[e.lanes[0]]
            pr, nr = self._lane_end(r_lane, at_end)
            right_pt = pr - nr * (r_lane.width / 2)
            p0, p1 = cw.endpoints[0, :2], cw.endpoints[1, :2]
            far = p1 if np.linalg.norm(p0 - right_pt) <= np.linalg.norm(p1 - right_pt) else p0
            sec, bis = self._sector(g, lanes, cw.node, far)
            groups.setdefault((cw.node, sec), []).append(far)
            sectors[(cw.node, sec)] = bis
        out = []
        for k, (key, fars) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
            d = np.array([math.cos(sectors[key]), math.sin(sectors[key])])
            start = np.mean(fars, axis=0)
            end = start + d * cfg["length"]
            out.append({"node": key[0], "edge_id": f"{pre['aux_edge']}{k}",
                        "anchor_node_id": f"{pre['aux_node']}{k}",
                        "shape": np.stack([start, end]), "end": end})
            ids.aux_edges.append(out[-1]["edge_id"])
            ids.aux_nodes.append(out[-1]["anchor_node_id"])
        return out

    def _sector(self, g, lanes, nid: int, p: np.ndarray) -> tuple[int, float]:
        """点 p 所在的节点角扇区（由接入 edge 横断面中点方位划分）及其角平分线方位。"""
        c = g.node(nid).pos[:2]
        angs = []
        for e in g.edges:
            if nid in (e.from_node, e.to_node):
                at_end = e.to_node == nid
                mids = [self._lane_end(lanes[l], at_end)[0] for l in e.lanes]
                m = np.mean(mids, axis=0)
                angs.append(math.atan2(m[1] - c[1], m[0] - c[0]))
        angs = sorted(a % (2 * math.pi) for a in angs)
        ap = math.atan2(p[1] - c[1], p[0] - c[0]) % (2 * math.pi)
        for k in range(len(angs)):
            lo = angs[k]
            hi = angs[(k + 1) % len(angs)] + (2 * math.pi if k == len(angs) - 1 else 0.0)
            x = ap if ap >= lo else ap + 2 * math.pi
            if lo <= x < hi:
                return k, ((lo + hi) / 2) % (2 * math.pi)
        return 0, (angs[0] + math.pi) % (2 * math.pi)

    # ------------------------------------------------------------------ XML 输出
    def _location(self, g: LaneGraph) -> str:
        pts = np.concatenate([l.points[:, :2] for l in g.lanes])
        x0, y0 = g.origin
        xmin, ymin = pts.min(0)
        xmax, ymax = pts.max(0)
        conv = f"{_fmt(xmin)},{_fmt(ymin)},{_fmt(xmax)},{_fmt(ymax)}"
        orig = f"{_fmt(xmin + x0)},{_fmt(ymin + y0)},{_fmt(xmax + x0)},{_fmt(ymax + y0)}"
        proj = self.cfg["projection"]["proj_parameter"]
        return (f'    <location netOffset="{_fmt(-x0)},{_fmt(-y0)}" convBoundary="{conv}" '
                f'origBoundary="{orig}" projParameter={quoteattr(proj)}/>\n')

    def _write_nodes(self, g, ids, node_shapes, anchors) -> str:
        c = self.cfg
        key = c["aux_param_key"]
        lines = ['<?xml version="1.0" encoding="UTF-8"?>\n',
                 '<nodes version="1.20" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n',
                 self._location(g)]
        for n in g.nodes:
            t = c["node_type_map"][n.node_type]
            attrs = f'id="{ids.node[n.id]}" x="{_fmt(n.pos[0])}" y="{_fmt(n.pos[1])}" type="{t}"'
            if n.is_port:
                attrs += f' fringe="{c["port_fringe"]}"'
            shp = node_shapes.get(n.id)
            if shp is not None:
                attrs += f' shape="{_shape(shp)}"'
            lines.append(f'    <node {attrs}>\n        <param key="node_type" value="{n.node_type}"/>\n    </node>\n')
        for a in anchors:
            x, y = a["end"]
            lines.append(f'    <node id="{a["anchor_node_id"]}" x="{_fmt(x)}" y="{_fmt(y)}" type="dead_end">\n'
                         f'        <param key="{key}" value="true"/>\n    </node>\n')
        lines.append("</nodes>\n")
        return "".join(lines)

    def _write_edges(self, g, ids, lanes, sidewalk_shapes, anchors) -> str:
        c = self.cfg
        d = c["edge_defaults"]
        forbid = c["lane_change_forbidden"]
        key = c["aux_param_key"]
        lines = ['<?xml version="1.0" encoding="UTF-8"?>\n',
                 '<edges version="1.20" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n']
        for e in g.edges:
            off = ids.lane_offset[e.id]
            ll = lanes[e.lanes[-1]]
            edge_shape = offset_polyline(ll.points, ll.width / 2)
            lines.append(
                f'    <edge id="{ids.edge[e.id]}" from="{ids.node[e.from_node]}" to="{ids.node[e.to_node]}" '
                f'numLanes="{len(e.lanes) + off}" speed="{d["speed"]}" priority="{d["priority"]}" '
                f'spreadType="{d["spread_type"]}" shape="{_shape(edge_shape)}">\n')
            if off:
                sw = c["aux_sidewalk"]["width"]
                lines.append(f'        <lane index="0" width="{sw}" allow="pedestrian" '
                             f'shape="{_shape(sidewalk_shapes[e.id])}">\n'
                             f'            <param key="{key}" value="true"/>\n        </lane>\n')
            for k, lid in enumerate(e.lanes):
                l = lanes[lid]
                perm = c["permission"][l.permission]
                attrs = f'index="{k + off}" width="{l.width:.4f}"'
                if "allow" in perm:
                    attrs += f' allow="{perm["allow"]}"'
                if "disallow" in perm:
                    attrs += f' disallow="{perm["disallow"]}"'
                if not l.allow_change_left:
                    attrs += f' changeLeft="{forbid}"'
                if not l.allow_change_right:
                    attrs += f' changeRight="{forbid}"'
                attrs += f' shape="{_shape(l.points)}"'
                params = [("g_lane_id", str(lid)), ("role", l.role)]
                if "lane_class" in perm:
                    params.append(("lane_class", perm["lane_class"]))
                body = "".join(f'            <param key="{pk}" value="{pv}"/>\n' for pk, pv in params)
                lines.append(f'        <lane {attrs}>\n{body}        </lane>\n')
            lines.append("    </edge>\n")
        ac = c["aux_anchor_edge"]
        for a in anchors:
            lines.append(
                f'    <edge id="{a["edge_id"]}" from="{ids.node[a["node"]]}" to="{a["anchor_node_id"]}" '
                f'numLanes="1" speed="{d["speed"]}" priority="{d["priority"]}" spreadType="{ac["spread_type"]}" '
                f'width="{ac["width"]}" allow="pedestrian" shape="{_shape(a["shape"])}">\n'
                f'        <param key="{key}" value="true"/>\n    </edge>\n')
        lines.append("</edges>\n")
        return "".join(lines)

    def _write_connections(self, g, ids) -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>\n',
                 '<connections version="1.20" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n']
        e_of = g.edge_of_lane()
        li = g.lane_index()
        has_out = {e_of[c.from_lane].id for c in g.connections}
        for c in g.connections:
            ei, ej = e_of[c.from_lane], e_of[c.to_lane]
            fl = li[c.from_lane] + ids.lane_offset[ei.id]
            tl = li[c.to_lane] + ids.lane_offset[ej.id]
            lines.append(f'    <connection from="{ids.edge[ei.id]}" to="{ids.edge[ej.id]}" fromLane="{fl}" '
                         f'toLane="{tl}" shape="{_shape(c.points)}"/>\n')
        nodes = {n.id: n for n in g.nodes}
        for e in g.edges:
            if e.id not in has_out and nodes[e.to_node].node_type != "roi_exit_port":
                lines.append(f'    <connection from="{ids.edge[e.id]}"/>\n')
        for cw in g.crosswalks:
            edges = " ".join(ids.edge[eid] for eid in cw.edges)
            lines.append(f'    <crossing node="{ids.node[cw.node]}" edges="{edges}" width="{cw.width:.4f}" '
                         f'shape="{_shape(cw.endpoints)}"/>\n')
        lines.append("</connections>\n")
        return "".join(lines)


def junction_core_ids(g: LaneGraph, ids: SumoIds) -> list[str]:
    types = set(load_config("sumo_export")["eval_tls_node_types"])
    return [ids.node[n.id] for n in g.nodes if n.node_type in types]


def angle_deg(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))
