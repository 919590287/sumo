"""标准化有向车道图 G = (V, L, K_conn, C, E_adj, E_from, E_to, E_succ, E_cross, E_cross_node)。

docs/DESIGN.md §4.1 的数据契约实现：
- ``LaneGraph``：约束图解码后的 100% 合法离散图（SUMO 序列化器的唯一输入）；
- ``GraphTensors``：定长填充张量形式（模型输出概率 / 训练真值 one-hot 均用此结构）。

所有坐标均为窗口局部米制坐标（全局 UTM EPSG:32649 坐标减去 ``origin``），
统一保存 (x, y, z) 三维，Phase 1 z ≡ 0（Phase 2 显示层序预留）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.config import load_config

_GCFG = load_config("graph")
K_LANE: int = _GCFG["k_lane"]
K_CONN: int = _GCFG["k_conn"]
POLY_K: int = _GCFG["poly_k"]
P_MAX: int = _GCFG["p_max"]
N_MAX: int = _GCFG["n_max"]
M_MAX: int = _GCFG["m_max"]
CONN_MAX: int = _GCFG["conn_max"]
PORT_DIM: int = _GCFG["port_embed_dim"]
NODE_TYPES: tuple[str, ...] = tuple(_GCFG["node_types"])
LANE_ROLES: tuple[str, ...] = tuple(_GCFG["lane_roles"])
PERMISSIONS: tuple[str, ...] = tuple(_GCFG["permissions"])
TURN_CLASSES: tuple[str, ...] = tuple(_GCFG["turn_classes"])
PORT_TYPES = ("roi_entry_port", "roi_exit_port")


def _xyz(a, n: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[None]
    if a.shape[-1] == 2:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (1,))], axis=-1)
    if n is not None and a.shape[0] != n:
        raise ValueError(f"期望 {n} 个点，实际 {a.shape[0]}")
    return a


@dataclass
class Node:
    id: int
    pos: np.ndarray                      # (3,)
    node_type: str
    polygon: np.ndarray | None = None    # (POLY_K, 2)，junction_core 必填
    port_embed: np.ndarray | None = None  # (PORT_DIM,)，端口节点用于跨块拼接
    score: float = 1.0

    def __post_init__(self):
        self.pos = _xyz(self.pos, 1)[0]
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"未知节点类型 {self.node_type}")
        if self.polygon is not None:
            self.polygon = np.asarray(self.polygon, dtype=np.float64)[:, :2]

    @property
    def is_port(self) -> bool:
        return self.node_type in PORT_TYPES


@dataclass
class Lane:
    id: int
    points: np.ndarray                   # (K_LANE, 3)，按行驶方向有序、等弧长
    width: float
    role: str = "approach"
    permission: str = "general"
    allow_change_left: bool = True
    allow_change_right: bool = True
    is_stop_line_end: bool = False
    score: float = 1.0

    def __post_init__(self):
        self.points = _xyz(self.points, K_LANE)
        if self.role not in LANE_ROLES:
            raise ValueError(f"未知车道角色 {self.role}")
        if self.permission not in PERMISSIONS:
            raise ValueError(f"未知车道权限 {self.permission}")
        if not self.width > 0:
            raise ValueError("车道宽度必须为正")


@dataclass
class Edge:
    """E_adj 编组解码出的一个 SUMO <edge>：lanes 由最右侧 lane_0 向最左侧排列。"""
    id: int
    lanes: list[int]
    from_node: int
    to_node: int


@dataclass
class Connection:
    from_lane: int
    to_lane: int
    points: np.ndarray                   # (K_CONN, 3)，首尾锁定 l_i 终点与 l_j 起点

    def __post_init__(self):
        self.points = _xyz(self.points, K_CONN)


@dataclass
class Crosswalk:
    id: int
    endpoints: np.ndarray                # (2, 3) 斑马线中心轴起止端点
    width: float
    node: int                            # E_cross_node 解码后的归属节点
    edges: list[int]                     # E_cross 解码后的跨越 edge 列表
    score: float = 1.0

    def __post_init__(self):
        self.endpoints = _xyz(self.endpoints, 2)


@dataclass
class LaneGraph:
    nodes: list[Node]
    lanes: list[Lane]
    edges: list[Edge]
    connections: list[Connection]
    crosswalks: list[Crosswalk] = field(default_factory=list)
    origin: tuple[float, float] = (0.0, 0.0)   # 全局 UTM 偏移 (X_0, Y_0)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 索引
    def node(self, nid: int) -> Node:
        return self._node_map()[nid]

    def lane(self, lid: int) -> Lane:
        return self._lane_map()[lid]

    def edge(self, eid: int) -> Edge:
        return {e.id: e for e in self.edges}[eid]

    def _node_map(self) -> dict[int, Node]:
        return {n.id: n for n in self.nodes}

    def _lane_map(self) -> dict[int, Lane]:
        return {l.id: l for l in self.lanes}

    def edge_of_lane(self) -> dict[int, Edge]:
        return {lid: e for e in self.edges for lid in e.lanes}

    def lane_index(self) -> dict[int, int]:
        """车道在其 edge 内的序号（0 = 最右侧）。"""
        return {lid: k for e in self.edges for k, lid in enumerate(e.lanes)}

    def lane_from(self, lid: int) -> int:
        return self.edge_of_lane()[lid].from_node

    def lane_to(self, lid: int) -> int:
        return self.edge_of_lane()[lid].to_node

    def succ_pairs(self) -> set[tuple[int, int]]:
        return {(c.from_lane, c.to_lane) for c in self.connections}

    # ------------------------------------------------------------------ 校验
    def validate(self) -> list[str]:
        """结构公理检查（返回问题列表，空列表即合法）。"""
        errs: list[str] = []
        nm, lm = self._node_map(), self._lane_map()
        if len(nm) != len(self.nodes):
            errs.append("节点 id 重复")
        if len(lm) != len(self.lanes):
            errs.append("车道 id 重复")
        seen: dict[int, int] = {}
        for e in self.edges:
            if e.from_node not in nm or e.to_node not in nm:
                errs.append(f"edge {e.id} 引用不存在的节点")
            if e.from_node == e.to_node:
                errs.append(f"edge {e.id} 自环")
            if not e.lanes:
                errs.append(f"edge {e.id} 无车道")
            for lid in e.lanes:
                if lid not in lm:
                    errs.append(f"edge {e.id} 引用不存在的车道 {lid}")
                if lid in seen:
                    errs.append(f"车道 {lid} 同时属于 edge {seen[lid]} 与 {e.id}")
                seen[lid] = e.id
        for lid in lm:
            if lid not in seen:
                errs.append(f"车道 {lid} 未编入任何 edge")
        e_of = self.edge_of_lane()
        for c in self.connections:
            if c.from_lane not in e_of or c.to_lane not in e_of:
                errs.append(f"连接 {c.from_lane}->{c.to_lane} 引用不存在的车道")
                continue
            if e_of[c.from_lane].to_node != e_of[c.to_lane].from_node:
                errs.append(f"连接 {c.from_lane}->{c.to_lane} 违反节点衔接约束")
            a = lm[c.from_lane].points[-1, :2]
            b = lm[c.to_lane].points[0, :2]
            if np.linalg.norm(c.points[0, :2] - a) > 1e-6 or np.linalg.norm(c.points[-1, :2] - b) > 1e-6:
                errs.append(f"连接 {c.from_lane}->{c.to_lane} 首尾点未锁定车道端点")
        if len(self.succ_pairs()) != len(self.connections):
            errs.append("存在重复连接")
        ids = {e.id for e in self.edges}
        for cw in self.crosswalks:
            if cw.node not in nm:
                errs.append(f"人行横道 {cw.id} 归属节点不存在")
            elif nm[cw.node].node_type not in ("junction_core", "midblock_node"):
                errs.append(f"人行横道 {cw.id} 只能绑定 junction_core / midblock_node")
            for eid in cw.edges:
                if eid not in ids:
                    errs.append(f"人行横道 {cw.id} 引用不存在的 edge {eid}")
                else:
                    e = self.edge(eid)
                    if cw.node not in (e.from_node, e.to_node):
                        errs.append(f"人行横道 {cw.id} 跨越的 edge {eid} 不与其节点相连")
            if not cw.edges:
                errs.append(f"人行横道 {cw.id} 未跨越任何 edge")
        for n in self.nodes:
            if n.node_type == "junction_core" and (n.polygon is None or len(n.polygon) != POLY_K):
                errs.append(f"junction_core {n.id} 缺少 {POLY_K} 点轮廓")
        if len(self.nodes) > P_MAX or len(self.lanes) > N_MAX or len(self.crosswalks) > M_MAX \
                or len(self.connections) > CONN_MAX:
            errs.append("实体数量超过单窗上限")
        return errs

    # ------------------------------------------------------------------ 张量化
    def to_tensors(self) -> "GraphTensors":
        """离散图 → 定长 one-hot 张量（训练真值 / 解码器往返单测）。"""
        P, N, M = P_MAX, N_MAX, M_MAX
        nidx = {n.id: k for k, n in enumerate(self.nodes)}
        lidx = {l.id: k for k, l in enumerate(self.lanes)}
        t = GraphTensors.empty()
        for k, n in enumerate(self.nodes):
            t.node_pos[k] = n.pos
            t.node_type[k, NODE_TYPES.index(n.node_type)] = 1.0
            if n.polygon is not None:
                t.node_poly[k] = n.polygon
            if n.port_embed is not None:
                t.port_embed[k] = n.port_embed
            t.node_score[k] = 1.0
        for k, l in enumerate(self.lanes):
            t.lane_pts[k] = l.points
            t.lane_width[k] = l.width
            t.lane_role[k, LANE_ROLES.index(l.role)] = 1.0
            t.lane_perm[k, PERMISSIONS.index(l.permission)] = 1.0
            t.lane_lc[k] = [float(l.allow_change_left), float(l.allow_change_right)]
            t.lane_stop[k] = float(l.is_stop_line_end)
            t.lane_score[k] = 1.0
        for e in self.edges:
            for a, b in zip(e.lanes[:-1], e.lanes[1:]):
                t.E_adj[lidx[a], lidx[b]] = 1.0      # b 紧邻 a 的左侧
            for lid in e.lanes:
                t.E_from[lidx[lid], nidx[e.from_node]] = 1.0
                t.E_to[lidx[lid], nidx[e.to_node]] = 1.0
        for c in self.connections:
            i, j = lidx[c.from_lane], lidx[c.to_lane]
            t.E_succ[i, j] = 1.0
            t.conn_pts[(i, j)] = c.points.copy()
        e_of = {e.id: e for e in self.edges}
        for k, cw in enumerate(self.crosswalks):
            t.cw_endpoints[k] = cw.endpoints
            t.cw_width[k] = cw.width
            t.cw_score[k] = 1.0
            t.E_cross_node[k, nidx[cw.node]] = 1.0
            for eid in cw.edges:
                for lid in e_of[eid].lanes:
                    t.E_cross[k, lidx[lid]] = 1.0
        t.origin = tuple(self.origin)
        return t

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        def arr(a):
            return None if a is None else np.asarray(a).round(6).tolist()
        return {
            "origin": list(self.origin),
            "meta": self.meta,
            "nodes": [{"id": n.id, "pos": arr(n.pos), "type": n.node_type, "polygon": arr(n.polygon),
                       "port_embed": arr(n.port_embed), "score": n.score} for n in self.nodes],
            "lanes": [{"id": l.id, "points": arr(l.points), "width": l.width, "role": l.role,
                       "permission": l.permission, "allow_change_left": l.allow_change_left,
                       "allow_change_right": l.allow_change_right, "is_stop_line_end": l.is_stop_line_end,
                       "score": l.score} for l in self.lanes],
            "edges": [{"id": e.id, "lanes": e.lanes, "from": e.from_node, "to": e.to_node} for e in self.edges],
            "connections": [{"from": c.from_lane, "to": c.to_lane, "points": arr(c.points)}
                            for c in self.connections],
            "crosswalks": [{"id": c.id, "endpoints": arr(c.endpoints), "width": c.width, "node": c.node,
                            "edges": c.edges, "score": c.score} for c in self.crosswalks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LaneGraph":
        def opt(a):
            return None if a is None else np.asarray(a, dtype=np.float64)
        return cls(
            nodes=[Node(n["id"], opt(n["pos"]), n["type"], opt(n.get("polygon")), opt(n.get("port_embed")),
                        n.get("score", 1.0)) for n in d["nodes"]],
            lanes=[Lane(l["id"], opt(l["points"]), l["width"], l["role"], l["permission"],
                        l["allow_change_left"], l["allow_change_right"], l["is_stop_line_end"],
                        l.get("score", 1.0)) for l in d["lanes"]],
            edges=[Edge(e["id"], list(e["lanes"]), e["from"], e["to"]) for e in d["edges"]],
            connections=[Connection(c["from"], c["to"], opt(c["points"])) for c in d["connections"]],
            crosswalks=[Crosswalk(c["id"], opt(c["endpoints"]), c["width"], c["node"], list(c["edges"]),
                                  c.get("score", 1.0)) for c in d.get("crosswalks", [])],
            origin=tuple(d.get("origin", (0.0, 0.0))),
            meta=d.get("meta", {}),
        )

    def save_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load_json(cls, path) -> "LaneGraph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class GraphTensors:
    """定长张量形式的 G（模型输出概率 / 训练真值）。

    类别字段统一保存为概率/one-hot 向量；``*_score`` 为存在置信度。
    ``conn_pts`` 以车道槽位对 (i, j) 为键保存成对连接轨迹（推理期由成对轨迹头对
    ``E_succ`` 激活对按需填充；动态上限 CONN_MAX）。
    """
    node_pos: np.ndarray
    node_type: np.ndarray
    node_poly: np.ndarray
    port_embed: np.ndarray
    node_score: np.ndarray
    lane_pts: np.ndarray
    lane_width: np.ndarray
    lane_role: np.ndarray
    lane_perm: np.ndarray
    lane_lc: np.ndarray
    lane_stop: np.ndarray
    lane_score: np.ndarray
    E_adj: np.ndarray
    E_from: np.ndarray
    E_to: np.ndarray
    E_succ: np.ndarray
    cw_endpoints: np.ndarray
    cw_width: np.ndarray
    cw_score: np.ndarray
    E_cross: np.ndarray
    E_cross_node: np.ndarray
    conn_pts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    origin: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def empty(cls) -> "GraphTensors":
        P, N, M = P_MAX, N_MAX, M_MAX
        z = np.zeros
        return cls(
            node_pos=z((P, 3)), node_type=z((P, len(NODE_TYPES))), node_poly=z((P, POLY_K, 2)),
            port_embed=z((P, PORT_DIM)), node_score=z(P),
            lane_pts=z((N, K_LANE, 3)), lane_width=z(N), lane_role=z((N, len(LANE_ROLES))),
            lane_perm=z((N, len(PERMISSIONS))), lane_lc=z((N, 2)), lane_stop=z(N), lane_score=z(N),
            E_adj=z((N, N)), E_from=z((N, P)), E_to=z((N, P)), E_succ=z((N, N)),
            cw_endpoints=z((M, 2, 3)), cw_width=z(M), cw_score=z(M), E_cross=z((M, N)), E_cross_node=z((M, P)),
        )
