"""按进口车道、实际出口边及出口车道联合分配转向。"""
from collections import defaultdict

from ortools.sat.python import cp_model

from .sumo_check import _IDEAL, approach_options, heuristic_connections, junction_demands

RANK = {"R": 0, "r": 1, "S": 2, "l": 3, "L": 4, "T": 5}
_ARROW_W = 20


def _solve_approach(mv, options):
    """options 按实际出口从右到左排列，返回 (进口车道, 箭头语义, 出口边, 出口车道)。"""
    if not options or not mv:
        return []
    md = cp_model.CpModel()
    given_any = set().union(*(codes for codes, observed in mv if observed))
    moves = [ch for ch in RANK if ch in given_any or ch in ("R", "S", "L")]
    moves = [ch for ch in moves if any(ch in o.movements for o in options)]
    if not moves:
        return []
    n = len(mv)
    positions = sum(o.lanes for o in options)
    serving = defaultdict(list)
    feeds = defaultdict(list)
    choices = {}
    cost, tie = [], []
    previous_left = None
    for k, (given, observed) in enumerate(mv):
        # 每条进口的目标区间按实际方位、平行出口的横向位置、出口车道序号排序。
        right = md.NewIntVar(0, positions - 1, f"right{k}")
        left = md.NewIntVar(0, positions - 1, f"left{k}")
        md.Add(right <= left)
        if previous_left is not None:
            md.Add(previous_left <= right)
        previous_left = left
        offset = 0
        by_move = defaultdict(list)
        for o in options:
            per_edge = []
            for ch in moves:
                if ch not in o.movements:
                    continue
                for j in range(o.lanes):
                    v = md.NewBoolVar(f"link{k}_{ch}_{o.edge}_{j}")
                    choices[k, ch, o.edge, j] = v
                    by_move[ch].append(v)
                    per_edge.append(v)
                    feeds[o.edge, j].append(v)
                    md.Add(right <= offset + j).OnlyEnforceIf(v)
                    md.Add(left >= offset + j).OnlyEnforceIf(v)
                    side = 0
                    if ch in ("R", "r"):
                        side = j
                    elif ch in ("L", "l", "T"):
                        side = o.lanes - 1 - j
                    cost.append((1 + 2 * side) * v)
                    ideal = _IDEAL.get(ch, 180.0)
                    diff = abs((o.angle - ideal + 180.0) % 360.0 - 180.0)
                    tie.append(round(diff) * v)
            if per_edge:
                md.Add(sum(per_edge) <= 1)
            offset += o.lanes
        for ch in moves:
            used = md.NewBoolVar(f"move{k}_{ch}")
            md.AddMaxEquality(used, by_move[ch])
            serving[ch].append(used)
            # 一次连接包含在原有转向代价中，额外分支另计。
            cost.append(-used)
            if observed:
                cost.append(_ARROW_W * (1 - used) if ch in given else _ARROW_W * used)
            else:
                natural = ch == "S" or (ch in ("R", "r") and k == 0) or (ch in ("L", "l") and k == n - 1)
                cost.append((1 if natural else 6) * used)
        if not observed or given.intersection(moves):
            md.Add(sum(serving[ch][-1] for ch in moves) >= 1)

    if not any(observed for _, observed in mv):
        # 重叠扇区中的同一个出口不强行制造两种转向需求。
        primary = set()
        for o in options:
            compatible = o.movements.intersection(moves)
            if compatible:
                primary.add(min(compatible, key=lambda ch: (abs((o.angle - _IDEAL[ch] + 180) % 360 - 180), RANK[ch])))
        for ch in sorted(primary):
            md.Add(sum(serving[ch]) >= 1)
    for (edge, j), variables in feeds.items():
        fed = md.NewBoolVar(f"fed_{edge}_{j}")
        md.AddMaxEquality(fed, variables)
        cost.append(-3 * fed)
    # 角度只打破同等拓扑代价的平局。项目既有箭头、左右次序和覆盖约束优先。
    md.Minimize(sum(cost) * (180 * n * len(options) + 1) + sum(tie))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.num_workers = 1
    status = solver.Solve(md)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [link for link, v in choices.items() if solver.Value(v)]


def cpsat_connections(res, netd, diagnostics=None):
    cons = []
    fallback = None
    for cw, eid, mv, heading in junction_demands(res, netd):
        opts = approach_options(res["model"], netd, cw, mv, heading)
        sol = _solve_approach([(set(chs), observed) for _, chs, observed in mv], opts)
        if sol is None:
            if diagnostics is not None:
                diagnostics["fallback_approaches"] += 1
            if fallback is None:
                fallback = heuristic_connections(res, netd)
            cons.extend(c for c in fallback if c[0] == eid)
        else:
            cons.extend((eid, k, edge, j) for k, ch, edge, j in sol)
    return sorted(set(cons))
