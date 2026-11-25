"""人工真值集：模板生成、读取与评分。

真值文件：data/评测/真值/<区>_<fid>/truth.json，标注对照图位于 输出/评测/真值/<区>_<fid>/。
只有 "status" 为 verified（人工核对）或 provisional（Claude 目检的临时真值）的文件参与评分；模板由当前算法结果预填（status=draft），标注者改对后改成 verified。

字段：
  arms[]   bearing（路臂外向方位角，度）、in_stop_lanes（停止线处进口机动车道数）、in_far_lanes（进口远端约 80 m 处车道数）、
           out_lanes（出口车道数）、minor（无标线小支路）
  slips[]  from_bearing、to_bearing、lanes（渠化右转车道数）
  add_drop_labeled  true 时才评 add_drop[]：{arm_bearing, role(approach/exit), dist_from_mouth(米), kind(add/drop)}

python -m lanegen.truth init      生成 / 补齐模板（不覆盖已有文件）
python -m lanegen.truth score <标签>  对评测结果 metrics.csv 里带 gt_ 前缀的列汇总
"""
import glob
import json
import os
import sys

import numpy as np

TRUTH_DIR = "data/评测/真值"


def load_truth(d=TRUTH_DIR, accept=("verified", "provisional")):
    """verified = 人工核对；provisional = Claude 目检的临时真值（只标了看得清的项），人工核对前仅作参考。"""
    out = {}
    for p in glob.glob(os.path.join(d, "*", "truth.json")):
        t = json.load(open(p, encoding="utf-8"))
        if t.get("status") in accept:
            out[t["key"]] = t
    return out


def _angdiff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def predicted(res):
    """把算法结果整理成与真值同口径的结构。"""
    from .frenet import _ax
    jm = res["model"]
    arms = []
    for a in jm["arms"]:
        cws = [c for c in jm["carriageways"] if c["arm"] == a["id"]]
        ins = [c for c in cws if c["role"] == "approach" and c.get("sections")]
        outs = [c for c in cws if c["role"] == "exit" and c.get("sections")]
        far = []
        for c in ins:
            ax = _ax(c)
            s80 = max(0.0, ax.length - 80.0)
            sec = next((s for s in c["sections"] if s["s_start"] <= s80 <= s["s_end"]), c["sections"][0])
            far.append(sec["n_lanes"])
        arms.append(dict(id=a["id"], bearing=float(a["bearing"]), minor=bool(a.get("minor_branch")),
                         in_stop_lanes=sum(c["sections"][-1]["n_lanes"] for c in ins),
                         in_far_lanes=sum(far), out_lanes=sum(c["sections"][0]["n_lanes"] for c in outs)))
    bear = {a["id"]: float(a["bearing"]) for a in jm["arms"]}
    slips = [dict(from_bearing=bear.get(s["from_arm"]), to_bearing=bear.get(s["to_arm"]), lanes=len(s.get("lanes", [])))
             for s in jm["slips"] if s.get("lanes")]
    events = []
    for c in jm["carriageways"]:
        if not c.get("sections"):
            continue
        L = _ax(c).length
        for sec in c["sections"][1:]:
            tr = str(sec.get("transition") or "")
            kind = "add" if "add" in tr else ("drop" if "drop" in tr else None)
            if kind:
                dist = L - sec["s_start"] if c["role"] == "approach" else sec["s_start"]
                events.append(dict(arm_bearing=bear[c["arm"]], role=c["role"], dist_from_mouth=round(float(dist), 1), kind=kind))
    return dict(arms=arms, slips=slips, add_drop=events)


def _match(pred, truth, key_fn, tol):
    """贪心一对一匹配（按角度差升序）。返回 [(p, t)]。"""
    cand = sorted((key_fn(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth))
    used_p, used_t, out = set(), set(), []
    for d, i, j in cand:
        if d > tol or i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j)
        out.append((pred[i], truth[j]))
    return out


def score(res, t):
    p = predicted(res)
    m = {}
    pa = [a for a in p["arms"]]
    ta = t.get("arms", [])
    pairs = _match(pa, ta, lambda x, y: _angdiff(x["bearing"], y["bearing"]), 25.0)
    m["arm_recall"] = len(pairs) / len(ta) if ta else np.nan
    m["arm_precision"] = len(pairs) / len(pa) if pa else np.nan
    for k, name in (("in_stop_lanes", "stop_lanes_acc"), ("out_lanes", "exit_lanes_acc"), ("in_far_lanes", "far_lanes_acc")):
        vals = [x[k] == y[k] for x, y in pairs if y.get(k) is not None]
        m[name] = float(np.mean(vals)) if vals else np.nan
    ts = t.get("slips", [])
    ps = [s for s in p["slips"] if s["from_bearing"] is not None and s["to_bearing"] is not None]
    sp = _match(ps, ts, lambda x, y: max(_angdiff(x["from_bearing"], y["from_bearing"]), _angdiff(x["to_bearing"], y["to_bearing"])), 30.0)
    m["slip_recall"] = len(sp) / len(ts) if ts else np.nan
    m["slip_precision"] = len(sp) / len(ps) if ps else (np.nan if not ts else 0.0)
    if t.get("add_drop_labeled"):
        te, pe = t.get("add_drop", []), p["add_drop"]
        key = lambda x, y: (_angdiff(x["arm_bearing"], y["arm_bearing"]) / 25.0 * 15.0 if x["role"] == y["role"] and x["kind"] == y["kind"]
                            else 1e9) + abs(x["dist_from_mouth"] - y["dist_from_mouth"])
        ep = _match(pe, te, key, 15.0)
        m["adddrop_recall"] = len(ep) / len(te) if te else np.nan
        m["adddrop_precision"] = len(ep) / len(pe) if pe else (np.nan if not te else 0.0)
    return m


def init_templates(n=24, extra=(6789000006, 9564000002, 8908000002, 9747000001, 15170000001, 4533000001,
                                7718000001, 3129000002)):
    """从留出集分层挑 n 个路口（按车道数均匀取）+ 已知难例，生成预填模板与对照图；已有文件不覆盖。"""
    from . import pipeline, run as runner
    cases = json.load(open("data/评测/留出集.json", encoding="utf-8"))
    met = {}
    mp = "输出/评测/v16/metrics.csv"
    if os.path.exists(mp):
        import pandas as pd
        d = pd.read_csv(mp)
        met = dict(zip(d.fid, d.n_lanes))
    cs = sorted(cases, key=lambda c: met.get(c["fid"], 0))
    pick = [cs[int(round(i))] for i in np.linspace(0, len(cs) - 1, n)]
    pick += [c for c in cases if c["fid"] in extra and c not in pick]
    os.makedirs(TRUTH_DIR, exist_ok=True)
    for c in pick:
        key = f"{c['district']}_{c['fid']}"
        d = os.path.join(TRUTH_DIR, key)
        if os.path.exists(os.path.join(d, "truth.json")):
            continue
        os.makedirs(d, exist_ok=True)
        res, _ = runner.main(key, c["lon"], c["lat"], "输出/评测/真值")
        p = predicted(res)
        t = dict(key=key, lon=c["lon"], lat=c["lat"], status="draft",
                 arms=[dict(bearing=round(a["bearing"]), in_stop_lanes=a["in_stop_lanes"], in_far_lanes=a["in_far_lanes"],
                            out_lanes=a["out_lanes"], minor=a["minor"]) for a in p["arms"]],
                 slips=[dict(from_bearing=round(s["from_bearing"]), to_bearing=round(s["to_bearing"]), lanes=s["lanes"])
                        for s in p["slips"]],
                 add_drop_labeled=False, add_drop=p["add_drop"],
                 notes=f"对照 输出/评测/真值/{key}/ 下的叠加图逐项核对预填值，补齐漏掉的臂和渠化右转，"
                       "多出来的删掉；全部核对后把 status 改为 verified。add_drop 也核对过才把 add_drop_labeled 改为 true。")
        json.dump(t, open(os.path.join(d, "truth.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return pick


def summarize(tag):
    import pandas as pd
    d = pd.read_csv(f"输出/评测/{tag}/metrics.csv")
    cols = [c for c in d.columns if c.startswith("gt_")]
    if not cols:
        print("没有已核对（verified）的真值路口")
        return
    print(d[cols].describe().round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "init":
        print([f"{c['district']}_{c['fid']}" for c in init_templates()])
    elif sys.argv[1] == "score":
        summarize(sys.argv[2])
