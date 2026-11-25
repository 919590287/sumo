"""对比两次评测：python -m lanegen.compare 基线v1 自适应v2"""
import json
import sys

import numpy as np
import pandas as pd

KEYS = ["arrow_recall", "arrow_recall80", "stop_lane_eq", "stop_lane_ge", "stop_good", "stop_cover", "crossings", "duplicates", "merged_flag",
        "unknown_dir", "frag_per_lane", "width_bad", "shift_stable", "n_lanes", "core_stops", "sec",
        "ref_out_recall", "ref_in_recall", "n_exit_lanes", "n_add_drop", "n_slip", "n_nonmotor", "core_cross_m", "n_arms", "n_arms_empty", "n_carriageways", "lanes_no_arm", "dir_conflicts", "island_m", "island_frac", "offroad_frac",
        "center_err_p50", "center_err_p90", "center_bad",
        "contract_errors", "orphan_lanes", "cw_arrow_dir_conflict", "narrow_cw_count", "jump_transitions", "dp_fallback_rate",
        "n_slips_model", "n_slip_dead_end", "n_cw_unmarked",
        "sumo_net_warnings", "sumo_mv_cover", "sumo_dead_lanes", "sumo_route_arrive_rate", "sumo_teleports",
        "sumo_collisions", "sumo_uncompiled_lanes", "sumo_unfed_lanes", "on_mark_m", "foreign_intrude_m",
        "sumo_turn_cross", "sumo_mv_unserved", "sumo_exit_fed", "sumo_route_pairs"]
KEYS += ["gt_" + k for k in ("arm_recall", "arm_precision", "stop_lanes_acc", "exit_lanes_acc", "far_lanes_acc",
                              "slip_recall", "slip_precision", "adddrop_precision", "adddrop_recall")]


def load(tag):
    return pd.read_csv(f"输出/评测/{tag}/metrics.csv")


if __name__ == "__main__":
    a, b = load(sys.argv[1]), load(sys.argv[2])
    rows = []
    for k in KEYS:
        if k in a and k in b:
            ra = a[k].astype(float)
            rb = b[k].astype(float)
            rows.append(dict(指标=k, 旧均值=round(ra.mean(), 3), 新均值=round(rb.mean(), 3),
                             旧P10=round(np.nanpercentile(ra, 10), 3), 新P10=round(np.nanpercentile(rb, 10), 3),
                             旧非零路口=int((ra > 0).sum()), 新非零路口=int((rb > 0).sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    print("报错：旧", int(a.get("error", pd.Series(dtype=str)).notna().sum()), "新", int(b.get("error", pd.Series(dtype=str)).notna().sum()))
