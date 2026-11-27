#!/usr/bin/env bash
# 实时查看训练进度（docs/DESIGN.md §6.2-4）：有 tmux 时进入对应会话（Ctrl-b d 退出不影响训练），
# 否则 tail -f 训练日志并附带最新指标汇总。
#   ./watch.sh [run_id]
set -uo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-$(cat "$BUNDLE/runs/LATEST" 2>/dev/null || true)}"
[[ -n "$RUN_ID" ]] || { echo "未指定 run_id，且 runs/LATEST 不存在"; exit 2; }
RUN_DIR="$BUNDLE/runs/$RUN_ID"
summary() {
  python3 - "$RUN_DIR/metrics.jsonl" <<'PY' 2>/dev/null || true
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tr = [r for r in rows if "loss" in r]
va = [r for r in rows if any(k.startswith("val/") for k in r)]
if tr:
    r = tr[-1]
    eta = r.get("eta_s", 0)
    print(f"最新：step {r['step']} epoch {r['epoch']} loss {r['loss']:.3f} lr {r['lr']:.2e} "
          f"{r['samples_per_s']:.2f} 样本/s ETA {int(eta)//3600}h{int(eta)%3600//60:02d}m 显存 {r.get('mem_gib',0):.1f}GiB")
if va:
    v = va[-1]
    print("最近验证：" + " ".join(f"{k[4:]}={x:.4f}" for k, x in v.items() if k.startswith("val/")))
PY
}
if command -v tmux >/dev/null && tmux has-session -t "b2s_${RUN_ID}" 2>/dev/null; then
  summary
  echo "进入 tmux 会话 b2s_${RUN_ID}（Ctrl-b d 退出，训练继续）"
  sleep 1
  exec tmux attach -t "b2s_${RUN_ID}"
fi
summary
echo "（无 tmux 会话）跟踪 $RUN_DIR/train.log，Ctrl-C 退出不影响训练"
exec tail -n 30 -f "$RUN_DIR/train.log"
