#!/usr/bin/env bash
# 打包训练成果供拷回本机（docs/DESIGN.md §6.2-5）：
#   ./pack_results.sh <run_id>  → results_<run_id>.tar.gz（含 MANIFEST、配置快照、种子、切分版本、calibration、
#   best/last checkpoint、汇总指标、metrics.jsonl、训练日志、验证样例叠图、CUDA 参考前向输出与 sha256 清单）
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-$(cat "$BUNDLE/runs/LATEST" 2>/dev/null || true)}"
[[ -n "$RUN_ID" ]] || { echo "用法：./pack_results.sh <run_id>"; exit 2; }
RUN_DIR="$BUNDLE/runs/$RUN_ID"
[[ -d "$RUN_DIR" ]] || { echo "不存在：$RUN_DIR"; exit 2; }
CK="$RUN_DIR/checkpoints/best.pt"; [[ -f "$CK" ]] || CK="$RUN_DIR/checkpoints/last.pt"
STAGE_DIR="$(mktemp -d)"
OUT="$STAGE_DIR/results_$RUN_ID"
mkdir -p "$OUT/run"
cp "$BUNDLE/MANIFEST.json" "$OUT/MANIFEST.json"
for f in config_snapshot.json calibration.json metrics.jsonl train.log summary.json console.log run_train.log; do
  [[ -f "$RUN_DIR/$f" ]] && cp "$RUN_DIR/$f" "$OUT/run/"
done
[[ -d "$RUN_DIR/env" ]] && cp -r "$RUN_DIR/env" "$OUT/run/env"
[[ -d "$RUN_DIR/figures" ]] && cp -r "$RUN_DIR/figures" "$OUT/run/figures"
mkdir -p "$OUT/run/checkpoints"
for f in best.pt last.pt; do [[ -f "$RUN_DIR/checkpoints/$f" ]] && cp "$RUN_DIR/checkpoints/$f" "$OUT/run/checkpoints/"; done
# CUDA 参考前向输出（本机导入时做跨设备一致性比对）
if [[ -f "$RUN_DIR/env/CONDA_BIN" ]]; then
  eval "$("$(cat "$RUN_DIR/env/CONDA_BIN")" shell.bash hook)"
  conda activate "$(cat "$RUN_DIR/env/ENV_NAME")"
  ( cd "$BUNDLE/code" && PYTHONPATH="$BUNDLE/code" PYTHONDONTWRITEBYTECODE=1 B2S_SYNTH_DIR="$BUNDLE/data/synth_v1" \
    B2S_CACHE_DIR="$BUNDLE/cache" B2S_DEVICE=cuda python -B -m scripts.export_reference --ckpt "$CK" \
    --out "$OUT/run/reference_cuda.npz" ) || echo "警告：参考前向输出生成失败（导入时将仅做本机 CPU / MPS 一致性检查）"
fi
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
tar -C "$STAGE_DIR" -czf "$BUNDLE/results_$RUN_ID.tar.gz" "results_$RUN_ID"
rm -rf "$STAGE_DIR"
echo "已生成：$BUNDLE/results_$RUN_ID.tar.gz（$(du -h "$BUNDLE/results_$RUN_ID.tar.gz" | cut -f1)）"
echo "拷回本机后执行：python -B -m scripts.import_results results_$RUN_ID.tar.gz"
