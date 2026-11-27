#!/usr/bin/env bash
# 一键训练入口（docs/DESIGN.md §6.2-3）：自检 → 建 conda 环境并安装依赖 → 包内自测 → 校准 → 启动训练（tmux）
# 幂等，可重复执行。训练包绝不安装 / 升级 / 改动系统级驱动与 CUDA，绝不结束或修改用户的其他进程与服务，
# 不修改 conda base 及其他已有环境。
#
# 用法：
#   ./run_train.sh [--env-name baidu2sumo] [--recreate-env] [--run-id ID] [--resume ID] [--skip-tests]
#                  [--skip-calibration] [--no-tmux] [--conda-channel URL] [--pip-index-url URL]
#                  [--torch-index-url URL] [--min-free-vram-gib N] [-- <透传给 scripts.train 的参数>]
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE"
ENV_NAME="baidu2sumo"
RECREATE=0; RUN_ID=""; RESUME=""; SKIP_TESTS=0; SKIP_CAL=0; NO_TMUX=0
CONDA_CHANNEL="${B2S_CONDA_CHANNEL:-}"
PIP_INDEX="${B2S_PIP_INDEX_URL:-}"
TORCH_INDEX="${B2S_TORCH_INDEX_URL:-$(python3 -c 'import json;print(json.load(open("MANIFEST.json"))["torch"]["index_url"])' 2>/dev/null || echo https://download.pytorch.org/whl/cu128)}"
MIN_FREE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2;;
    --recreate-env) RECREATE=1; shift;;
    --run-id) RUN_ID="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --skip-tests) SKIP_TESTS=1; shift;;
    --skip-calibration) SKIP_CAL=1; shift;;
    --no-tmux) NO_TMUX=1; shift;;
    --conda-channel) CONDA_CHANNEL="$2"; shift 2;;
    --pip-index-url) PIP_INDEX="$2"; shift 2;;
    --torch-index-url) TORCH_INDEX="$2"; shift 2;;
    --min-free-vram-gib) MIN_FREE="$2"; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "未知参数：$1"; exit 2;;
  esac
done

STAGE="$(python3 -c 'import json;print(json.load(open("MANIFEST.json"))["stage"])' 2>/dev/null || echo layer1)"
if [[ -n "$RESUME" ]]; then RUN_ID="$RESUME"; fi
if [[ -z "$RUN_ID" ]]; then RUN_ID="${STAGE}_$(date +%Y%m%d_%H%M%S)"; fi
RUN_DIR="$BUNDLE/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run_train.log"
say() { echo "[run_train $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
fail() { say "停止：$*"; exit 3; }

# ------------------------------------------------------------------ 1. 自检（只读）
say "========== 1/6 自检 =========="
say "OS: $(uname -s) $(uname -r)  架构: $(uname -m)"
[[ "$(uname -s)" == "Linux" ]] || fail "训练包面向 Linux 服务器"
command -v nvidia-smi >/dev/null || fail "未找到 nvidia-smi（NVIDIA 驱动未就绪；训练包不安装驱动）"
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits | head -1)"
say "GPU: $GPU_LINE"
DRV_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
say "驱动支持的最高 CUDA: ${DRV_CUDA:-未知}"
if command -v nvcc >/dev/null; then say "系统 nvcc（只读）: $(nvcc --version | tail -1)"; else say "系统 nvcc: 未找到（不影响：PyTorch 自带 CUDA 运行库）"; fi
TORCH_CUDA="$(python3 -c 'import json;print(json.load(open("MANIFEST.json"))["torch"]["cuda"])' 2>/dev/null || echo 12.8)"
if [[ -n "$DRV_CUDA" ]] && python3 - "$DRV_CUDA" "$TORCH_CUDA" <<'PY'
import sys
d=[int(x) for x in sys.argv[1].split(".")[:2]]; t=[int(x) for x in sys.argv[2].split(".")[:2]]
sys.exit(0 if d>=t else 1)
PY
then say "PyTorch CUDA 构建 $TORCH_CUDA ≤ 驱动支持 $DRV_CUDA：兼容"
else fail "PyTorch CUDA 构建 $TORCH_CUDA 高于驱动支持的 ${DRV_CUDA}（请更换 --torch-index-url 为兼容构建）"; fi
FREE_MIB="$(echo "$GPU_LINE" | awk -F', ' '{print $3}')"
if [[ -z "$MIN_FREE" ]]; then
  MIN_FREE="$(python3 - <<'PY' 2>/dev/null || echo 18
import re
t=open("code/configs/train.yaml",encoding="utf-8").read()
m=re.search(r"min_free_vram_gib:\s*([0-9.]+)",t); print(m.group(1) if m else 18)
PY
)"
fi
if [[ -f "$BUNDLE/runs/latest_calibration.json" ]]; then
  CAL_MIN="$(python3 -c 'import json;v=json.load(open("runs/latest_calibration.json"))["recommended"].get("min_free_vram_gib");print(v if v else "")' 2>/dev/null || true)"
  [[ -n "$CAL_MIN" ]] && MIN_FREE="$CAL_MIN"
fi
say "GPU 空闲显存: ${FREE_MIB} MiB，门槛 ${MIN_FREE} GiB"
if python3 -c "import sys;sys.exit(0 if float('$FREE_MIB')/1024 >= float('$MIN_FREE') else 1)"; then :; else
  say "GPU 占用进程（只读列出，训练包不会结束或修改它们）："
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG" || true
  fail "空闲显存不足 ${MIN_FREE} GiB，请自行释放（例如停止 ollama 等常驻服务）后重试"
fi
say "CPU 核数: $(nproc)  内存: $(free -g | awk '/Mem:/{print $2" GiB 总 / "$7" GiB 可用"}')  磁盘剩余: $(df -h "$BUNDLE" | awk 'NR==2{print $4}')"
HAS_TMUX=0; command -v tmux >/dev/null && HAS_TMUX=1
say "tmux: $([[ $HAS_TMUX == 1 ]] && echo 可用 || echo 不可用（将退化为 nohup）)"
CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda" "/opt/conda/bin/conda" \
           "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" "/root/miniconda3/bin/conda" "/root/anaconda3/bin/conda"; do
    [[ -x "$c" ]] && CONDA_BIN="$c" && break
  done
fi
[[ -n "$CONDA_BIN" ]] || fail "未找到 conda（请确认服务器已安装 conda 并在 PATH 中）"
say "conda: $CONDA_BIN"
say "校验包内文件 sha256 ..."
sha256sum --quiet -c SHA256SUMS || fail "sha256 校验失败（训练包损坏或被修改）"
say "sha256 校验通过"

# ------------------------------------------------------------------ 2. conda 环境
say "========== 2/6 conda 环境 '$ENV_NAME' =========="
eval "$("$CONDA_BIN" shell.bash hook)"
REQ_SHA="$(cat env/environment.yml env/requirements-torch.txt env/requirements.txt | sha256sum | cut -c1-16)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ $RECREATE == 1 ]]; then say "--recreate-env：删除并重建 $ENV_NAME"; conda env remove -y -n "$ENV_NAME"; fi
fi
PIP_ARGS=(); [[ -n "$PIP_INDEX" ]] && PIP_ARGS+=(--index-url "$PIP_INDEX")
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "新建环境 $ENV_NAME（不改动 base 与其他已有环境）"
  if [[ -n "$CONDA_CHANNEL" ]]; then
    conda create -y -n "$ENV_NAME" -c "$CONDA_CHANNEL" --override-channels python=3.12 pip 2>&1 | tee -a "$LOG"
  else
    conda env create -n "$ENV_NAME" -f env/environment.yml 2>&1 | tee -a "$LOG"
  fi
fi
conda activate "$ENV_NAME"
STAMP="$CONDA_PREFIX/.b2s_requirements_sha"
if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$REQ_SHA" ]]; then
  say "环境依赖清单未变化（$REQ_SHA），复用"
else
  say "安装 / 同步依赖：PyTorch（$TORCH_INDEX）"
  python -m pip install --upgrade pip "${PIP_ARGS[@]}" 2>&1 | tail -1 | tee -a "$LOG"
  python -m pip install -r env/requirements-torch.txt --index-url "$TORCH_INDEX" 2>&1 | tail -2 | tee -a "$LOG"
  say "安装 / 同步其余依赖"
  python -m pip install -r env/requirements.txt "${PIP_ARGS[@]}" 2>&1 | tail -2 | tee -a "$LOG"
  echo "$REQ_SHA" > "$STAMP"
fi
mkdir -p "$RUN_DIR/env"
echo "$ENV_NAME" > "$RUN_DIR/env/ENV_NAME"
echo "$CONDA_BIN" > "$RUN_DIR/env/CONDA_BIN"
conda list > "$RUN_DIR/env/conda_list.txt"
python -m pip freeze > "$RUN_DIR/env/pip_freeze.txt"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BUNDLE/code" B2S_DEVICE=cuda
export B2S_CACHE_DIR="$BUNDLE/cache" B2S_SYNTH_DIR="$BUNDLE/data/synth_v1" B2S_RUNS_ROOT="$BUNDLE/runs"
export B2S_BUNDLE_MANIFEST="$BUNDLE/MANIFEST.json"
export B2S_SUMO_BIN="$(python -c 'import sumo,os;print(os.path.join(sumo.SUMO_HOME,"bin"))')"
python -B "$BUNDLE/code/scripts/bundle/verify_env.py" --manifest "$BUNDLE/MANIFEST.json" --out "$RUN_DIR/env/verify_env.json" 2>&1 | tee -a "$LOG"

cd "$BUNDLE/code"
# ------------------------------------------------------------------ 3. 包内自测
say "========== 3/6 包内自测 =========="
if [[ $SKIP_TESTS == 1 || -n "$RESUME" ]]; then say "跳过包内自测（--skip-tests / --resume）"
else
  set +e
  python -B -m pytest -q -p no:cacheprovider tests -m "not mps and not gpkg" > "$RUN_DIR/env/pytest.txt" 2>&1
  RC=$?
  set -e
  tail -5 "$RUN_DIR/env/pytest.txt" | tee -a "$LOG"
  [[ $RC == 0 ]] || fail "包内自测未通过（见 $RUN_DIR/env/pytest.txt）"
fi

# ------------------------------------------------------------------ 4. 校准
say "========== 4/6 校准 =========="
if [[ -n "$RESUME" ]]; then say "续训：沿用 $RUN_DIR/calibration.json"
elif [[ $SKIP_CAL == 1 ]]; then say "跳过校准（--skip-calibration）"
else
  python -B -m scripts.calibrate --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/env/calibration_console.txt" | tail -12
  cp "$RUN_DIR/calibration.json" "$BUNDLE/runs/latest_calibration.json"
fi
TRAIN_ARGS=(--stage "$STAGE" --run-id "$RUN_ID" --runs-root "$BUNDLE/runs")
if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume); fi
if [[ -f "$RUN_DIR/calibration.json" ]]; then
  read -r CAL_BS CAL_NW < <(python -c "import json;r=json.load(open('$RUN_DIR/calibration.json'))['recommended'];print(r['batch_size'],r['num_workers'])")
  TRAIN_ARGS+=(--batch-size "$CAL_BS" --num-workers "$CAL_NW")
  say "按校准：batch_size=$CAL_BS num_workers=$CAL_NW（命令行 -- 之后的参数可覆盖）"
fi
TRAIN_ARGS+=("${EXTRA[@]}")

# ------------------------------------------------------------------ 5. 启动训练
say "========== 5/6 启动训练 =========="
CMD="cd '$BUNDLE/code' && source '$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh' 2>/dev/null; conda activate '$ENV_NAME'; \
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$BUNDLE/code' B2S_DEVICE=cuda B2S_CACHE_DIR='$BUNDLE/cache' \
B2S_SYNTH_DIR='$BUNDLE/data/synth_v1' B2S_RUNS_ROOT='$BUNDLE/runs' B2S_BUNDLE_MANIFEST='$BUNDLE/MANIFEST.json' B2S_SUMO_BIN='$B2S_SUMO_BIN'; \
python -B -m scripts.train ${TRAIN_ARGS[*]} 2>&1 | tee -a '$RUN_DIR/console.log'"
SESSION="b2s_${RUN_ID}"
if [[ $HAS_TMUX == 1 && $NO_TMUX == 0 ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then fail "tmux 会话 $SESSION 已存在（训练可能仍在运行：./watch.sh $RUN_ID）"; fi
  tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\"; echo '训练进程已退出，按回车关闭'; read"
  say "已在 tmux 会话 $SESSION 中启动（SSH 断开不影响训练）"
else
  nohup bash -lc "$CMD" > "$RUN_DIR/nohup.out" 2>&1 &
  echo $! > "$RUN_DIR/train.pid"
  say "已用 nohup 后台启动（PID $(cat "$RUN_DIR/train.pid")）"
fi
echo "$RUN_ID" > "$BUNDLE/runs/LATEST"

# ------------------------------------------------------------------ 6. 提示
say "========== 6/6 完成 =========="
say "查看进度：  ./watch.sh $RUN_ID        （tmux 中 Ctrl-b d 退出而不影响训练）"
say "断点续训：  ./run_train.sh --resume $RUN_ID"
say "打包成果：  ./pack_results.sh $RUN_ID"
