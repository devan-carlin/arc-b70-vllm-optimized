#!/usr/bin/env bash
# =============================================================================
# serve.sh — Launch a vLLM OpenAI server on Intel Arc Pro B70 (XPU) with the
# tuned B70 settings.
#
# Usage:
#   bash serve.sh /path/to/model                     # interactive options
#   bash serve.sh /path/to/model --dry-run           # print command only
#
# Env overrides (skip the interactive prompts):
#   VENV_DIR, TP_SIZE, GPU_SET, MAX_LEN, KV_DTYPE, GPU_UTIL, PORT,
#   BLOCK_SIZE, MAX_BATCHED, MAX_SEQS, LM_ONLY, PREFIX_CACHING, SERVED_NAME
#
# The XPU environment block (XPU graph, CCL, device selector) is the part
# that matters most — it is set unconditionally below.
# =============================================================================
set -uo pipefail

MODEL="${1:-}"
[[ -z "$MODEL" ]] && { echo "usage: bash serve.sh /path/to/model [--dry-run]"; exit 1; }
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1 || DRY_RUN=0

VENV="${VENV_DIR:-$HOME/vllm-xpu/.venv}"

ask() { local prompt="$1" def="${2:-}" ans; if [[ -n "$def" ]]; then read -rp "$prompt [$def]: " ans; else read -rp "$prompt: " ans; fi; ans="${ans:-$def}"; printf '%s' "$ans"; }

# --- GPU selection -----------------------------------------------------------
SYCL_OUT="$(sycl-ls 2>/dev/null)"
GPU_IDS=""
for i in 0 1 2 3 4 5 6 7; do
  if grep -q "level_zero:$i" <<<"$SYCL_OUT"; then GPU_IDS+="$i "; fi
done
GPU_IDS="${GPU_IDS:-0 1 2 3}"

if [[ -z "${GPU_SET:-}" ]]; then
  echo "Detected GPUs: $GPU_IDS"
  read -rp "Which GPUs (comma-separated, e.g. 0,1,2,3): " gsel
  GPU_SET="${gsel:-$(seq -s, 0 3)}"
fi
TP_SIZE="${TP_SIZE:-$(echo "$GPU_SET" | awk -F, '{print NF}')}"

# --- Options (defaults = the tuned B70 values) -------------------------------
MAX_LEN="${MAX_LEN:-$(ask "Context length" "262144")}"
KV_DTYPE="${KV_DTYPE:-$(ask "KV cache dtype (fp8|auto)" "fp8")}"
GPU_UTIL="${GPU_UTIL:-$(ask "GPU memory utilization" "0.85")}"
PORT="${PORT:-$(ask "Port" "8000")}"
# B70 perf flags (from the B70 vLLM tuning gist):
BLOCK_SIZE="${BLOCK_SIZE:-$(ask "KV block size" "32")}"
MAX_BATCHED="${MAX_BATCHED:-$(ask "Max batched tokens (prefill)" "16384")}"
MAX_SEQS="${MAX_SEQS:-$(ask "Max concurrent sequences" "2")}"
LM_ONLY="${LM_ONLY:-$(ask "Language-model-only (skip vision tower)? (y/n)" "y")}"
PREFIX_CACHING="${PREFIX_CACHING:-$(ask "Prefix caching? (y/n)" "y")}"
SERVED_NAME="${SERVED_NAME:-$(ask "Served model name" "$(basename "$MODEL")")}"

# --- XPU environment ---------------------------------------------------------
export ONEAPI_DEVICE_SELECTOR="level_zero:$GPU_SET"
export ZE_AFFINITY_MASK="$GPU_SET"
export UR_L0_SYNC_MODE=BLOCKING
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export VLLM_TARGET_DEVICE=xpu
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export UVICORN_KEEP_ALIVE_TIMEOUT=300

# --- Build command -----------------------------------------------------------
CMD=(python3 -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_LEN"
  --kv-cache-dtype "$KV_DTYPE"
  --gpu-memory-utilization "$GPU_UTIL"
  --block-size "$BLOCK_SIZE"
  --max-num-batched-tokens "$MAX_BATCHED"
  --max-num-seqs "$MAX_SEQS"
  --trust-remote-code
)
[[ "${PREFIX_CACHING,,}" == "y" ]] && CMD+=(--enable-prefix-caching) || CMD+=(--no-enable-prefix-caching)
[[ "${LM_ONLY,,}" == "y" ]] && CMD+=(--language-model-only)

# Qwen models: add the reasoning/tool parsers automatically
if grep -qi "qwen" <<<"$MODEL"; then
  CMD+=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm)
fi

echo
echo "=== Launch command ==="
printf '%q ' "${CMD[@]}"; echo
echo "GPUs=$GPU_SET TP=$TP_SIZE ctx=$MAX_LEN kv=$KV_DTYPE util=$GPU_UTIL block=$BLOCK_SIZE batched=$MAX_BATCHED seqs=$MAX_SEQS lm-only=$LM_ONLY"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry-run: not launching)"
  exit 0
fi

source "$VENV/bin/activate"
exec "${CMD[@]}"
