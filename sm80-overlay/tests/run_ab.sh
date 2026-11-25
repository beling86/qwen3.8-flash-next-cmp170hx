#!/usr/bin/env bash
# End-to-end A/B runner for Qwen3.8-Flash-Next SM80 phases.
# Usage: FN_SM80_HC=1 FN_SM80_THIN_GEMM=1 ./tests/run_ab.sh phase12
set -euo pipefail

TAG="${1:?Usage: run_ab.sh <tag>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8002}"
CNAME="fn-sm80-${TAG}"
LOG="${HERE}/results/${TAG}_server.log"
OUT="${HERE}/results/${TAG}.json"
PLE_CACHE_MB="${PLE_CACHE_MB:-8192}"
PLE_PREFETCH="${PLE_PREFETCH:-32768}"
KV_BYTES="${KV_BYTES:-10450000000}"
if [[ -z "${SPEC_CFG:-}" ]]; then
    SPEC_CFG='{"method":"dflash","model":"/home/beling/data/models/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":4,"kv_cache_dtype":"auto"}'
fi
if [[ -z "${COMP_CFG:-}" ]]; then
    COMP_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,5,10,15,20,25,30]}'
fi
EXTRA_ARGS=("${@:2}")

mkdir -p "${HERE}/results"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

cleanup() {
    docker rm -f "${CNAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

t_start=$(date +%s)
PORT="${PORT}" CONTAINER_NAME="${CNAME}" "${HERE}/serve_flash_next.sh" \
    --served-model-name qwen3.8-flash-next-dflash2 Qwen3.8-Flash-Next-DFlash2 \
    --language-model-only \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --max_num_seqs 6 \
    --max-model-len 262144 \
    --kv-cache-memory="${KV_BYTES}" \
    --max-num-batched-tokens 2048 \
    --safetensors-load-strategy lazy \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-config "${SPEC_CFG}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --override-generation-config '{"temperature":1,"top_p":0.95,"top_k":20,"repetition_penalty":1.0}' \
    --enable-per-request-metrics \
    --per-request-spec-decode-metrics summary \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --compilation-config "${COMP_CFG}" \
    --additional-config "{\"ple_ssd_offload\":true,\"ple_ssd_workers\":16,\"ple_ssd_cache_mb\":${PLE_CACHE_MB},\"ple_ssd_native_library\":\"/workspace/optimization/ple_ssd_io.so\",\"ple_ssd_io_depth\":256,\"ple_ssd_prefetch_tokens\":${PLE_PREFETCH}}" \
    --host 0.0.0.0 \
    "${EXTRA_ARGS[@]}" \
    >"${LOG}" 2>&1 &
SRV_PID=$!

echo "[${TAG}] Waiting for server on :${PORT} (container=${CNAME}, log=${LOG})..."
ready=0
for i in $(seq 1 900); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
        echo "[${TAG}] Server exited prematurely! Last 60 lines:"
        tail -n 60 "${LOG}"
        exit 1
    fi
    sleep 1
done

if [[ "${ready}" != "1" ]]; then
    echo "[${TAG}] Timed out waiting for server!"
    tail -n 60 "${LOG}"
    exit 1
fi

boot_s=$(( $(date +%s) - t_start ))
echo "[${TAG}] Server ready in ${boot_s}s!"
grep -E "FN_SM80_|GPU KV cache size|Maximum concurrency|PLE|adaptive|checkpoint|humming" "${LOG}" | tail -n 30 || true

python3 "${HERE}/tests/bench_flash_next.py" \
    --url "http://127.0.0.1:${PORT}/v1" \
    --model qwen3.8-flash-next-dflash2 \
    --tag "${TAG}" \
    --out "${OUT}"

if [[ -f "${HERE}/results/baseline.json" ]]; then
    echo ""
    python3 "${HERE}/tests/verify_e2e_parity.py" "${HERE}/results/baseline.json" "${OUT}"
fi
