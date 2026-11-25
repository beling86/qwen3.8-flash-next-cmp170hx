#!/usr/bin/env bash
# Launcher for Qwen3.8-Flash-Next SM80 (vllm-cmp170hx:qwen3.8-flash-next + bind-mounted overlays).
# Used both for standalone A/B testing (PORT=8002) and llama-swap serving (PORT=8001).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/home/beling/data/models/Qwen3.8-Flash-Next-3.43BPW-MTP}"
DRAFT="${DRAFT:-/home/beling/data/models/Qwen3.8-Flash-Next-DFlash2}"
CACHE="${CACHE:-/home/beling/.cache/qwen38-flash-next}"
IMAGE="${IMAGE:-vllm-cmp170hx:qwen3.8-flash-next}"
VLLM_SRC="/workspace/vllm/src/vllm"
PORT="${PORT:-}"
TRACE_DIR="${TRACE_DIR:-}"
FN_SM80_HC="${FN_SM80_HC:-1}"
FN_SM80_THIN_GEMM="${FN_SM80_THIN_GEMM:-1}"
FN_SM80_MOE_ROUTE="${FN_SM80_MOE_ROUTE:-1}"
FN_SM80_HUMMING_0120="${FN_SM80_HUMMING_0120:-1}"
NAME_ARGS=()
if [[ -n "${CONTAINER_NAME:-}" ]]; then
    NAME_ARGS=(--name "${CONTAINER_NAME}")
fi

mkdir -p "${CACHE}" "${HERE}/patches/vllm" "${HERE}/site"

mounts=()
while IFS= read -r f; do
    mounts+=(-v "${HERE}/patches/vllm/${f}:${VLLM_SRC}/${f}:ro")
done < <(cd "${HERE}/patches/vllm" && find . -name '*.py' | sed 's|^\./||' | sort)

# Optional Humming 0.1.20 drop-in upgrade (Phase 6c: SM80 mma_m8, ldmatrix dequant, 2-CTA tuning)
if [[ "${FN_SM80_HUMMING_0120:-0}" == "1" ]] && [[ -d "${HERE}/upstream/humming_0120/humming" ]]; then
    mounts+=(-v "${HERE}/upstream/humming_0120/humming:/workspace/vllm/.venv/lib/python3.12/site-packages/humming:ro")
fi

# Optional Humming epilogue header override (Phase 6c)
if [[ "${FN_SM80_HUMMING_EPILOGUE:-0}" == "1" ]] && [[ -f "${HERE}/site/humming_include/humming/epilogue/gmem_writer.cuh" ]]; then
    mounts+=(-v "${HERE}/site/humming_include/humming/epilogue/gmem_writer.cuh:/workspace/vllm/.venv/lib/python3.12/site-packages/humming/include/humming/epilogue/gmem_writer.cuh:ro")
fi

extra_docker=()
extra_vllm=()
if [[ -n "${PORT}" ]]; then
    extra_vllm+=(--port "${PORT}")
fi
if [[ -n "${TRACE_DIR}" ]]; then
    mkdir -p "${TRACE_DIR}"
    extra_docker+=(-v "${TRACE_DIR}:/traces")
    extra_vllm+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_with_stack":false}')
fi

exec docker run --rm --init "${NAME_ARGS[@]}" \
    --gpus all --ipc=host --network host \
    -v /home/beling/data/models:/home/beling/data/models \
    -v "${CACHE}:/root/.cache" \
    -v "${HERE}/site:/sm80site:ro" \
    "${mounts[@]}" \
    "${extra_docker[@]}" \
    -e PYTHONPATH="/sm80site:/workspace/vllm/src" \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e HF_HUB_OFFLINE=1 \
    -e FN_SM80_HC="${FN_SM80_HC}" \
    -e FN_SM80_THIN_GEMM="${FN_SM80_THIN_GEMM}" \
    -e FN_SM80_MOE_ROUTE="${FN_SM80_MOE_ROUTE}" \
    -e FN_SM80_ADAPTIVE_K="${FN_SM80_ADAPTIVE_K:-0}" \
    -e FN_SM80_DECODE_CKPT="${FN_SM80_DECODE_CKPT:-0}" \
    -e FN_SM80_FLA_PINS="${FN_SM80_FLA_PINS:-0}" \
    -e FN_SM80_HUMMING_0120="${FN_SM80_HUMMING_0120}" \
    -e FN_SM80_HUMMING_EPILOGUE="${FN_SM80_HUMMING_EPILOGUE:-0}" \
    -w /workspace \
    "${IMAGE}" \
    serve "${MODEL}" \
    "${extra_vllm[@]}" \
    "$@"
