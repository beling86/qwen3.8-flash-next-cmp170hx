# SPDX-License-Identifier: Apache-2.0
"""Fused SM80 MoE Router GEMM + Softmax Top-10 + Deterministic Block Alignment (M <= 32).

Adapted from Morrowmake/vllm-cmp170hx `vllm_sm80/kernels/moe/route.py` for
Qwen3.8-Flash-Next (E=512, K=2560, top_k=10, softmax, renormalize=True, block_size=16).

Replaces three separate launches per MoE layer (x48 layers):
  1. mlp.gate:              x [M, 2560] @ W [512, 2560]^T
  2. ops.topk_softmax:      softmax + top-10 + renormalize
  3. ops.moe_align_block_size: histogram + prefix sum + scatter

With ONE single Triton launch:
  - Tensor-core split-K GEMM CTAs (with per-128-K `add.rn.f32` to avoid sm_80 MMA truncation drift)
  - One routing CTA per token row (waits via single-thread `ld.acquire.gpu`, sums partials
    in fixed 0..SPLIT-1 order, bitonic top-16 on packed int64 keys, exact register unpack,
    FP32 softmax + renorm over top-10, 32-bit column bitmask `atomic_or`)
  - Last arriving routing CTA computes deterministic `sorted_ids`, `expert_ids`, `num_tokens_post_pad`
    and resets scratch buffers to 0 for CUDA-graph replay.

Zero precision compromise, 100% bitwise deterministic, CUDA-graph capturable.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_MAX_M = 32
_MAX_SPLIT = 8
_SCRATCH: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_BF16_LOGITS_DEFAULT = os.environ.get("FN_SM80_MOE_ROUTE_BF16_LOGITS", "0") == "1"

# Single-slot handoff stashes (valid only for the immediately following consumer on the same thread).
_ROUTE_STASH: tuple[int, tuple[torch.Tensor, torch.Tensor]] | None = None
_ALIGN_STASH: tuple[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None


def stash_route_result(
    key_tensor: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> None:
    global _ROUTE_STASH
    _ROUTE_STASH = (id(key_tensor), (topk_weights, topk_ids))


def take_route_result(
    key_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _ROUTE_STASH
    entry = _ROUTE_STASH
    _ROUTE_STASH = None
    if entry is not None and entry[0] == id(key_tensor):
        return entry[1]
    return None


def stash_align_result(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    result: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    global _ALIGN_STASH
    _ALIGN_STASH = ((id(topk_ids), int(block_size), int(num_experts)), result)


def take_align_result(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    global _ALIGN_STASH
    entry = _ALIGN_STASH
    _ALIGN_STASH = None
    if entry is not None and entry[0] == (id(topk_ids), int(block_size), int(num_experts)):
        return entry[1]
    return None


def _get_scratch(
    device: torch.device, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device.index or 0, int(num_experts))
    buf = _SCRATCH.get(key)
    if buf is None:
        part = torch.empty(
            _MAX_SPLIT * _MAX_M * num_experts, dtype=torch.float32, device=device
        )
        col = torch.zeros(num_experts, dtype=torch.int32, device=device)
        arrive = torch.zeros(2, dtype=torch.int32, device=device)
        buf = (part, col, arrive)
        _SCRATCH[key] = buf
    return buf


def align_sizes(numel: int, num_experts: int, block_size: int) -> tuple[int, int]:
    """Exact match to vLLM `moe_align_block_size` output buffer sizing."""
    mnp = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        mnp = min(numel * block_size, mnp)
    return mnp, triton.cdiv(mnp, block_size)


@triton.jit
def _spin_one_thread(ptr, target):
    """Thread 0 spins on `ld.acquire.gpu` until `*ptr >= target`."""
    return tl.inline_asm_elementwise(
        """{
        .reg .pred p;
        .reg .u32 t;
        .reg .s32 v;
        mov.u32 t, %tid.x;
        setp.ne.u32 p, t, 0;
        @p bra SPIN_DONE_${:uid};
        SPIN_LOOP_${:uid}:
        ld.acquire.gpu.global.b32 v, [$1];
        setp.lt.s32 p, v, $2;
        @p bra SPIN_LOOP_${:uid};
        SPIN_DONE_${:uid}:
        mov.u32 $0, 0;
        }""",
        "=r,l,r",
        [ptr, target],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gemv_tc_part(
    x_ptr,
    w_ptr,
    part_ptr,
    done_ptr,
    pid_n,
    pid_s,
    stride_xm,
    stride_wn,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    GBM: tl.constexpr,
    GBE: tl.constexpr,
    GBK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    KS: tl.constexpr = K // SPLIT
    offs_m = tl.arange(0, GBM)
    offs_e = pid_n * GBE + tl.arange(0, GBE)
    offs_k = tl.arange(0, GBK)
    mask_m = offs_m < M
    mask_e = offs_e < E
    x_m = tl.where(mask_m, offs_m, 0)
    w_e = tl.where(mask_e, offs_e, 0)

    k0 = pid_s * KS
    x_ptrs = x_ptr + x_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = w_ptr + w_e[:, None] * stride_wn + (k0 + offs_k)[None, :]

    acc = tl.zeros([GBM, GBE], tl.float32)
    for _ in range(KS // GBK):
        xk = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        wk = tl.load(w_ptrs, mask=mask_e[:, None], other=0.0)
        d = tl.dot(xk, tl.trans(wk), out_dtype=tl.float32)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=r,r,r",
            [acc, d],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        x_ptrs += GBK
        w_ptrs += GBK

    p_off = offs_m[:, None] * E + offs_e[None, :]
    tl.store(
        part_ptr + pid_s * (M * E) + p_off,
        acc,
        mask=mask_m[:, None] & mask_e[None, :],
        cache_modifier=".cg",
    )
    tl.debug_barrier()
    tl.atomic_add(done_ptr, 1, sem="release", scope="gpu")


@triton.jit
def _pair_ids(topk_ids_ptr, i0, CH: tl.constexpr, NUMEL: tl.constexpr):
    i = i0 + tl.arange(0, CH)
    return tl.load(topk_ids_ptr + i, mask=i < NUMEL, other=-1, cache_modifier=".cg")


@triton.jit
def _place_pairs_ng1(
    src,
    e_raw,
    i0,
    sorted_ids_ptr,
    expert_ids_ptr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    CH: tl.constexpr,
):
    i = i0 + tl.arange(0, CH)
    vi = e_raw >= 0
    e = tl.where(vi, e_raw, 0)
    m = i // TOPK
    low = (tl.full([CH], 1, tl.int32) << m) - 1
    pk = tl.gather(src, e, 0)
    ex = (pk >> 32).to(tl.int32)
    wg = pk.to(tl.int32)
    rank = libdevice.popc(wg & low)
    pos = ex + rank
    tl.store(sorted_ids_ptr + pos, i.to(tl.int32), mask=vi)
    tl.store(expert_ids_ptr + pos // BS, e, mask=vi & (rank % BS == 0))


@triton.jit
def _fn_sm80_moe_route_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    topk_w_ptr,
    topk_ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    ntpp_ptr,
    col_ptr,
    arrive_ptr,
    stride_xm,
    stride_wn,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BS: tl.constexpr,
    MNP: tl.constexpr,
    NBLK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FILL_P: tl.constexpr,
    GBM: tl.constexpr,
    GBE: tl.constexpr,
    GBK: tl.constexpr,
    SPLIT: tl.constexpr,
    BF16_LOGITS: tl.constexpr,
):
    """Single-launch router GEMM + softmax top-K + deterministic block align (1 <= M <= 32)."""
    NT: tl.constexpr = (E + GBE - 1) // GBE
    NGEMV: tl.constexpr = NT * SPLIT
    NUMEL: tl.constexpr = M * TOPK
    pid = tl.program_id(0)

    if pid < NGEMV:
        _gemv_tc_part(
            x_ptr,
            w_ptr,
            part_ptr,
            arrive_ptr + 1,
            pid % NT,
            pid // NT,
            stride_xm,
            stride_wn,
            M,
            K,
            E,
            GBM,
            GBE,
            GBK,
            SPLIT,
        )
        return

    row = pid - NGEMV
    # Fill this row's slice of sentinel values while GEMV runs (or right before wait)!
    # Doing sentinel fill BEFORE waiting hides HBM fill latency behind the GEMV!
    p = row * FILL_P + tl.arange(0, FILL_P)
    tl.store(sorted_ids_ptr + p, tl.full([FILL_P], NUMEL, tl.int32), mask=p < MNP)
    tl.store(expert_ids_ptr + p, tl.full([FILL_P], -1, tl.int32), mask=p < NBLK)

    # Wait for all GEMV CTAs to publish their partials
    _spin_one_thread(arrive_ptr + 1, NGEMV)
    tl.debug_barrier()

    offs_e = tl.arange(0, BLOCK_E)
    ve = offs_e < E
    tot = tl.zeros([BLOCK_E], tl.float32)
    for s in tl.static_range(SPLIT):
        tot += tl.load(
            part_ptr + s * (M * E) + row * E + offs_e,
            mask=ve,
            other=0.0,
            cache_modifier=".cv",
        )

    if BF16_LOGITS:
        tot = tot.to(tl.bfloat16).to(tl.float32)

    # Exact monotone float32 -> int32 key packed with (E - 1 - e) so lower expert wins ties.
    key = tl.where(ve, tot, float("-inf"))
    b32 = key.to(tl.int32, bitcast=True)
    m32 = b32 ^ ((b32 >> 31) & 0x7FFFFFFF)
    packed = (m32.to(tl.int64) << 32) | (E - 1 - offs_e).to(tl.int64)
    packed = tl.where(ve, packed, -0x8000000000000000)

    # Bitonic top-BLOCK_K (BLOCK_K=16 for TOPK=10) over all 512 experts in registers.
    top = tl.topk(tl.reshape(packed, [1, BLOCK_E]), BLOCK_K)
    top1d = tl.reshape(top, [BLOCK_K])
    sel_i = (E - 1 - (top1d & 0xFFFFFFFF)).to(tl.int32)
    top_m32 = (top1d >> 32).to(tl.int32)
    # Involution: f(f(b)) == b recovers exact FP32 logit without any HBM round-trip!
    sel_x = (top_m32 ^ ((top_m32 >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)

    offs_k = tl.arange(0, BLOCK_K)
    keep = offs_k < TOPK
    row_max = tl.max(tl.where(keep, sel_x, float("-inf")), axis=0)
    exps = tl.where(keep, tl.exp(sel_x - row_max), 0.0)
    w = exps / tl.sum(exps, axis=0)

    out_off = row * TOPK + offs_k
    tl.store(topk_ids_ptr + out_off, sel_i, mask=keep)
    tl.store(topk_w_ptr + out_off, w, mask=keep)

    # Mark this row's bit in each selected expert's column word (M <= 32 -> single int32 word)
    bit = tl.full([BLOCK_K], 1, tl.int32) << row
    tl.atomic_or(col_ptr + sel_i, bit, mask=keep, sem="relaxed", scope="gpu")
    tl.debug_barrier()

    arrived = tl.atomic_add(arrive_ptr, 1, sem="acq_rel", scope="gpu")
    if arrived != M - 1:
        return

    # Last arriving routing CTA builds deterministic alignment in O(E + NUMEL).
    CH: tl.constexpr = BLOCK_N if BLOCK_N < 128 else 128
    NCH: tl.constexpr = BLOCK_N // CH

    e0 = _pair_ids(topk_ids_ptr, 0, CH, NUMEL)
    if NCH > 1:
        e1 = _pair_ids(topk_ids_ptr, CH, CH, NUMEL)
    if NCH > 2:
        e2 = _pair_ids(topk_ids_ptr, 2 * CH, CH, NUMEL)
    if NCH > 3:
        e3 = _pair_ids(topk_ids_ptr, 3 * CH, CH, NUMEL)

    w0 = tl.load(col_ptr + offs_e, mask=ve, other=0, cache_modifier=".cv")
    cnt = libdevice.popc(w0)
    padded = tl.where(ve, ((cnt + BS - 1) // BS) * BS, 0)
    incl = tl.cumsum(padded, axis=0)
    excl = incl - padded

    # Store total padded tokens (last element of incl)
    RQ: tl.constexpr = BLOCK_E // 4
    tail = tl.max(tl.reshape(incl, [RQ, 4]), axis=1)
    rq = tl.arange(0, RQ)
    tl.store(ntpp_ptr + rq * 0, tail, mask=rq == RQ - 1)

    src = (excl.to(tl.int64) << 32) | (w0.to(tl.int64) & 0xFFFFFFFF)
    _place_pairs_ng1(src, e0, 0, sorted_ids_ptr, expert_ids_ptr, TOPK, BS, CH)
    if NCH > 1:
        _place_pairs_ng1(src, e1, CH, sorted_ids_ptr, expert_ids_ptr, TOPK, BS, CH)
    if NCH > 2:
        _place_pairs_ng1(src, e2, 2 * CH, sorted_ids_ptr, expert_ids_ptr, TOPK, BS, CH)
    if NCH > 3:
        _place_pairs_ng1(src, e3, 3 * CH, sorted_ids_ptr, expert_ids_ptr, TOPK, BS, CH)

    # Reset scratch to zero for next call / CUDA-graph replay
    tl.store(col_ptr + offs_e, tl.zeros([BLOCK_E], tl.int32), mask=ve)
    tl.store(arrive_ptr + 1, 0)
    tl.store(arrive_ptr, 0)


def _next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1)).bit_length()


def _select_route_cfg(M: int) -> tuple[int, int, int, int, int]:
    """Measured CMP 170HX (70 SMs, sm_80) winners for E=512, K=2560, topk=10, bs=16."""
    if M <= 8:
        # 0.514 ms vs 1.306 ms baseline (2.54x) @ M=5
        return 32, 128, 5, 4, 3
    if M <= 16:
        # 0.548 ms vs 1.306 ms baseline (2.38x) @ M=10
        return 32, 128, 4, 4, 4
    if M <= 24:
        # 0.604 ms vs 1.297 ms baseline (2.15x) @ M=20
        return 32, 128, 4, 4, 3
    # 0.632 ms vs 1.326 ms baseline (2.10x) @ M=30
    return 32, 128, 4, 4, 4


def moe_route_align(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk: int = 10,
    block_size: int = 16,
    *,
    gbe: int | None = None,
    gbk: int | None = None,
    split: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    bf16_logits: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch router GEMM + softmax top-k + deterministic block align.

    Returns (topk_weights, topk_ids, sorted_ids, expert_ids, num_tokens_post_pad).
    """
    M, K = x.shape
    E = weight.shape[0]
    assert 1 <= M <= _MAX_M
    d_gbe, d_gbk, d_split, d_nw, d_ns = _select_route_cfg(M)
    gbe = d_gbe if gbe is None else gbe
    gbk = d_gbk if gbk is None else gbk
    split = d_split if split is None else split
    num_warps = d_nw if num_warps is None else num_warps
    num_stages = d_ns if num_stages is None else num_stages
    if bf16_logits is None:
        bf16_logits = _BF16_LOGITS_DEFAULT
    assert K % (gbk * split) == 0, f"K={K} must be divisible by {gbk * split}"
    dev = x.device
    numel = M * topk
    mnp, nblk = align_sizes(numel, E, block_size)

    topk_w = torch.empty((M, topk), dtype=torch.float32, device=dev)
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=dev)
    sorted_ids = torch.empty((mnp,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((nblk,), dtype=torch.int32, device=dev)
    ntpp = torch.empty((1,), dtype=torch.int32, device=dev)

    part, col, arrive = _get_scratch(dev, E)
    gbm = 16 if M <= 16 else 32
    block_k = max(4, _next_pow2(topk))
    block_e = max(64, _next_pow2(E))
    block_n = max(16, _next_pow2(numel))
    fill_p = _next_pow2(triton.cdiv(max(mnp, nblk), M))
    nt = triton.cdiv(E, gbe)
    grid = (nt * split + M,)

    _fn_sm80_moe_route_kernel[grid](
        x,
        weight,
        part,
        topk_w,
        topk_ids,
        sorted_ids,
        expert_ids,
        ntpp,
        col,
        arrive,
        x.stride(0),
        weight.stride(0),
        M=M,
        K=K,
        E=E,
        BS=block_size,
        MNP=mnp,
        NBLK=nblk,
        TOPK=topk,
        BLOCK_K=block_k,
        BLOCK_E=block_e,
        BLOCK_N=block_n,
        FILL_P=fill_p,
        GBM=gbm,
        GBE=gbe,
        GBK=gbk,
        SPLIT=split,
        BF16_LOGITS=bf16_logits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return topk_w, topk_ids, sorted_ids, expert_ids, ntpp


def warmup(
    ms=(1, 2, 5, 10, 15, 20, 25, 30),
    topk: int = 10,
    block_size: int = 16,
    num_experts: int = 512,
    hidden: int = 2560,
    device: torch.device | None = None,
) -> None:
    dev = device or torch.device("cuda", torch.cuda.current_device())
    w = torch.zeros((num_experts, hidden), dtype=torch.bfloat16, device=dev)
    for m in sorted({int(v) for v in ms}):
        x = torch.zeros((m, hidden), dtype=torch.bfloat16, device=dev)
        moe_route_align(x, w, topk=topk, block_size=block_size)
    torch.cuda.synchronize(dev)
