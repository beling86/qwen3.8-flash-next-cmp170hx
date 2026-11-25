# SPDX-License-Identifier: Apache-2.0
"""Fused SM80 HyperConnection (GatedResidual) down+SiLU kernel for Qwen3.8-Flash-Next (M <= 32).

Fuses:
  xn [M, 10240] @ W_down [N, 10240]^T  (N=336 with injection or N=320 final mixer)
    -> lora_silu [M, 320] = silu(bf16(out[:, :320]) / hc_count)
    -> injection [M, 4]   = bf16(out[:, 320:324])  (when use_combine=True)

Uses single-launch split-K with deterministic fixed-order reduction (last arriving CTA
reduces partials 0..SPLIT_K-1), FP32 accumulation, .cg/.cv L2 coherence modifiers,
and exact BF16 boundary before silu to match upstream GatedResidual numerically.

Zero precision compromise: BF16 x BF16 -> FP32 accumulate -> BF16 stage boundary -> FP32
silu -> BF16 output. Bitwise deterministic run-to-run and CUDA-graph capturable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_MAX_M = 32
_MAX_N = 336
_MAX_SPLIT_K = 8
_MAX_TILES_N = 32
_SCRATCH: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = device.index or 0
    buf = _SCRATCH.get(idx)
    if buf is None:
        partials = torch.empty(
            (_MAX_SPLIT_K, _MAX_M, _MAX_N), dtype=torch.float32, device=device
        )
        locks = torch.zeros((_MAX_TILES_N,), dtype=torch.int32, device=device)
        buf = (partials, locks)
        _SCRATCH[idx] = buf
    return buf


@triton.jit
def _hc_down_silu_kernel(
    X,
    W,
    LORA,
    INJ,
    P,
    LOCK,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    LORA_RANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    USE_COMBINE: tl.constexpr,
    stride_xm,
    stride_wn,
    stride_lm,
    stride_im,
    stride_pk,
    stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    x_m = tl.where(m_mask, offs_m, 0)
    w_n = tl.where(n_mask, offs_n, 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step = BLOCK_K * SPLIT_K
    n_iter = tl.cdiv(K, step)
    k_base = pid_k * BLOCK_K

    for it in range(0, n_iter):
        k = k_base + it * step + offs_k
        k_mask = k < K
        a = tl.load(
            X + x_m[:, None] * stride_xm + k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            W + w_n[:, None] * stride_wn + k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(b), acc)

    if SPLIT_K == 1:
        y = acc.to(tl.bfloat16)
        lora_mask = m_mask[:, None] & (offs_n[None, :] < LORA_RANK)
        z = y.to(tl.float32) * (1.0 / HC_COUNT)
        s = (z * tl.sigmoid(z)).to(tl.bfloat16)
        tl.store(LORA + offs_m[:, None] * stride_lm + offs_n[None, :], s, mask=lora_mask)
        if USE_COMBINE:
            inj_mask = (
                m_mask[:, None]
                & (offs_n[None, :] >= LORA_RANK)
                & (offs_n[None, :] < LORA_RANK + HC_COUNT)
            )
            tl.store(
                INJ + offs_m[:, None] * stride_im + (offs_n[None, :] - LORA_RANK),
                y,
                mask=inj_mask,
            )
        return

    # Bypass non-coherent L1 with .cg so the reducing CTA on another SM sees our partial in L2.
    p_mask = m_mask[:, None] & n_mask[None, :]
    p_tile = P + pid_k * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :]
    tl.store(p_tile, acc, mask=p_mask, cache_modifier=".cg")

    # Release-semantics GPU-scoped arrival counter; last arriving CTA reduces 0..SPLIT_K-1.
    arrived = tl.atomic_add(LOCK + pid_n, 1, sem="release", scope="gpu")
    if arrived != SPLIT_K - 1:
        return

    tl.atomic_xchg(LOCK + pid_n, 0, sem="acquire", scope="gpu")

    total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sk in tl.static_range(0, SPLIT_K):
        total += tl.load(
            P + sk * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :],
            mask=p_mask,
            other=0.0,
            cache_modifier=".cv",
        )

    # Exact BF16 stage boundary matching upstream linear() -> hc_silu()
    y = total.to(tl.bfloat16)
    lora_mask = m_mask[:, None] & (offs_n[None, :] < LORA_RANK)
    z = y.to(tl.float32) * (1.0 / HC_COUNT)
    s = (z * tl.sigmoid(z)).to(tl.bfloat16)
    tl.store(LORA + offs_m[:, None] * stride_lm + offs_n[None, :], s, mask=lora_mask)
    if USE_COMBINE:
        inj_mask = (
            m_mask[:, None]
            & (offs_n[None, :] >= LORA_RANK)
            & (offs_n[None, :] < LORA_RANK + HC_COUNT)
        )
        tl.store(
            INJ + offs_m[:, None] * stride_im + (offs_n[None, :] - LORA_RANK),
            y,
            mask=inj_mask,
        )


def _select_hc_down_cfg(M: int) -> tuple[int, int, int, int, int, int]:
    """Measured CMP 170HX (90 SM, sm_80) winners for N in {320, 336}, K=10240."""
    if M <= 16:
        # 0.967 ms vs 1.726 ms cuBLAS+silu (1.79x) at M=5
        return 16, 16, 128, 8, 2, 4
    # 1.102 ms vs 1.452 ms cuBLAS+silu (1.32x) at M=20
    return 32, 32, 128, 8, 4, 3


def hc_down_silu(
    xn: torch.Tensor,
    w_down: torch.Tensor,
    lora_rank: int = 320,
    hc_count: int = 4,
    use_combine: bool = True,
    *,
    block_n: int | None = None,
    block_k: int | None = None,
    split_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused HC down projection + silu for M <= 32, with exact fallback for M > 32."""
    M, K = xn.shape
    N = w_down.shape[0]
    if (
        1 <= M <= _MAX_M
        and N <= _MAX_N
        and xn.dtype == torch.bfloat16
        and w_down.dtype == torch.bfloat16
        and xn.is_cuda
        and w_down.is_cuda
        and xn.stride(1) == 1
        and w_down.stride(1) == 1
    ):
        bm, bn, bk, sk, nw, ns = _select_hc_down_cfg(M)
        if block_n is not None:
            bn = block_n
        if block_k is not None:
            bk = block_k
        if split_k is not None:
            sk = split_k
        if num_warps is not None:
            nw = num_warps
        if num_stages is not None:
            ns = num_stages

        tiles_n = triton.cdiv(N, bn)
        partials, locks = _get_scratch(xn.device)
        lora = torch.empty((M, lora_rank), dtype=xn.dtype, device=xn.device)
        inj = torch.empty(
            (M, hc_count if use_combine else 0), dtype=xn.dtype, device=xn.device
        )
        _hc_down_silu_kernel[(tiles_n, sk)](
            xn,
            w_down,
            lora,
            inj,
            partials,
            locks,
            M,
            N=N,
            K=K,
            LORA_RANK=lora_rank,
            HC_COUNT=hc_count,
            USE_COMBINE=use_combine,
            stride_xm=xn.stride(0),
            stride_wn=w_down.stride(0),
            stride_lm=lora.stride(0),
            stride_im=inj.stride(0) if use_combine else 0,
            stride_pk=partials.stride(0),
            stride_pm=partials.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            SPLIT_K=sk,
            num_warps=nw,
            num_stages=ns,
        )
        return lora, inj

    # Fallback for M > 32 (prefill) or non-BF16/non-contiguous inputs.
    out = F.linear(xn, w_down)
    if use_combine:
        lora_raw = out[:, :lora_rank]
        inj = out[:, lora_rank : lora_rank + hc_count]
    else:
        lora_raw = out
        inj = torch.empty((M, 0), dtype=xn.dtype, device=xn.device)
    z = lora_raw.to(torch.float32) / hc_count
    lora = (z * torch.sigmoid(z)).to(xn.dtype)
    return lora, inj


def warmup(device: torch.device | None = None) -> None:
    """Pre-allocate persistent scratch and JIT-compile all M<=32 specializations before capture."""
    dev = device or torch.device("cuda", torch.cuda.current_device())
    _get_scratch(dev)
    for N, use_combine in ((336, True), (320, False)):
        w = torch.empty((N, 10240), dtype=torch.bfloat16, device=dev)
        for M in (5, 20):
            x = torch.empty((M, 10240), dtype=torch.bfloat16, device=dev)
            hc_down_silu(x, w, 320, 4, use_combine)
    torch.cuda.synchronize(dev)
