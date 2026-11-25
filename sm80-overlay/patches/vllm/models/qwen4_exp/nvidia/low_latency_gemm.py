# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection on Hopper and Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import os

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_FN_SM80_THIN_GEMM_ENABLED = os.environ.get("FN_SM80_THIN_GEMM", "0") == "1"
_FN_SM80_HC_ENABLED = os.environ.get("FN_SM80_HC", "0") == "1"
_FN_SM80_MOE_ROUTE_ENABLED = os.environ.get("FN_SM80_MOE_ROUTE", "0") == "1"
_FN_SM80_MAX_M = 32

# Measured winning BF16 (N, K) shapes on CMP 170HX (sm_80, 90 SMs).
# LM head (248320, 2560) is intentionally excluded because cuBLAS already saturates HBM (~1.7 TB/s).
QWEN4_EXP_SM80_THIN_SHAPES: frozenset[tuple[int, int]] = frozenset({
    (10240, 320),  # HC input_mix_weight_up (x97): 1.23x @ M=5, 1.25x @ M=20
    (512, 2560),   # MoE router mlp.gate (x48): 1.40x @ M=5, 1.17x @ M=20
    (96, 2560),    # GDN in_proj_ba (x36): 1.85x @ M=5, 1.51x @ M=20
    (1, 2560),     # MoE shared_expert_gate (x48): 1.14x @ M=5, 1.39x @ M=20
})

QWEN4_EXP_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
    },
}

# H200 plans selected by exhaustive CUDA graph replay measurements over
# M={1, 2, 4, 8, 16}. Only points that beat the standard linear implementation
# in both hot-cache and L2-flush measurements are retained; other token counts
# keep the standard implementation and its GEMM heuristics.
QWEN4_EXP_SM90_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=2560),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=6, vector_width=4),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 3, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=5),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 256, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 2, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=5, vector_width=4),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=2, static_k=2560),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=5),
        2: SkinnyGemmConfig(2, 256, 3, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 3, static_k=10240),
        8: SkinnyGemmConfig(8, 256, 3, static_k=10240),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
    },
}


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_sm90() -> bool:
    return current_platform.is_device_capability((9, 0))


def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
    if _is_sm103():
        return QWEN4_EXP_GEMM_PLANS
    if _is_sm90():
        return QWEN4_EXP_SM90_GEMM_PLANS
    return {}


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen4ExpLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpLowLatencyLinearMethod(_Qwen4ExpLowLatencyApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpLowLatencyEmbeddingMethod(
    _Qwen4ExpLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def _is_sm80() -> bool:
    return current_platform.is_device_capability((8, 0))


def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        _FN_SM80_THIN_GEMM_ENABLED
        and _is_sm80()
        and 1 <= x.shape[0] <= _FN_SM80_MAX_M
        and (weight.shape[0], weight.shape[1]) in QWEN4_EXP_SM80_THIN_SHAPES
        and _runtime_ok(x, weight)
    ):
        from fn_sm80_thin_gemm import thin_gemm

        return thin_gemm(x, weight)

    plan = _gemm_plans().get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


def _qwen4_exp_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_low_latency_gemm",
    op_func=_qwen4_exp_low_latency_gemm,
    fake_impl=_qwen4_exp_low_latency_gemm_fake,
)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    if dtype != torch.bfloat16:
        return

    if _is_sm80():
        if _FN_SM80_HC_ENABLED:
            from fn_sm80_hc import warmup as warmup_hc

            warmup_hc()
            logger.info("FN_SM80_HC enabled: warmed up fused HC down+SiLU kernel (M<=32)")

        if _FN_SM80_MOE_ROUTE_ENABLED:
            from fn_sm80_moe_route import warmup as warmup_moe_route

            bs_list = (8,) if os.environ.get("FN_SM80_HUMMING_0120", "0") == "1" else (16,)
            # Warm up M=1..30 so any adaptive-K / rounded CUDA-graph capture size is precompiled.
            for bs in bs_list:
                warmup_moe_route(ms=tuple(range(1, 31)), block_size=bs)
            logger.info(
                "FN_SM80_MOE_ROUTE enabled: warmed up fused MoE route+softmax+align (E=512, K=2560, topk=10, bs=%s, M=1..30)",
                bs_list,
            )

        if _FN_SM80_THIN_GEMM_ENABLED:
            from fn_sm80_thin_gemm import warmup as warmup_thin

            matched_shapes: set[tuple[int, int]] = set()
            count = 0
            for child in module.modules():
                is_linear = (
                    isinstance(child, LinearBase)
                    and type(child.quant_method) is UnquantizedLinearMethod
                )
                if not is_linear:
                    continue
                weight = getattr(child, "weight", None)
                if weight is None or weight.dim() != 2:
                    continue
                shape = (weight.shape[0], weight.shape[1])
                if shape not in QWEN4_EXP_SM80_THIN_SHAPES:
                    continue
                child.quant_method = Qwen4ExpLowLatencyLinearMethod()
                matched_shapes.add(shape)
                count += 1
            if matched_shapes:
                # Warm up M=5 (BLOCK_M=16) and M=20 (BLOCK_M=32) to JIT-compile & allocate shared workspace.
                warmup_thin(sorted(matched_shapes), (5, 20))
                logger.info(
                    "FN_SM80_THIN_GEMM enabled: patched %d BF16 layers across shapes %s (M<=%d)",
                    count,
                    sorted(matched_shapes),
                    _FN_SM80_MAX_M,
                )
        return

    plans = _gemm_plans()
    if not plans:
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = plans.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen4ExpLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen4ExpLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
