#!/usr/bin/env python3
"""Phase 2 mathematical equivalence & CUDA-graph micro-benchmark (CMP 170HX)."""
import sys
sys.path.insert(0, "/sm80site")

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_silu, hc_gate_mix

import fn_sm80_hc as hc
import fn_sm80_thin_gemm as tg

device = "cuda"
torch.manual_seed(42)


def fp64_err(ref64: torch.Tensor, cand: torch.Tensor) -> tuple[float, float]:
    d = (ref64 - cand.double()).abs()
    return d.max().item(), d.mean().item()


def test_hc_equivalence():
    print("=== 1. MATHEMATICAL EQUIVALENCE: fn_sm80_hc (vs FP64 & upstream BF16) ===", flush=True)
    w_down = (torch.randn((336, 10240), dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    w_up = (torch.randn((10240, 320), dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    for M in (1, 2, 4, 5, 8, 10, 15, 20, 25, 30):
        xn = (torch.randn((M, 10240), dtype=torch.bfloat16, device=device) * 0.5).contiguous()

        # FP64 exact reference
        di64 = xn.double() @ w_down.double().T
        z64 = di64[:, :320] * 0.25
        lora64 = z64 * torch.sigmoid(z64)
        inj64 = di64[:, 320:324]

        # Upstream BF16 op
        di_u = F.linear(xn, w_down)
        lora_u = hc_silu(di_u[:, :320].contiguous(), 4)
        inj_u = di_u[:, 320:324].contiguous()

        # Our fused SM80 op (run twice to verify bitwise determinism!)
        lora_f1, inj_f1 = hc.hc_down_silu(xn, w_down, 320, 4, True)
        lora_f2, inj_f2 = hc.hc_down_silu(xn, w_down, 320, 4, True)
        assert torch.equal(lora_f1, lora_f2), f"NON-DETERMINISTIC lora at M={M}"
        assert torch.equal(inj_f1, inj_f2), f"NON-DETERMINISTIC inj at M={M}"

        e_u_max, e_u_mean = fp64_err(lora64, lora_u)
        e_f_max, e_f_mean = fp64_err(lora64, lora_f1)
        ei_u_max, _ = fp64_err(inj64, inj_u)
        ei_f_max, _ = fp64_err(inj64, inj_f1)

        # Up + gate_mix
        g64 = lora_u.double() @ w_up.double().T
        out64 = (torch.sigmoid(g64.view(M, 4, 2560)) * xn.double().view(M, 4, 2560)).mean(dim=1)
        gate_u = F.linear(lora_u, w_up)
        out_u = hc_gate_mix(xn, gate_u, 4)
        out_f1 = hc.hc_up_gate_mix(lora_u, w_up, xn, 2560)
        out_f2 = hc.hc_up_gate_mix(lora_u, w_up, xn, 2560)
        assert torch.equal(out_f1, out_f2), f"NON-DETERMINISTIC up_gate_mix at M={M}"
        eo_u_max, eo_u_mean = fp64_err(out64, out_u)
        eo_f_max, eo_f_mean = fp64_err(out64, out_f1)

        print(
            f"  M={M:2d} | down_silu fp64_err: up={e_u_max:.5f}/{e_u_mean:.6f} ours={e_f_max:.5f}/{e_f_mean:.6f} "
            f"| inj: up={ei_u_max:.5f} ours={ei_f_max:.5f} "
            f"| up_gmix: up={eo_u_max:.5f}/{eo_u_mean:.6f} ours={eo_f_max:.5f}/{eo_f_mean:.6f} | DET=OK",
            flush=True,
        )
        assert e_f_max <= max(e_u_max * 1.25, 1e-3), f"down_silu error too high at M={M}"
        assert eo_f_max <= max(eo_u_max * 1.25, 1e-3), f"up_gmix error too high at M={M}"


def test_thin_gemm_equivalence():
    print("\n=== 2. MATHEMATICAL EQUIVALENCE: fn_sm80_thin_gemm (vs FP64 & cuBLAS) ===", flush=True)
    for name, N, K in [("mlp.gate", 512, 2560), ("in_proj_ba", 96, 2560), ("shared_gate", 16, 2560)]:
        w = (torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.02).contiguous()
        for M in (1, 5, 15, 20, 30):
            x = (torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.5).contiguous()
            r64 = x.double() @ w.double().T
            y_cu = F.linear(x, w)
            y_t1 = tg.thin_gemm(x, w)
            y_t2 = tg.thin_gemm(x, w)
            assert torch.equal(y_t1, y_t2), f"NON-DETERMINISTIC thin_gemm {name} M={M}"
            ec_max, ec_mean = fp64_err(r64, y_cu)
            et_max, et_mean = fp64_err(r64, y_t1)
            print(f"  {name:12s} M={M:2d} | cuBLAS={ec_max:.5f}/{ec_mean:.6f} thin={et_max:.5f}/{et_mean:.6f} | DET=OK", flush=True)
            assert et_max <= max(ec_max * 1.25, 1e-3)


def bench_graph(build_step, n_replay=20):
    for _ in range(3):
        build_step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_step()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_replay):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def sweep_and_bench():
    print("\n=== 3. CUDA-GRAPH WEIGHT-ROTATED BENCHMARK (vs cuBLAS+Triton) ===", flush=True)
    K_ROT = 12
    w_downs = [torch.randn((336, 10240), dtype=torch.bfloat16, device=device) for _ in range(K_ROT)]
    w_ups = [torch.randn((10240, 320), dtype=torch.bfloat16, device=device) for _ in range(K_ROT)]
    R_ROT = 24
    w_gates = [torch.randn((512, 2560), dtype=torch.bfloat16, device=device) for _ in range(R_ROT)]
    w_bas = [torch.randn((96, 2560), dtype=torch.bfloat16, device=device) for _ in range(R_ROT)]

    for M in (5, 20):
        print(f"\n--- M={M} ---", flush=True)
        xn = torch.randn((M, 10240), dtype=torch.bfloat16, device=device)
        lora = torch.randn((M, 320), dtype=torch.bfloat16, device=device)
        xh = torch.randn((M, 2560), dtype=torch.bfloat16, device=device)

        # Baseline HC down+silu (x96)
        def base_down():
            for i in range(96):
                di = F.linear(xn, w_downs[i % K_ROT])
                _ = hc_silu(di[:, :320], 4)
        t_base_down = bench_graph(base_down)
        print(f"  HC down+silu (x96) baseline: {t_base_down:.3f} ms", flush=True)
        for bn, bk, sk, nw, ns in [
            (32, 128, 8, 4, 3),
            (32, 128, 16, 4, 3),
            (32, 64, 16, 4, 4),
            (16, 128, 8, 2, 4),
            (64, 128, 16, 4, 3),
        ]:
            t_f = bench_graph(lambda: [hc.hc_down_silu(xn, w_downs[i % K_ROT], 320, 4, True, block_n=bn, block_k=bk, split_k=sk, num_warps=nw, num_stages=ns) for i in range(96)])
            print(f"    fused bn={bn:2d} bk={bk:3d} sk={sk:2d} w={nw} s={ns}: {t_f:.3f} ms ({t_base_down/t_f:.2f}x)", flush=True)

        # Baseline HC up+gmix (x96)
        def base_up():
            for i in range(96):
                gt = F.linear(lora, w_ups[i % K_ROT])
                _ = hc_gate_mix(xn, gt, 4)
        t_base_up = bench_graph(base_up)
        print(f"  HC up+gmix   (x96) baseline: {t_base_up:.3f} ms", flush=True)
        for bn, bk, nw, ns in [
            (32, 64, 4, 3),
            (32, 64, 4, 4),
            (64, 64, 4, 3),
            (16, 64, 2, 4),
            (32, 32, 4, 4),
        ]:
            t_f = bench_graph(lambda: [hc.hc_up_gate_mix(lora, w_ups[i % K_ROT], xn, 2560, block_n=bn, block_k=bk, num_warps=nw, num_stages=ns) for i in range(96)])
            print(f"    fused bn={bn:2d} bk={bk:2d} w={nw} s={ns}: {t_f:.3f} ms ({t_base_up/t_f:.2f}x)", flush=True)

        # thin_gemm vs cuBLAS on mlp.gate (x48) & in_proj_ba (x36)
        t_gate_cu = bench_graph(lambda: [F.linear(xh, w_gates[i % R_ROT]) for i in range(48)])
        t_gate_tg = bench_graph(lambda: [tg.thin_gemm(xh, w_gates[i % R_ROT]) for i in range(48)])
        print(f"  mlp.gate   (x48): cuBLAS={t_gate_cu:.3f} ms vs thin_gemm={t_gate_tg:.3f} ms ({t_gate_cu/t_gate_tg:.2f}x)", flush=True)

        t_ba_cu = bench_graph(lambda: [F.linear(xh, w_bas[i % R_ROT]) for i in range(36)])
        t_ba_tg = bench_graph(lambda: [tg.thin_gemm(xh, w_bas[i % R_ROT]) for i in range(36)])
        print(f"  in_proj_ba (x36): cuBLAS={t_ba_cu:.3f} ms vs thin_gemm={t_ba_tg:.3f} ms ({t_ba_cu/t_ba_tg:.2f}x)", flush=True)


if __name__ == "__main__":
    test_hc_equivalence()
    test_thin_gemm_equivalence()
    sweep_and_bench()
