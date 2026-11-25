#!/usr/bin/env python3
"""Capture torch.profiler traces from vLLM (/start_profile, /stop_profile) and break down GPU kernel time by family."""
import argparse
import concurrent.futures
import gzip
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


def post_empty(url):
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def run_decode(base_url, model, prompt, max_tokens=160):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def classify(name: str) -> str:
    n = name.lower()
    if "humming" in n:
        return "1_Humming_MoE"
    if "marlin" in n:
        return "2_Marlin_Linear_MoE"
    if any(k in n for k in ["fused_recurrent", "gated_delta", "gdn", "causal_conv1d", "chunk_gated", "chunk_fwd", "wy_fast", "solve_tril", "l2norm", "cumsum"]):
        return "3_GDN_LinearAttn"
    if any(k in n for k in ["flash_fwd", "flash_attn", "qsa", "paged_attn", "reshape_and_cache", "indexer", "topk_indexer"]):
        return "4_QSA_FullAttn_DFlashAttn"
    if any(k in n for k in ["hc_", "grouped_gemma_rmsnorm", "hyper"]):
        return "5_HC_Glue_Triton"
    if any(k in n for k in ["gemm", "gemv", "cutlass", "cublas", "ampere_bf16", "ampere_s168", "splitk", "nvjet"]):
        return "6_BF16_GEMM_cuBLAS (HC+Gate+LMHead+DFlash)"
    if any(k in n for k in ["moe_align", "topk", "softmax", "moe_sum", "count_and_sort", "routing", "fused_moe"]):
        return "7_MoE_Router_TopK_Align_Sum"
    if any(k in n for k in ["rms_norm", "rmsnorm", "silu", "act_and_mul", "rotary", "rope"]):
        return "8_Norm_Act_RoPE"
    if any(k in n for k in ["memcpy", "memset", "copy", "cat", "scatter", "gather", "index", "embedding"]):
        return "9_Mem_Gather_PLE_Copy"
    if any(k in n for k in ["reject", "sample", "gumbel", "argmax"]):
        return "10_Sampling_SpecVerify"
    return "11_Other"


def parse_trace(trace_path: Path):
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    by_cat = defaultdict(float)
    by_cat_cnt = defaultdict(int)
    by_kernel = defaultdict(float)
    by_kernel_cnt = defaultdict(int)
    total_us = 0.0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(ev.get("dur", 0.0))
        name = ev.get("name") or "unknown"
        bucket = classify(name)
        by_cat[bucket] += dur
        by_cat_cnt[bucket] += 1
        by_kernel[name] += dur
        by_kernel_cnt[name] += 1
        total_us += dur
    return total_us, dict(by_cat), dict(by_cat_cnt), dict(by_kernel), dict(by_kernel_cnt)


def capture_case(base_url, model, trace_dir: Path, label: str, n_users: int):
    before = set(trace_dir.glob("**/*.json*"))
    post_empty(f"{base_url.rstrip('/')}/start_profile")
    prompts = [
        f"[prof-{label}-{i}-{int(time.time())}] Write a detailed Python implementation of a red-black tree with comments."
        for i in range(n_users)
    ]
    if n_users == 1:
        run_decode(base_url, model, prompts[0], max_tokens=120)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as ex:
            list(ex.map(lambda p: run_decode(base_url, model, p, max_tokens=100), prompts))
    post_empty(f"{base_url.rstrip('/')}/stop_profile")
    # Wait up to 30s for trace file to finish flushing
    new_files = []
    for _ in range(60):
        after = set(trace_dir.glob("**/*.json*"))
        cand = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
        if cand and cand[0].stat().st_size > 1024:
            time.sleep(2)
            new_files = cand
            break
        time.sleep(0.5)
    if not new_files:
        raise RuntimeError(f"No trace file written in {trace_dir} for {label}")
    tf = new_files[0]
    total_us, by_cat, by_cat_cnt, by_kernel, by_kernel_cnt = parse_trace(tf)
    print(f"\n=== PROFILE [{label}] ({tf.name}, total GPU kernel time: {total_us/1000:.2f} ms) ===", flush=True)
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / max(total_us, 1e-9)
        print(f"  {k:45s}: {v/1000:8.2f} ms ({pct:5.1f}%) [{by_cat_cnt[k]:6d} calls]", flush=True)
    print(f"  --- Top 20 individual kernels [{label}] ---", flush=True)
    for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:20]:
        pct = 100.0 * v / max(total_us, 1e-9)
        short = k if len(k) <= 95 else k[:92] + "..."
        print(f"    {v/1000:7.2f} ms ({pct:5.1f}%) [{by_kernel_cnt[k]:5d}x] {short}", flush=True)
    return {
        "trace_file": str(tf),
        "total_gpu_ms": round(total_us / 1000.0, 2),
        "by_category_ms": {k: round(v / 1000.0, 2) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "by_category_pct": {k: round(100.0 * v / max(total_us, 1e-9), 2) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "top_25_kernels": [
            {"name": k, "ms": round(v / 1000.0, 2), "pct": round(100.0 * v / max(total_us, 1e-9), 2), "calls": by_kernel_cnt[k]}
            for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:25]
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="qwen3.8-flash-next-dflash2")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    td = Path(args.trace_dir)
    td.mkdir(parents=True, exist_ok=True)
    summary = {}
    summary["decode_1u"] = capture_case(args.base_url, args.model, td, "decode_1u", 1)
    summary["decode_4u"] = capture_case(args.base_url, args.model, td, "decode_4u", 4)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved profile summary to {args.out}", flush=True)


if __name__ == "__main__":
    main()
