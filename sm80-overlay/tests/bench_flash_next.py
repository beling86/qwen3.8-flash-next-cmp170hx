#!/usr/bin/env python3
"""Standardized A/B benchmark & parity capture harness for Qwen3.8-Flash-Next SM80."""
import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

DOCS_PATH = Path("/home/beling/Model Benchmarks/datasets/06_long_doc_extraction/documents.jsonl")

DECODE_PROMPTS = [
    (
        "list",
        "Output a numbered list of integers from 1 to 300, one per line, in exactly this format:\n"
        "1 = one\n2 = two\n3 = three\nNever use ellipses or stop early. Begin immediately:",
    ),
    (
        "code",
        "Write a complete, well-commented Python implementation of an in-memory LRU cache "
        "with per-key TTL expiration, thread safety via threading.RLock, O(1) get/put via "
        "a custom doubly-linked list, and a unittest suite.",
    ),
    (
        "prose",
        "Write an exhaustive engineering handbook on serving transformer language models in "
        "production with numbered sections (1. through 15.) covering tokenisation, paged KV cache, "
        "continuous batching, speculative decoding, quantization, CUDA graphs, and observability. "
        "Begin immediately with section 1.",
    ),
    (
        "chat",
        "Explain in depth how TCP Cubic and TCP BBR differ in congestion control, how BBR "
        "estimates bottleneck bandwidth and minimum RTT, why bufferbloat breaks loss-based "
        "algorithms, and what fairness issues arise when Cubic and BBR flows share a bottleneck queue.",
    ),
]


def vram_info():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    u, f, t = [int(x.strip()) for x in out.splitlines()[0].split(",")]
    return {"used_mib": u, "free_mib": f, "total_mib": t}


class PeakVRAMMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_used = 0
        self.min_free = 10**9
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                v = vram_info()
                if v["used_mib"] > self.peak_used:
                    self.peak_used = v["used_mib"]
                if v["free_mib"] < self.min_free:
                    self.min_free = v["free_mib"]
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=2)


def stream_chat(
    url,
    model,
    messages,
    max_tokens=400,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    logprobs=False,
    top_logprobs=None,
    timeout=600,
):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs:
            payload["top_logprobs"] = top_logprobs

    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    content_parts = []
    reasoning_parts = []
    lp_samples = []
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0
    chunks = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                completion_tokens = usage.get("completion_tokens") or completion_tokens
                ptd = usage.get("prompt_tokens_details") or {}
                if ptd.get("cached_tokens") is not None:
                    cached_tokens = ptd["cached_tokens"]
            choices = obj.get("choices") or []
            if choices:
                ch0 = choices[0]
                delta = ch0.get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if (c or r) and t_first is None:
                    t_first = time.perf_counter()
                if c:
                    content_parts.append(c)
                    chunks += 1
                if r:
                    reasoning_parts.append(r)
                    chunks += 1
                if logprobs and ch0.get("logprobs") and ch0["logprobs"].get("content"):
                    for item in ch0["logprobs"]["content"]:
                        if len(lp_samples) < 64:
                            lp_samples.append({
                                "token": item.get("token"),
                                "logprob": round(float(item.get("logprob", 0.0)), 6),
                            })

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ttft = t_first - t0
    total_s = t_end - t0
    decode_s = max(t_end - t_first, 1e-6)
    if completion_tokens == 0:
        completion_tokens = chunks
    decode_tok_s = ((completion_tokens - 1) / decode_s) if completion_tokens > 1 else (completion_tokens / max(total_s, 1e-6))
    prefill_tok_s = (prompt_tokens / ttft) if (prompt_tokens > 0 and ttft > 0) else 0.0
    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 4),
        "decode_s": round(decode_s, 4),
        "total_s": round(total_s, 4),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "decode_tok_s": round(decode_tok_s, 1),
        "wall_tok_s": round(completion_tokens / max(total_s, 1e-6), 1),
        "reasoning_len": len(full_reasoning),
        "content_len": len(full_content),
        "reasoning_text": full_reasoning,
        "content_text": full_content,
        "logprobs_head": lp_samples,
    }


def load_docs(doc_ids):
    out = {}
    with open(DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["doc_id"] in doc_ids:
                out[r["doc_id"]] = r["text_flat"]
    return out


def fetch_kv_metrics(base_url):
    metrics_url = base_url.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    try:
        txt = urllib.request.urlopen(f"{metrics_url}/metrics", timeout=10).read().decode()
        for line in txt.splitlines():
            if "vllm:cache_config_info" in line and not line.startswith("#"):
                return line
    except Exception as e:
        return f"ERR: {e}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next-dflash2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-nonce", default="fn-parity-v1", help="Fixed prefix for deterministic parity comparisons")
    args = ap.parse_args()

    # Use fixed nonce ONLY for parity prompts (unique per-prompt so no cross-prompt cache hit,
    # and run with a fresh server so 0% prefix hit on single-turn prompts!).
    p_nonce = args.fixed_nonce
    r_nonce = f"{args.tag}-{int(time.time())}"
    res = {"tag": args.tag, "url": args.url, "model": args.model, "ts": int(time.time())}

    print(f"=== [{args.tag}] Warmup ===", flush=True)
    stream_chat(args.url, args.model, [{"role": "user", "content": f"[{r_nonce}-w1] Say hello in 5 words."}], max_tokens=32, temperature=0.0)
    w2 = stream_chat(args.url, args.model, [{"role": "user", "content": f"[{r_nonce}-w2] Count 1 to 15."}], max_tokens=64, temperature=0.0)
    res["vram_idle"] = vram_info()
    res["kv_metrics_line"] = fetch_kv_metrics(args.url)
    print(f"Warmup OK | w2={w2['decode_tok_s']} tok/s | idle VRAM={res['vram_idle']}", flush=True)

    with PeakVRAMMonitor() as mon:
        print(f"\n=== [{args.tag}] 1. Single-user Decode (greedy temp=0.0, max_tokens=600) ===", flush=True)
        res["decode_1u_greedy"] = {}
        for name, prompt in DECODE_PROMPTS:
            want_lp = name in ("code", "prose")
            r = stream_chat(
                args.url, args.model,
                [{"role": "user", "content": f"[{p_nonce}-1u-{name}]\n{prompt}"}],
                max_tokens=600, temperature=0.0, logprobs=want_lp, top_logprobs=3 if want_lp else None,
            )
            res["decode_1u_greedy"][name] = r
            print(f"  {name:6s}: {r['decode_tok_s']:6.1f} tok/s | {r['completion_tokens']:4d} tok in {r['decode_s']:5.2f}s (ttft {r['ttft_s']*1000:5.1f}ms)", flush=True)

        print(f"\n=== [{args.tag}] 2. Single-user Realistic Sampling (temp=1.0, top_p=0.95, top_k=20) ===", flush=True)
        rs = stream_chat(
            args.url, args.model,
            [{"role": "user", "content": f"[{r_nonce}-samp]\n{DECODE_PROMPTS[1][1]}"}],
            max_tokens=600, temperature=1.0, top_p=0.95, top_k=20,
        )
        res["decode_1u_sampled"] = rs
        print(f"  code(sampled): {rs['decode_tok_s']:6.1f} tok/s | {rs['completion_tokens']} tok in {rs['decode_s']:.2f}s", flush=True)

        print(f"\n=== [{args.tag}] 3. 4-User Concurrent Decode (4x400 tok, temp=0.0) ===", flush=True)
        b4 = [([{"role": "user", "content": f"[{r_nonce}-4u-{i}]\n{DECODE_PROMPTS[i][1]}"}], 400) for i in range(4)]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            out4 = list(ex.map(lambda x: stream_chat(args.url, args.model, x[0], max_tokens=x[1], temperature=0.0), b4))
        w4 = time.perf_counter() - t0
        sum4 = sum(x["decode_tok_s"] for x in out4)
        tot4 = sum(x["completion_tokens"] for x in out4)
        res["decode_4u"] = {"wall_s": round(w4, 3), "total_tokens": tot4, "wall_agg_tok_s": round(tot4 / w4, 1), "sum_stream_tok_s": round(sum4, 1)}
        print(f"  4-user: sum_stream={sum4:6.1f} tok/s | wall_agg={res['decode_4u']['wall_agg_tok_s']:6.1f} tok/s ({tot4} tok in {w4:.2f}s)", flush=True)

        print(f"\n=== [{args.tag}] 4. Long-Context Prefill & Needle Parity (32.9k & 132.5k) ===", flush=True)
        docs = load_docs({"lde-00018", "lde-00029"})
        res["prefill"] = {}
        for did in ["lde-00018", "lde-00029"]:
            r = stream_chat(
                args.url, args.model,
                [{"role": "user", "content": f"[{p_nonce}-pf-{did}]\n{docs[did]}\n\nSummarise the key program IDs and main facts in two sentences."}],
                max_tokens=128, temperature=0.0,
            )
            res["prefill"][did] = r
            print(f"  {did}: pt={r['prompt_tokens']:7d} | ttft={r['ttft_s']:6.2f}s | pf={r['prefill_tok_s']:7.1f} tok/s | dec={r['decode_tok_s']:5.1f} tok/s", flush=True)

        print(f"\n=== [{args.tag}] 5. 4-Turn Conversational Multi-Turn TTFT (32.9k base + 3 follow-ups) ===", flush=True)
        mt_msgs = [
            {"role": "user", "content": f"[{r_nonce}-mt]\n{docs['lde-00018']}\n\nList all major program IDs and dates mentioned in the document in detail."}
        ]
        followups = [
            "Now compare the timelines of the programs you just listed in 4 bullet points.",
            "Which discrepancies or risks stand out most across those entries? Be thorough.",
            "Give a concise executive recommendation in 3 sentences based on our discussion.",
        ]
        res["multiturn"] = []
        for turn_idx in range(4):
            r = stream_chat(args.url, args.model, mt_msgs, max_tokens=512 if turn_idx < 3 else 128, temperature=0.0)
            res["multiturn"].append({
                "turn": turn_idx + 1,
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "completion_tokens": r["completion_tokens"],
                "ttft_s": r["ttft_s"],
                "decode_tok_s": r["decode_tok_s"],
            })
            print(f"  turn {turn_idx+1}: pt={r['prompt_tokens']:6d} cached={r['cached_tokens']:6d} | ttft={r['ttft_s']:6.3f}s | ct={r['completion_tokens']:4d} dec={r['decode_tok_s']:5.1f} tok/s", flush=True)
            asst_text = r["content_text"] or r["reasoning_text"]
            mt_msgs.append({"role": "assistant", "content": asst_text})
            if turn_idx < len(followups):
                mt_msgs.append({"role": "user", "content": followups[turn_idx]})

        print(f"\n=== [{args.tag}] 6. 6-User Concurrent Mixed Stress (2x 32.9k + 4x 400 decode) ===", flush=True)
        jobs6 = [
            ([{"role": "user", "content": f"[{r_nonce}-6u-0]\n{docs['lde-00018']}\n\nSummarise in one sentence."}], 64),
            ([{"role": "user", "content": f"[{r_nonce}-6u-1]\n{docs['lde-00018']}\n\nList 3 key facts."}], 64),
            ([{"role": "user", "content": f"[{r_nonce}-6u-2]\n{DECODE_PROMPTS[0][1]}"}], 400),
            ([{"role": "user", "content": f"[{r_nonce}-6u-3]\n{DECODE_PROMPTS[1][1]}"}], 400),
            ([{"role": "user", "content": f"[{r_nonce}-6u-4]\n{DECODE_PROMPTS[2][1]}"}], 400),
            ([{"role": "user", "content": f"[{r_nonce}-6u-5]\n{DECODE_PROMPTS[3][1]}"}], 400),
        ]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            out6 = list(ex.map(lambda x: stream_chat(args.url, args.model, x[0], max_tokens=x[1], temperature=0.0), jobs6))
        w6 = time.perf_counter() - t0
        res["stress_6u"] = {"wall_s": round(w6, 2), "ok": len(out6) == 6}
        print(f"  6-concurrent mixed OK in {w6:.2f}s", flush=True)

    res["vram_peak"] = {"peak_used_mib": mon.peak_used, "min_free_mib": mon.min_free, "spike_mib": mon.peak_used - res["vram_idle"]["used_mib"]}
    res["vram_final"] = vram_info()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} | VRAM peak={res['vram_peak']}", flush=True)


if __name__ == "__main__":
    main()
