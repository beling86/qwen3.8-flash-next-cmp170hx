#!/usr/bin/env python3
"""Compare candidate benchmark JSON against baseline JSON for parity & A/B speedup."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()

    a = json.loads(Path(args.baseline).read_text())
    b = json.loads(Path(args.candidate).read_text())

    print(f"=== A/B COMPARISON: {a['tag']} vs {b['tag']} ===")
    for k in ("list", "code", "prose", "chat"):
        ra, rb = a["decode_1u_greedy"][k], b["decode_1u_greedy"][k]
        pct = 100.0 * (rb["decode_tok_s"] - ra["decode_tok_s"]) / max(ra["decode_tok_s"], 1e-6)
        ta = (ra["reasoning_text"] + ra["content_text"])[:300]
        tb = (rb["reasoning_text"] + rb["content_text"])[:300]
        eq = "EXACT" if ta == tb else f"DIFF@{next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y), min(len(ta),len(tb)))}"
        lp_note = ""
        if ra.get("logprobs_head") and rb.get("logprobs_head"):
            pairs = list(zip(ra["logprobs_head"], rb["logprobs_head"]))
            tok_match = sum(1 for x, y in pairs if x["token"] == y["token"])
            diffs = [abs(x["logprob"] - y["logprob"]) for x, y in pairs if x["token"] == y["token"]]
            max_lp = max(diffs) if diffs else 0.0
            lp_note = f" | top1_match={tok_match}/{len(pairs)} max_dLP={max_lp:.4f}"
        print(f"  1u {k:6s}: {ra['decode_tok_s']:6.1f} -> {rb['decode_tok_s']:6.1f} tok/s ({pct:+5.1f}%) | text300={eq}{lp_note}")

    sa, sb = a["decode_1u_sampled"]["decode_tok_s"], b["decode_1u_sampled"]["decode_tok_s"]
    print(f"  1u samp  : {sa:6.1f} -> {sb:6.1f} tok/s ({100*(sb-sa)/max(sa,1e-6):+5.1f}%)")

    d4a, d4b = a["decode_4u"], b["decode_4u"]
    print(f"  4u sum   : {d4a['sum_stream_tok_s']:6.1f} -> {d4b['sum_stream_tok_s']:6.1f} tok/s ({100*(d4b['sum_stream_tok_s']-d4a['sum_stream_tok_s'])/max(d4a['sum_stream_tok_s'],1e-6):+5.1f}%) | wall {d4a['wall_agg_tok_s']} -> {d4b['wall_agg_tok_s']}")

    for did in ("lde-00018", "lde-00029"):
        pa, pb = a["prefill"][did], b["prefill"][did]
        dp = 100.0 * (pb["prefill_tok_s"] - pa["prefill_tok_s"]) / max(pa["prefill_tok_s"], 1e-6)
        dd = 100.0 * (pb["decode_tok_s"] - pa["decode_tok_s"]) / max(pa["decode_tok_s"], 1e-6)
        print(f"  pf {did}: {pa['prefill_tok_s']:7.1f} -> {pb['prefill_tok_s']:7.1f} ({dp:+5.1f}%) | dec {pa['decode_tok_s']:5.1f} -> {pb['decode_tok_s']:5.1f} ({dd:+5.1f}%)")

    print("  Multi-turn TTFT:")
    for ta, tb in zip(a["multiturn"], b["multiturn"]):
        print(f"    T{ta['turn']}: {ta['ttft_s']:6.3f}s (cached={ta['cached_tokens']}) -> {tb['ttft_s']:6.3f}s (cached={tb['cached_tokens']})")

    print(f"  Peak VRAM free: {a['vram_peak']['min_free_mib']} MiB -> {b['vram_peak']['min_free_mib']} MiB (spike {a['vram_peak']['spike_mib']} -> {b['vram_peak']['spike_mib']} MiB)")


if __name__ == "__main__":
    main()
