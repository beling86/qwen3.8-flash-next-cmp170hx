#!/usr/bin/env python3
"""Patch vLLM's Qwen4Exp + hybrid Mamba KV cache for DFlash2 on SM80 (CMP 170HX).

1. Allow "dflash" in Qwen4ExpForConditionalGenerationConfig.verify_and_update_config.
2. Implement SupportsEagle3 on Qwen4ExpForCausalLM and Qwen4ExpForConditionalGeneration,
   capturing completed, zero-based target layer outputs (via Eagle3's +1 layer index
   convention used by get_eagle3_aux_layers_from_config) averaged over the hc_count
   hyper-connection streams, matching incoai/Qwen3.8-Flash-Next-DFlash2's training
   and SGLang capture patch.
3. Round QSAKeyStateCache ring capacity up by compress_ratio until it divides
   cache_config.block_size.
4. Guard pre_hc_hidden_states against None in GPUModelRunner when target defines
   get_mtp_target_hidden_states() but runs a non-MTP speculator.
5. Cap SlidingWindowSpec page budget to mamba_page_size_padded on hybrid Mamba
   models so wider draft SWA layers do not bloat all target pages, preferring a
   kernel-supported divisor of primary block_size (with a >= 50% floor).
6. Clear tp_replicated on MambaSpec at TP=1 so single-layer replicated conv layers
   (Qwen4Exp PLE) merge with GDN Mamba layers instead of claiming a separate group.
7. Align hybrid attn_block_size to 64 when DFlash is enabled so 4x-wider BF16
   DFlash2 SWA blocks divide primary block_size evenly at a multiple of 16
   (e.g. 3264 / 4 = 816, 0% SWA padding).
8. In _get_packed_kv_cache_groups, pack 1-layer Mamba spillover (36 GDN + 1 PLE =
   37 layers) into 3 groups (13+12+12) instead of 4 (10+9+9+9) when block width
   stays within 5% of anchor_bytes.
"""

from __future__ import annotations

import pathlib
import py_compile
import sys


def patch_file(path: pathlib.Path, transform) -> None:
    orig = path.read_text()
    updated = transform(orig)
    if updated == orig:
        raise RuntimeError(f"No changes applied to {path}")
    path.write_text(updated)
    py_compile.compile(str(path), doraise=True)
    print(f"Patched & verified syntax: {path}")


def main() -> None:
    root = pathlib.Path("/workspace/vllm/src/vllm")
    if not root.is_dir():
        raise SystemExit(f"vLLM source not found at {root}")

    # 1. Allow "dflash" in Qwen4Exp config verification
    def transform_config(s: str) -> str:
        old = (
            '        if spec_config is not None and spec_config.method not in {\n'
            '            "mtp",\n'
            '            "ngram",\n'
            '            "ngram_gpu",\n'
            '        }:'
        )
        new = (
            '        if spec_config is not None and spec_config.method not in {\n'
            '            "mtp",\n'
            '            "dflash",\n'
            '            "ngram",\n'
            '            "ngram_gpu",\n'
            '        }:'
        )
        if old not in s:
            raise RuntimeError("Could not find Qwen4Exp spec_config gate in config.py")
        return s.replace(old, new, 1)

    patch_file(root / "model_executor/models/config.py", transform_config)

    # 2. Add SupportsEagle3 + HC-averaged aux hidden state capture to Qwen4Exp
    def transform_model(s: str) -> str:
        # 2a. Import SupportsEagle3
        old_imp = (
            "from vllm.model_executor.models.interfaces import (\n"
            "    HasInnerState,\n"
            "    IsHybrid,\n"
            "    MixtureOfExperts,\n"
            "    MultiModalEmbeddings,\n"
            "    SupportsLoRA,\n"
            "    SupportsMRoPE,\n"
            "    SupportsPP,\n"
            "    _require_is_multimodal,\n"
            ")"
        )
        new_imp = (
            "from vllm.model_executor.models.interfaces import (\n"
            "    HasInnerState,\n"
            "    IsHybrid,\n"
            "    MixtureOfExperts,\n"
            "    MultiModalEmbeddings,\n"
            "    SupportsEagle3,\n"
            "    SupportsLoRA,\n"
            "    SupportsMRoPE,\n"
            "    SupportsPP,\n"
            "    _require_is_multimodal,\n"
            ")"
        )
        if old_imp not in s:
            raise RuntimeError("Could not find interfaces import in qwen4_exp/nvidia/model.py")
        s = s.replace(old_imp, new_imp, 1)

        # 2b. Initialize self.aux_hidden_state_layers on Qwen4ExpModel
        old_init = "        else:\n            self._mtp_hidden_buffer = None\n"
        new_init = (
            "        else:\n"
            "            self._mtp_hidden_buffer = None\n"
            "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
        )
        if old_init not in s:
            raise RuntimeError("Could not find _mtp_hidden_buffer init in Qwen4ExpModel")
        s = s.replace(old_init, new_init, 1)

        # 2c. Init aux_hidden_states list in Qwen4ExpModel.forward
        old_fwd_start = "        block_output = None\n        injection = None\n        last_layer = None\n"
        new_fwd_start = (
            "        block_output = None\n"
            "        injection = None\n"
            "        last_layer = None\n"
            "        aux_hidden_states: list[torch.Tensor] = []\n"
        )
        if old_fwd_start not in s:
            raise RuntimeError("Could not find forward loop init in Qwen4ExpModel")
        s = s.replace(old_fwd_start, new_fwd_start, 1)

        # 2d. Capture completed layer outputs averaged over hc_count streams
        old_deepstack_end = (
            "                block_output = None\n"
            "                injection = None\n"
            "                hidden_states = hidden_states + deepstack_embed\n"
        )
        new_deepstack_end = (
            "                block_output = None\n"
            "                injection = None\n"
            "                hidden_states = hidden_states + deepstack_embed\n"
            "            if (layer_idx + 1) in self.aux_hidden_state_layers:\n"
            "                if block_output is not None:\n"
            "                    hidden_states = layer.mlp_hyper_connection.combine(\n"
            "                        hidden_states, block_output, injection\n"
            "                    )\n"
            "                    block_output = None\n"
            "                    injection = None\n"
            "                aux_hidden_states.append(\n"
            "                    hidden_states.unflatten(\n"
            "                        -1, (self.config.hc_count, self.config.hidden_size)\n"
            "                    ).mean(dim=-2)\n"
            "                )\n"
        )
        if old_deepstack_end not in s:
            raise RuntimeError("Could not find deepstack block end in Qwen4ExpModel.forward")
        s = s.replace(old_deepstack_end, new_deepstack_end, 1)

        # 2e. Return (sample_hidden_states, aux_hidden_states) when capturing
        old_ret = "            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)\n        return sample_hidden_states\n"
        new_ret = (
            "            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)\n"
            "        if self.aux_hidden_state_layers:\n"
            "            return sample_hidden_states, aux_hidden_states\n"
            "        return sample_hidden_states\n"
        )
        if old_ret not in s:
            raise RuntimeError("Could not find return sample_hidden_states in Qwen4ExpModel.forward")
        s = s.replace(old_ret, new_ret, 1)

        # 2f. Inherit SupportsEagle3 on Qwen4ExpForCausalLM & add methods
        old_clm = (
            "class Qwen4ExpForCausalLM(\n"
            "    nn.Module,\n"
            "    HasInnerState,\n"
            "    SupportsLoRA,\n"
            "    SupportsMRoPE,\n"
            "    SupportsPP,\n"
            "    Qwen4ExpMixtureOfExperts,\n"
            "    IsHybrid,\n"
            "):"
        )
        new_clm = (
            "class Qwen4ExpForCausalLM(\n"
            "    nn.Module,\n"
            "    HasInnerState,\n"
            "    SupportsEagle3,\n"
            "    SupportsLoRA,\n"
            "    SupportsMRoPE,\n"
            "    SupportsPP,\n"
            "    Qwen4ExpMixtureOfExperts,\n"
            "    IsHybrid,\n"
            "):"
        )
        if old_clm not in s:
            raise RuntimeError("Could not find Qwen4ExpForCausalLM class header")
        s = s.replace(old_clm, new_clm, 1)

        old_clm_embed = "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n        return self.model.embed_input_ids(input_ids)\n"
        new_clm_embed = (
            "    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n"
            "        self.model.aux_hidden_state_layers = tuple(layers)\n\n"
            "    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:\n"
            "        num_layers = self.config.num_hidden_layers\n"
            "        return (2, num_layers // 2, num_layers - 3)\n\n"
            "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
            "        return self.model.embed_input_ids(input_ids)\n"
        )
        if old_clm_embed not in s:
            raise RuntimeError("Could not find Qwen4ExpForCausalLM.embed_input_ids")
        s = s.replace(old_clm_embed, new_clm_embed, 1)

        # 2g. Inherit SupportsEagle3 on Qwen4ExpForConditionalGeneration & delegate
        old_cg = (
            "class Qwen4ExpForConditionalGeneration(\n"
            "    Qwen3_5ForConditionalGeneration,\n"
            "    HasInnerState,\n"
            "    Qwen4ExpMixtureOfExperts,\n"
            "):"
        )
        new_cg = (
            "class Qwen4ExpForConditionalGeneration(\n"
            "    Qwen3_5ForConditionalGeneration,\n"
            "    HasInnerState,\n"
            "    SupportsEagle3,\n"
            "    Qwen4ExpMixtureOfExperts,\n"
            "):"
        )
        if old_cg not in s:
            raise RuntimeError("Could not find Qwen4ExpForConditionalGeneration class header")
        s = s.replace(old_cg, new_cg, 1)

        old_cg_mtp = (
            "    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:\n"
            "        return self.language_model.get_mtp_target_hidden_states()\n"
        )
        new_cg_mtp = (
            "    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n"
            "        self.language_model.set_aux_hidden_state_layers(layers)\n\n"
            "    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:\n"
            "        return self.language_model.get_eagle3_default_aux_hidden_state_layers()\n\n"
            "    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:\n"
            "        return self.language_model.get_mtp_target_hidden_states()\n"
        )
        if old_cg_mtp not in s:
            raise RuntimeError("Could not find Qwen4ExpForConditionalGeneration.get_mtp_target_hidden_states")
        return s.replace(old_cg_mtp, new_cg_mtp, 1)

    patch_file(root / "models/qwen4_exp/nvidia/model.py", transform_model)

    # 3. Round QSA ring capacity up to a multiple of compress_ratio that divides
    # cache_config.block_size (e.g. num_spec=7 -> span=11 -> 12 -> 16 dividing 3280).
    def transform_qsa_cache(s: str) -> str:
        old = (
            "        span = self.compress_ratio + vllm_config.num_speculative_tokens\n"
            "        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)\n"
            "        assert self.cache_config.block_size % capacity == 0, ("
        )
        new = (
            "        span = self.compress_ratio + vllm_config.num_speculative_tokens\n"
            "        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)\n"
            "        while (\n"
            "            self.cache_config.block_size % capacity != 0\n"
            "            and capacity < self.cache_config.block_size\n"
            "        ):\n"
            "            capacity += self.compress_ratio\n"
            "        assert self.cache_config.block_size % capacity == 0, ("
        )
        if old not in s:
            raise RuntimeError("Could not find QSA ring capacity block in qsa_cache.py")
        return s.replace(old, new, 1)

    patch_file(root / "models/qwen4_exp/common/qsa_cache.py", transform_qsa_cache)

    # 4. Guard pre_hc_hidden_states against None when target implements
    # get_mtp_target_hidden_states() (e.g. Qwen4Exp) but runs a non-MTP
    # speculator (DFlash2).
    def transform_model_runner(s: str) -> str:
        old1 = (
            '            if hasattr(self.model, "get_mtp_target_hidden_states"):\n'
            "                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()\n"
            "                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]"
        )
        new1 = (
            '            if hasattr(self.model, "get_mtp_target_hidden_states"):\n'
            "                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()\n"
            "                if pre_hc_hidden_states is not None:\n"
            "                    spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]"
        )
        old2 = (
            '            if hasattr(self.model, "get_mtp_target_hidden_states"):\n'
            "                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()\n"
            "                spec_hidden_states = pre_hc_hidden_states[: draft_hidden_states.size(0)]"
        )
        new2 = (
            '            if hasattr(self.model, "get_mtp_target_hidden_states"):\n'
            "                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()\n"
            "                if pre_hc_hidden_states is not None:\n"
            "                    spec_hidden_states = pre_hc_hidden_states[: draft_hidden_states.size(0)]"
        )
        if old1 not in s or old2 not in s:
            raise RuntimeError("Could not find get_mtp_target_hidden_states guards in model_runner.py")
        return s.replace(old1, new1, 1).replace(old2, new2, 1)

    patch_file(root / "v1/worker/gpu/model_runner.py", transform_model_runner)

    # 5. Prevent SlidingWindowSpec (e.g. DFlash2 draft SWA layers) from exceeding
    # a hybrid Mamba model's already-aligned mamba_page_size_padded and bloating
    # all target layers' page sizes (2x-4x). Also align sw_block_size to divide
    # primary block_size so scheduler_block_size LCM does not inflate.
    def transform_attention_spec(s: str) -> str:
        old = (
            "            page_budget = shared_page or sw_per_token * block_size\n"
            "            sw_block_size = _largest_kernel_block_within(\n"
            "                self.attn_backend, sw_per_token, page_budget, block_size\n"
            "            )\n"
        )
        new = (
            "            mamba_page = vllm_config.cache_config.mamba_page_size_padded\n"
            "            if (\n"
            "                shared_page is None\n"
            "                and mamba_page is not None\n"
            "                and sw_per_token * block_size > mamba_page\n"
            "            ):\n"
            "                shared_page = mamba_page\n"
            "            page_budget = shared_page or sw_per_token * block_size\n"
            "            sw_block_size = _largest_kernel_block_within(\n"
            "                self.attn_backend, sw_per_token, page_budget, block_size\n"
            "            )\n"
            "            if (\n"
            "                mamba_page is not None\n"
            "                and shared_page == mamba_page\n"
            "                and block_size % sw_block_size != 0\n"
            "            ):\n"
            "                sizes = self.attn_backend.get_supported_kernel_block_sizes()\n"
            "                for cand in range(sw_block_size, sw_block_size // 2 - 1, -1):\n"
            "                    if block_size % cand != 0:\n"
            "                        continue\n"
            "                    if any(\n"
            "                        (cand == sz if isinstance(sz, int) else cand % sz.base == 0)\n"
            "                        for sz in sizes\n"
            "                    ):\n"
            "                        sw_block_size = cand\n"
            "                        break\n"
        )
        if old not in s:
            raise RuntimeError("Could not find SlidingWindowSpec page_budget in attention.py")
        return s.replace(old, new, 1)

    patch_file(root / "model_executor/layers/attention/attention.py", transform_attention_spec)

    # 6. At TP=1, clear tp_replicated on MambaSpec so single-layer replicated
    # conv layers (e.g. Qwen4Exp PLE short_conv) pack into the existing GDN
    # Mamba groups instead of claiming a separate full-pool group (saving
    # (2 + num_spec_tokens) ~41 MiB pool blocks per active request).
    def transform_mamba_abstract(s: str) -> str:
        old = "            tp_replicated=self.is_kv_cache_tp_replicated,\n"
        new = (
            "            tp_replicated=(\n"
            "                self.is_kv_cache_tp_replicated\n"
            "                and vllm_config.parallel_config.tensor_parallel_size > 1\n"
            "            ),\n"
        )
        if old not in s:
            raise RuntimeError("Could not find tp_replicated in mamba/abstract.py")
        return s.replace(old, new, 1)

    patch_file(root / "model_executor/layers/mamba/abstract.py", transform_mamba_abstract)

    # 7. When DFlash is enabled on a hybrid Mamba model, align attn_block_size to
    # 64 (4 * 16) so 4x-wider BF16 DFlash SWA pages divide attn_block_size evenly
    # at a kernel-supported multiple of 16 (e.g. 3264 -> 816, 0% SWA padding).
    def transform_platform_interface(s: str) -> str:
        old = (
            "            indexer_align = cls._get_indexer_block_alignment(vllm_config)\n"
            "            if indexer_align:\n"
            "                attn_block_size = indexer_align * cdiv(attn_block_size, indexer_align)\n"
        )
        new = (
            "            indexer_align = cls._get_indexer_block_alignment(vllm_config)\n"
            "            if indexer_align:\n"
            "                attn_block_size = indexer_align * cdiv(attn_block_size, indexer_align)\n"
            "            if (\n"
            "                vllm_config.speculative_config is not None\n"
            "                and vllm_config.speculative_config.method == \"dflash\"\n"
            "            ):\n"
            "                dflash_align = max(kernel_block_alignment_size * 4, 64)\n"
            "                attn_block_size = dflash_align * cdiv(attn_block_size, dflash_align)\n"
        )
        if old not in s:
            raise RuntimeError("Could not find indexer_align in platforms/interface.py")
        return s.replace(old, new, 1)

    patch_file(root / "platforms/interface.py", transform_platform_interface)

    # 8. In _get_packed_kv_cache_groups, avoid spawning an extra Mamba group for
    # a 1-layer spillover (e.g. 36 GDN + 1 PLE = 37 layers at 12 states/block)
    # when packing into 1 fewer group (13+12+12) widens the block by <= 5%.
    def transform_kv_cache_packing(s: str) -> str:
        old = (
            "        if anchor_bytes and is_state_bucket(spec):\n"
            "            states_per_block = max(anchor_bytes // spec.first_spec.page_size_bytes, 1)\n"
            "            num_groups = max(\n"
            "                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)\n"
            "            )\n"
        )
        new = (
            "        if anchor_bytes and is_state_bucket(spec):\n"
            "            page_sz = spec.first_spec.page_size_bytes\n"
            "            states_per_block = max(anchor_bytes // page_sz, 1)\n"
            "            n_layers = len(spec.kv_cache_specs)\n"
            "            split_groups = cdiv(n_layers, states_per_block)\n"
            "            if (\n"
            "                split_groups > 1\n"
            "                and isinstance(spec.first_spec, MambaSpec)\n"
            "                and cdiv(n_layers, split_groups - 1) * page_sz\n"
            "                <= int(anchor_bytes * 1.05)\n"
            "            ):\n"
            "                split_groups -= 1\n"
            "            num_groups = max(num_groups, split_groups)\n"
        )
        if old not in s:
            raise RuntimeError("Could not find is_state_bucket split in kv_cache_utils.py")
        return s.replace(old, new, 1)

    patch_file(root / "v1/core/kv_cache_utils.py", transform_kv_cache_packing)

    print("All Qwen4Exp DFlash2 patches applied successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
