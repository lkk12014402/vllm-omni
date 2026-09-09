# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

from vllm.renderers import BaseRenderer


class OmniRenderer:
    """Delegate tokenization/chat rendering while extending engine-input routing.

    Keep the original renderer's tokenizer, executors, and multimodal caches.
    The upstream pipeline methods run on this adapter so both sync and async
    entrypoints reach the Omni singleton processing below.
    """

    render_cmpl = BaseRenderer.render_cmpl
    render_cmpl_async = BaseRenderer.render_cmpl_async
    render_chat = BaseRenderer.render_chat
    render_chat_async = BaseRenderer.render_chat_async
    process_for_engine = BaseRenderer.process_for_engine
    process_for_engine_async = BaseRenderer.process_for_engine_async
    _process_enc_dec = BaseRenderer._process_enc_dec
    _process_enc_dec_async = BaseRenderer._process_enc_dec_async

    def __init__(self, renderer: BaseRenderer) -> None:
        self._renderer = renderer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)

    @staticmethod
    def _with_omni_extras(inputs, prompt):
        for key in ("prompt", "cache_salt", "additional_information", "model_intermediate_buffer"):
            if key in prompt:
                inputs[key] = prompt[key]
        return inputs

    def _process_singleton(self, prompt, *, skip_mm_cache: bool = False):
        if "prompt_embeds" not in prompt and "mm_processor_kwargs" in prompt and not prompt.get("multi_modal_data"):
            inputs = self._renderer._process_multimodal(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            inputs = self._renderer._process_singleton(prompt, skip_mm_cache=skip_mm_cache)
        return self._with_omni_extras(inputs, prompt)

    async def _process_singleton_async(self, prompt, *, skip_mm_cache: bool = False):
        if "prompt_embeds" not in prompt and "mm_processor_kwargs" in prompt and not prompt.get("multi_modal_data"):
            inputs = await self._renderer._process_multimodal_async(
                prompt["prompt_token_ids"],
                {},
                mm_processor_kwargs=prompt["mm_processor_kwargs"],
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            inputs = await self._renderer._process_singleton_async(prompt, skip_mm_cache=skip_mm_cache)
        return self._with_omni_extras(inputs, prompt)
