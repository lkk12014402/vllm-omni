# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Omni routing through the active upstream rendering pipeline."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.inputs import tokens_input
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.inputs.preprocess import OmniRenderer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Renderer(BaseRenderer):
    def __init__(self):
        self.model_config = SimpleNamespace(is_encoder_decoder=False, enable_prompt_embeds=True)
        self.tokenizer = None
        self.default_cmpl_tok_params = TokenizeParams(max_total_tokens=128)
        self._tokenize_prompt = lambda prompt, params: {**prompt, "prompt_token_ids": [1, 2, 3]}
        self._tokenize_prompt_async = AsyncMock(side_effect=self._tokenize_prompt)
        self._process_multimodal = Mock(return_value=tokens_input([1, 2, 3, 99]))
        self._process_multimodal_async = AsyncMock(side_effect=self._process_multimodal)

    def render_messages(self, messages, params):
        return messages, {"prompt": "hello"}


@pytest.fixture
def renderer():
    return OmniRenderer(_Renderer())


@pytest.mark.parametrize("kwargs", [{}, {"target_h": 512, "target_w": 768}])
@pytest.mark.parametrize("tokenized", [False, True])
def test_process_inputs_routes_no_media_processor_kwargs(renderer, kwargs, tokenized):
    # Keep real process_inputs/render_cmpl; stub unrelated config validation.
    processor = object.__new__(InputProcessor)
    processor.renderer = renderer
    processor.model_config = renderer.model_config
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )
    processor._validate_params = Mock()
    processor._validate_lora = Mock()
    processor._validate_model_inputs = Mock()
    prompt = {"prompt_token_ids": [1, 2, 3]} if tokenized else {"prompt": "hello"}
    prompt.update(mm_processor_kwargs=kwargs, cache_salt="salt")
    request = processor.process_inputs("image-request", prompt, PoolingParams(), ("embed",))
    assert request.prompt_token_ids == [1, 2, 3, 99]
    assert request.cache_salt == "salt"
    renderer._renderer._process_multimodal.assert_called_once_with(
        [1, 2, 3],
        {},
        mm_processor_kwargs=kwargs,
        mm_uuids=None,
        skip_mm_cache=False,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["text", "tokens", "media", "kwargs", "embeds"])
def test_renderer_preserves_routes_extras_and_cache_policy(renderer, async_mode, kind):
    prompt = {"prompt": "hello", "additional_information": {"speaker": 1}, "model_intermediate_buffer": {"x": 2}}
    if kind == "tokens":
        prompt["prompt_token_ids"] = [1, 2, 3]
    elif kind == "media":
        prompt["multi_modal_data"] = {"image": "image"}
    elif kind == "kwargs":
        prompt["mm_processor_kwargs"] = {}
    elif kind == "embeds":
        prompt["prompt_embeds"] = torch.ones(3, 4)
    if async_mode:
        (result,) = asyncio.run(renderer.render_cmpl_async([prompt], skip_mm_cache=True))
    else:
        (result,) = renderer.render_cmpl([prompt], skip_mm_cache=True)
    assert result["additional_information"] == {"speaker": 1}
    assert result["model_intermediate_buffer"] == {"x": 2}
    assert result["prompt"] == "hello"
    if kind in ("media", "kwargs"):
        assert result["prompt_token_ids"] == [1, 2, 3, 99]
        assert renderer._renderer._process_multimodal.call_args.kwargs["skip_mm_cache"] is True
    else:
        renderer._renderer._process_multimodal.assert_not_called()
        if kind == "embeds":
            torch.testing.assert_close(result["prompt_embeds"], prompt["prompt_embeds"])
        else:
            assert result["prompt_token_ids"] == [1, 2, 3]
