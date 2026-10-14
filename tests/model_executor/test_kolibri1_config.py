# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.config import StructuredOutputsConfig
from vllm.model_executor.models.config import Kolibri1ForCausalLMConfig

RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "content": {"close": "<|im_end|>", "content": "text"},
    },
}


def _reasoning_parser(tmp_path, tokenizer_config: dict, reasoning_parser: str):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            tokenizer=str(tmp_path),
            trust_remote_code=False,
            tokenizer_revision=None,
        ),
        structured_outputs_config=StructuredOutputsConfig(
            reasoning_parser=reasoning_parser
        ),
    )
    Kolibri1ForCausalLMConfig.verify_and_update_config(vllm_config)
    return vllm_config.structured_outputs_config.reasoning_parser


@pytest.mark.parametrize(
    ("tokenizer_config", "reasoning_parser", "expected"),
    [
        pytest.param({"response_template": RESPONSE_TEMPLATE}, "", "hf", id="default"),
        pytest.param({}, "", "", id="no-template"),
        pytest.param(
            {"response_template": RESPONSE_TEMPLATE}, "qwen3", "qwen3", id="explicit"
        ),
    ],
)
def test_reasoning_parser_defaults_to_hf_with_response_template(
    tmp_path, tokenizer_config, reasoning_parser, expected
):
    """Older Kolibri 1 checkpoints ship no response template; defaulting them
    to `hf` would fail at startup. An explicit `--reasoning-parser` wins."""
    assert _reasoning_parser(tmp_path, tokenizer_config, reasoning_parser) == expected
