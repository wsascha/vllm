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
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {"close": "<|im_end|>", "content": "text"},
    },
}


def _parsers(
    tmp_path, tokenizer_config: dict, reasoning_parser: str, tool_call_parser: str
) -> tuple[str, str]:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            tokenizer=str(tmp_path),
            trust_remote_code=False,
            tokenizer_revision=None,
        ),
        structured_outputs_config=StructuredOutputsConfig(
            reasoning_parser=reasoning_parser, tool_call_parser=tool_call_parser
        ),
    )
    Kolibri1ForCausalLMConfig.verify_and_update_config(vllm_config)
    config = vllm_config.structured_outputs_config
    return config.reasoning_parser, config.tool_call_parser


@pytest.mark.parametrize(
    ("tokenizer_config", "flags", "expected"),
    [
        pytest.param(
            {"response_template": RESPONSE_TEMPLATE},
            ("", ""),
            ("hf", "hf"),
            id="default",
        ),
        pytest.param({}, ("", ""), ("", ""), id="no-template"),
        pytest.param(
            {"response_template": RESPONSE_TEMPLATE},
            ("qwen3", ""),
            ("qwen3", ""),
            id="explicit-reasoning",
        ),
        pytest.param(
            {"response_template": RESPONSE_TEMPLATE},
            ("", "hermes"),
            ("", "hermes"),
            id="explicit-tool",
        ),
        pytest.param(
            {"response_template": RESPONSE_TEMPLATE},
            ("hf", ""),
            ("hf", "hf"),
            id="explicit-hf",
        ),
    ],
)
def test_parsers_default_to_hf_with_response_template(
    tmp_path, tokenizer_config, flags, expected
):
    """Older Kolibri 1 checkpoints ship no response template, where `hf` would
    fail at startup. `hf` cannot be combined with other parsers, so an explicit
    non-`hf` parser suppresses both defaults."""
    assert _parsers(tmp_path, tokenizer_config, *flags) == expected
