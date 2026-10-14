# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.config import StructuredOutputsConfig
from vllm.model_executor.models.config import ResponseTemplateParsersConfig

THINKING = {"open": "<think>", "close": "</think>", "content": "text"}
TOOL_CALLS = {
    "open": "<tool_call>",
    "close": "</tool_call>",
    "repeats": True,
    "content": "json",
    "transform": {"type": "function", "function": "{content}"},
}
CONTENT = {"close": "<|im_end|>", "content": "text"}


def _template(**fields) -> dict:
    return {
        "response_template": {
            "defaults": {"role": "assistant"},
            "start_anchor": "<|im_start|>assistant\n",
            "fields": {**fields, "content": CONTENT},
        }
    }


FULL = _template(thinking=THINKING, tool_calls=TOOL_CALLS)


def _parsers(
    tmp_path,
    tokenizer_config: dict,
    reasoning_parser: str = "",
    tool_call_parser: str = "",
    skip_tokenizer_init: bool = False,
) -> tuple[str, str]:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            tokenizer=str(tmp_path),
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=False,
            tokenizer_revision=None,
        ),
        structured_outputs_config=StructuredOutputsConfig(
            reasoning_parser=reasoning_parser, tool_call_parser=tool_call_parser
        ),
    )
    ResponseTemplateParsersConfig.verify_and_update_config(vllm_config)
    config = vllm_config.structured_outputs_config
    return config.reasoning_parser, config.tool_call_parser


@pytest.mark.parametrize(
    ("tokenizer_config", "kwargs", "expected"),
    [
        pytest.param(FULL, {}, ("hf", ""), id="default"),
        pytest.param({}, {}, ("", ""), id="no-template"),
        pytest.param(_template(thinking=THINKING), {}, ("hf", ""), id="thinking-only"),
        pytest.param(_template(tool_calls=TOOL_CALLS), {}, ("", ""), id="tools-only"),
        pytest.param(
            FULL, {"reasoning_parser": "qwen3"}, ("qwen3", ""), id="explicit-reasoning"
        ),
        pytest.param(
            FULL, {"tool_call_parser": "hermes"}, ("", "hermes"), id="explicit-tool"
        ),
        pytest.param(FULL, {"reasoning_parser": "hf"}, ("hf", ""), id="explicit-hf"),
        pytest.param(
            FULL, {"skip_tokenizer_init": True}, ("", ""), id="skip-tokenizer-init"
        ),
    ],
)
def test_reasoning_parser_defaults_to_hf_with_response_template(
    tmp_path, tokenizer_config, kwargs, expected
):
    """`hf` fails at startup without a `thinking` field and cannot be combined
    with other parsers, so an explicit non-`hf` tool call parser suppresses the
    default."""
    assert _parsers(tmp_path, tokenizer_config, **kwargs) == expected
