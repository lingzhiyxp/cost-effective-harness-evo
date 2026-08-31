# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute per-trial LLM spend from the raw InMemoryTracer dump.

Neither harbor's own token accounting (harbor.agents.installed.nexau.NexAU.
populate_context_post_run, which reads chat-completions-style `prompt_tokens`/
`completion_tokens` keys) nor trace_converter.py's `calculated_total_cost`
(which expects a Langfuse-style `calculatedTotalCost` field the InMemoryTracer
never emits) actually work for this project: code_agent/evolve_agent both run
under api_type: openai_responses, whose usage payload uses `input_tokens`/
`output_tokens`/`input_tokens_details.cached_tokens` -- a different schema.
This module reads that schema directly off the raw span tree and prices it
against a small hardcoded table (USD per 1M tokens, from
https://developers.openai.com/api/docs/pricing).
"""

from __future__ import annotations

from typing import Any

# USD per 1,000,000 tokens. `cached_input` is what a cache-hit prompt token
# costs (input_tokens_details.cached_tokens); everything else in the input
# count is priced at the plain `input` rate.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.5-pro": {"input": 30.00, "cached_input": 30.00, "output": 180.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-5.4-pro": {"input": 30.00, "cached_input": 30.00, "output": 180.00},
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "cached_input": 21.00, "output": 168.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "cached_input": 15.00, "output": 120.00},
    "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-pro": {"input": 150.00, "cached_input": 150.00, "output": 600.00},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "o3-pro": {"input": 20.00, "cached_input": 20.00, "output": 80.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    # DeepSeek 官方 rate card 的挂牌价（2026-08 查得：cache-miss 0.14 /
    # cache-hit 0.0028 / output 0.28）。本项目经 Scale 的 LiteLLM 代理调用
    # azure_ai/DeepSeek-V4-Flash，而该代理对这个模型返回 usage.cost = null，
    # 即 litellm 侧没有配价，Scale 实际计费的费率无从读取；已有报告指出
    # Azure Foundry 上的 DeepSeek 计费显著高于 DeepSeek 自家挂牌价。
    # 因此这一行给出的美元数是按挂牌价折算的下界，不是账单金额，凡是引用它的
    # 地方都应同时给出 token 数。
    "deepseek-v4-flash": {"input": 0.14, "cached_input": 0.0028, "output": 0.28},
    # Anthropic 首方费率。与 OpenAI 系不同，Anthropic 的缓存写入单独计价（基础输入价
    # 的 1.25 倍），因此这一族多一个 cache_write 键；缺该键时按 1.25 倍输入价折算。
    "claude-haiku-4-5": {"input": 1.00, "cached_input": 0.10, "cache_write": 1.25, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "cached_input": 0.30, "cache_write": 3.75, "output": 15.00},
}


def _price_for_model(model: str | None) -> dict[str, float] | None:
    if not model:
        return None
    # Model strings often carry a provider/date suffix (e.g. "openai/gpt-5.4",
    # "gpt-5.4-2026-01-01") -- match the longest known key that's a substring.
    normalized = model.lower()
    matches = [k for k in MODEL_PRICING if k in normalized]
    if not matches:
        return None
    best = max(matches, key=len)
    return MODEL_PRICING[best]


def _iter_llm_spans(spans: list[dict[str, Any]]):
    """Depth-first walk of the raw InMemoryTracer dump (a list of root spans,
    each with a `children` list) yielding every span with type == "LLM"."""
    for span in spans:
        if not isinstance(span, dict):
            continue
        if span.get("type") == "LLM":
            yield span
        children = span.get("children") or []
        yield from _iter_llm_spans(children)


def usage_and_cost_from_raw_spans(
    raw_spans: list[dict[str, Any]], model: str | None,
) -> dict[str, Any]:
    """Sum token usage across every LLM span in a raw InMemoryTracer dump and
    price it. Returns token totals unconditionally; `cost_usd` is None if
    `model` isn't in MODEL_PRICING (so a silent $0.00 never gets reported for
    an untracked model)."""
    input_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    output_tokens = 0
    n_calls = 0

    for span in _iter_llm_spans(raw_spans):
        usage = ((span.get("outputs") or {}).get("usage")) or {}
        # 两种 usage 形状，字段含义不同，不能混算：
        #
        # openai_responses / chat-completions：input_tokens 是提示词的全部 token，
        #   缓存命中数在 input_tokens_details.cached_tokens 里，是 input_tokens 的
        #   子集，因此未命中部分 = input_tokens − cached_tokens。
        # anthropic messages：input_tokens 已经不含缓存部分，缓存读取与写入分别记在
        #   cache_read_input_tokens 与 cache_creation_input_tokens，三者互不重叠。
        #
        # 对外仍然按前一种口径返回 input_tokens（提示词全部 token），这样两族模型的
        # 数字可以并排比较；计价则各按各自的口径。
        out_tok = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        anthropic_read = usage.get("cache_read_input_tokens")
        anthropic_write = usage.get("cache_creation_input_tokens")
        if anthropic_read is not None or anthropic_write is not None:
            read_tok = anthropic_read or 0
            write_tok = anthropic_write or 0
            uncached = usage.get("input_tokens", 0) or 0
            in_tok = uncached + read_tok + write_tok
            cache_tok = read_tok
        else:
            in_tok = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            write_tok = 0
            cache_tok = (
                (usage.get("input_tokens_details") or {}).get("cached_tokens")
                or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                or usage.get("cache_read_tokens")
                or 0
            )
        input_tokens += in_tok
        output_tokens += out_tok
        cached_tokens += cache_tok
        cache_write_tokens += write_tok
        n_calls += 1

    price = _price_for_model(model)
    cost_usd = None
    if price is not None:
        # 缓存写入不在 cached_tokens 里，因此未命中部分要把它也扣掉
        uncached_input = max(input_tokens - cached_tokens - cache_write_tokens, 0)
        write_rate = price.get("cache_write", price["input"] * 1.25)
        cost_usd = (
            uncached_input * price["input"]
            + cached_tokens * price["cached_input"]
            + cache_write_tokens * write_rate
            + output_tokens * price["output"]
        ) / 1_000_000

    return {
        "model": model,
        "n_llm_calls": n_calls,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
