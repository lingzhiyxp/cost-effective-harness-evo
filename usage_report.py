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

"""Aggregate usage_tracer.UsageTracer's JSONL output into per-role token and cost.

Shared by evolve.py (which writes a cost summary at the end of a run, and folds
per-iteration cost into iteration_scores.yaml) and scripts/price_run.py (which
prices an experiment directory after the fact, and additionally understands the
older cost_summary.json and raw-tracer sources). Keeping the arithmetic here means
a run's own numbers and a post-hoc audit of the same run cannot disagree.

Two conventions that matter and are easy to get wrong:

- **Each record is priced with the model that served it.** agent_debugger runs on
  its own ADB_LLM_* config and can differ from the agents', so one
  experiment-wide model would misprice it.
- **A call whose record carries no usage is counted as unmeasured, never as
  free.** Streamed chat-completions omit the usage payload; an earlier version of
  the audit reported explore_agent at a confident $0.00 for five real calls
  because of exactly this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# pricing.py lives beside this module in this repo (it is under step_evolve_runner/
# in the sibling repo this was ported from).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricing  # noqa: E402

#: Where UsageTracer output lands. The first covers host-side roles written under
#: an experiment's or iteration's usage/ directory; the second covers code_agent,
#: whose file harbor copies back inside each trial's agent/ directory.
USAGE_GLOBS = ("**/usage/**/*.jsonl", "**/usage.jsonl")

#: Report order. Roles outside this list are still reported, appended after.
KNOWN_ROLES = ("code_agent", "evolve_agent", "explore_agent", "agent_debugger")


def new_bucket() -> dict:
    return {
        "n_llm_calls": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        # Anthropic 独有：缓存写入单独计价（基础输入价的 1.25 倍），既不在
        # cached_tokens 里也不该按未命中输入计价。OpenAI 系恒为 0。
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        # Kept apart because they mean different things: no usage payload in the
        # record at all, vs. usage we have on a model pricing.MODEL_PRICING has
        # never heard of.
        "n_no_usage": 0,
        "n_unpriced_model": 0,
        "models": set(),
    }


def add_usage(bucket: dict, usage: dict, model: str | None) -> None:
    """Fold one call's usage into a bucket, priced with its own model."""
    priced = pricing.usage_and_cost_from_raw_spans(
        [{"type": "LLM", "outputs": {"usage": usage}}], model
    )
    bucket["n_llm_calls"] += 1
    bucket["input_tokens"] += priced["input_tokens"]
    bucket["cached_tokens"] += priced["cached_tokens"]
    bucket["cache_write_tokens"] += priced.get("cache_write_tokens", 0)
    bucket["output_tokens"] += priced["output_tokens"]
    if priced["cost_usd"] is None:
        bucket["n_unpriced_model"] += 1
    else:
        bucket["cost_usd"] += priced["cost_usd"]
    if model:
        bucket["models"].add(model)


def iter_usage_files(root: Path) -> list[Path]:
    """Every UsageTracer file under `root`, de-duplicated: the two globs overlap
    for a code_agent log that happens to sit inside a usage/ directory."""
    seen: dict[Path, None] = {}
    for pattern in USAGE_GLOBS:
        for path in root.glob(pattern):
            seen.setdefault(path.resolve(), None)
    return list(seen)


def aggregate(root: Path, roles: dict[str, dict] | None = None) -> dict[str, dict]:
    """Group every usage record under `root` by role and price it.

    Pass `roles` to accumulate into an existing result (used to add an
    experiment-level role such as explore_agent to a per-iteration total).
    """
    roles = roles if roles is not None else {}
    for path in iter_usage_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run killed mid-write can leave a partial line
            bucket = roles.setdefault(record.get("role") or "unattributed", new_bucket())
            usage = record.get("usage")
            if isinstance(usage, dict):
                add_usage(bucket, usage, record.get("model"))
            else:
                bucket["n_llm_calls"] += 1
                bucket["n_no_usage"] += 1
    return roles


def total_usd(roles: dict[str, dict]) -> float:
    return sum(b["cost_usd"] for b in roles.values())


def n_unmeasured(roles: dict[str, dict]) -> int:
    return sum(b["n_no_usage"] + b["n_unpriced_model"] for b in roles.values())


def ordered_roles(roles: dict[str, dict]) -> list[str]:
    return [r for r in KNOWN_ROLES if r in roles] + sorted(
        r for r in roles if r not in KNOWN_ROLES
    )


def to_plain(roles: dict[str, dict]) -> dict[str, dict]:
    """JSON-serialisable view, in report order (sets become sorted lists)."""
    return {
        role: {
            **{k: v for k, v in roles[role].items() if k != "models"},
            "models": sorted(roles[role]["models"]),
        }
        for role in ordered_roles(roles)
    }


def summarize(root: Path) -> dict:
    """Per-role breakdown plus totals for everything under `root`."""
    roles = aggregate(root)
    return {
        "by_role": to_plain(roles),
        "total_usd": total_usd(roles),
        "n_unmeasured_calls": n_unmeasured(roles),
        # An unmeasured call means real spend we cannot price, so the total is a
        # floor rather than a figure.
        "is_lower_bound": n_unmeasured(roles) > 0,
    }
