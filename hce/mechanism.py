"""What actually happened when a task ran, reduced to predicate-checkable features.

This is the empirical stand-in for the static dependency graph that regression
test selection uses in ordinary software: an LLM agent's behaviour cannot be
derived from a code diff, so a change's blast radius is estimated from what the
tasks were observed to do.

FEATURE_SCHEMA is the single source of truth. It gates predicate validation --
the evolve agent may only condition on keys that appear here -- and it is what
gets rendered into the evolve prompt as the reference table. A key the engine
cannot evaluate therefore cannot be written into a contract in the first place.

Note on `extract_agent_behavior_stats` in evolve.py, which this replaces: it
reads the trace as `spans[0]`, i.e. a list. The real cleaned trace is a dict, so
that subscript raises KeyError, which its own `except` swallows -- it has always
returned empty for every task. The output shape was worth keeping; the traversal
was not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pricing

# Terminal-Bench truncation marker, from
# agents/code_agent_simple/tools/shell_tools/run_shell_command.py:47,51
_TRUNCATION_MARKER = re.compile(r"^Output too large\. Showing the last ", re.M)

_ERROR_PATTERNS = {
    "has_python_traceback": re.compile(r"^Traceback \(most recent call last\):", re.M),
    "has_command_not_found": re.compile(r"(?m)^(?:bash: )?(?:line \d+: )?.*: command not found"),
    "has_permission_error": re.compile(r"Permission denied|PermissionError|EACCES"),
    "has_oom": re.compile(r"MemoryError|Out of memory|\bKilled\b"),
    "has_network_error": re.compile(
        r"Could not resolve host|Connection refused|Temporary failure in name resolution"),
}

# type, reducer over k rollouts, and the one-line description shown to the agent.
FEATURE_SCHEMA: dict[str, dict] = {
    # (a) output scale and truncation
    "max_command_output_tokens": {"type": "int", "reducer": "max",
        "desc": "Largest single command observation, in estimated tokens."},
    "total_output_tokens": {"type": "int", "reducer": "mean",
        "desc": "Sum of all command observations in one rollout, estimated tokens."},
    "truncation_triggered": {"type": "bool", "reducer": "any",
        "desc": "The harness truncated at least one observation."},
    # (b) trajectory shape
    "n_steps": {"type": "int", "reducer": "mean",
        "desc": "LLM generations in the rollout."},
    "n_tool_calls": {"type": "int", "reducer": "mean",
        "desc": "Tool invocations in the rollout."},
    "hit_step_limit": {"type": "bool", "reducer": "any",
        "desc": "The rollout reached the harness's max_iterations."},
    "timed_out": {"type": "bool", "reducer": "any",
        "desc": "The rollout hit the task's agent timeout."},
    # (c) error features
    "has_python_traceback": {"type": "bool", "reducer": "any",
        "desc": "A Python traceback appeared in a command observation."},
    "has_command_not_found": {"type": "bool", "reducer": "any",
        "desc": "A shell command was not found."},
    "has_permission_error": {"type": "bool", "reducer": "any",
        "desc": "A permission error appeared."},
    "has_oom": {"type": "bool", "reducer": "any",
        "desc": "An out-of-memory or kill signal appeared."},
    "has_network_error": {"type": "bool", "reducer": "any",
        "desc": "A DNS or connection error appeared."},
    "has_nonzero_exit": {"type": "bool", "reducer": "any",
        "desc": "At least one command exited nonzero."},
    "n_nonzero_exit": {"type": "int", "reducer": "mean",
        "desc": "Count of commands exiting nonzero."},
    # (d) context and cost
    "peak_context_tokens": {"type": "int", "reducer": "max",
        "desc": "Largest prompt sent in the rollout, in tokens."},
    "near_context_limit": {"type": "bool", "reducer": "any",
        "desc": "Peak prompt reached 90% of the harness context limit."},
    "rollout_tokens": {"type": "int", "reducer": "mean",
        "desc": "Total tokens consumed by the rollout."},
    "rollout_usd": {"type": "float", "reducer": "mean",
        "desc": "Dollar cost of the rollout."},
}

TOKEN_ESTIMATOR = "chars_div_4"


def est_tokens(text: str) -> int:
    """Token estimate for an observation.

    The agent writes thresholds against this number, so it has to be stable and
    recorded: `token_estimator` is stamped into every profile. Four characters
    per token is the usual English approximation and needs no tokenizer on the
    critical path.
    """
    return len(text) // 4


def _reduce(key: str, values: list) -> object:
    present = [v for v in values if v is not None]
    if not present:
        return None
    reducer = FEATURE_SCHEMA[key]["reducer"]
    if reducer == "max":
        return max(present)
    if reducer == "any":
        return any(present)
    total = sum(present)
    return round(total / len(present), 4) if FEATURE_SCHEMA[key]["type"] == "float" \
        else int(round(total / len(present)))


def extract_rollout(trial_dir: Path, *, max_iterations: int | None = None,
                    max_context_tokens: int | None = None,
                    agent_timeout_sec: float | None = None) -> dict:
    """Mechanism features for one rollout directory."""
    feat: dict[str, object] = {k: None for k in FEATURE_SCHEMA}

    trace_src = trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json"
    observations: list[str] = []
    if trace_src.exists():
        try:
            trace = json.loads(trace_src.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            trace = {}
        if isinstance(trace, dict):
            calls = [c for m in (trace.get("messages") or [])
                     if isinstance(m, dict)
                     for c in (m.get("tool_calls") or [])
                     if isinstance(c, dict)]
            sizes, exits = [], []
            for call in calls:
                result = ((call.get("output") or {}).get("result") or {})
                content = result.get("content")
                if isinstance(content, str):
                    observations.append(content)
                    sizes.append(est_tokens(content))
                code = result.get("exit_code")
                if isinstance(code, int):
                    exits.append(code)
            feat["n_tool_calls"] = len(calls)
            feat["max_command_output_tokens"] = max(sizes) if sizes else 0
            feat["total_output_tokens"] = sum(sizes)
            feat["n_nonzero_exit"] = sum(1 for c in exits if c != 0)
            feat["has_nonzero_exit"] = feat["n_nonzero_exit"] > 0
            steps = trace.get("generation_count")
            if isinstance(steps, int):
                feat["n_steps"] = steps
                if max_iterations:
                    feat["hit_step_limit"] = steps >= max_iterations

    blob = "\n".join(observations)
    for key, pattern in _ERROR_PATTERNS.items():
        feat[key] = bool(pattern.search(blob))
    feat["truncation_triggered"] = bool(_TRUNCATION_MARKER.search(blob))

    # Peak context comes from usage.jsonl, never from the trace: the trace's
    # total_tokens is a running sum over the whole rollout, not a maximum.
    usage_src = trial_dir / "agent" / "usage.jsonl"
    if usage_src.exists():
        inputs, tokens, n_records, usd, priced_any = [], 0, 0, 0.0, False
        for line in usage_src.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line) or {}
            except json.JSONDecodeError:
                continue
            usage = record.get("usage") or {}
            n_records += 1
            got_in = usage.get("input_tokens")
            if isinstance(got_in, int):
                inputs.append(got_in)
                tokens += got_in
            got_out = usage.get("output_tokens")
            if isinstance(got_out, int):
                tokens += got_out
            # Price with the model that served this call, as usage_report does:
            # roles can and do run on different models within one iteration.
            if usage:
                priced = pricing.usage_and_cost_from_raw_spans(
                    [{"type": "LLM", "outputs": {"usage": usage}}], record.get("model"))
                if priced["cost_usd"] is not None:
                    usd += priced["cost_usd"]
                    priced_any = True
        if inputs:
            feat["peak_context_tokens"] = max(inputs)
            if max_context_tokens:
                feat["near_context_limit"] = max(inputs) >= 0.9 * max_context_tokens
        feat["rollout_tokens"] = tokens or None
        # Left as None, never 0.0, when nothing could be priced -- a confident
        # zero is the failure mode the cost audit exists to remove.
        feat["rollout_usd"] = round(usd, 6) if priced_any else None
        if feat["n_steps"] is None and n_records:
            feat["n_steps"] = n_records

    result_src = trial_dir / "result.json"
    if result_src.exists():
        try:
            result = json.loads(result_src.read_text(encoding="utf-8", errors="replace")) or {}
        except json.JSONDecodeError:
            result = {}
        exc = (result.get("exception_info") or {}).get("exception_type") or ""
        timed_out = exc == "AgentTimeoutError"
        if not timed_out and agent_timeout_sec:
            block = result.get("agent_execution") or {}
            for key in ("duration_sec", "duration_s", "elapsed_sec"):
                got = block.get(key)
                if isinstance(got, (int, float)):
                    timed_out = got >= 0.95 * agent_timeout_sec
                    break
        feat["timed_out"] = timed_out

    return feat


def aggregate(per_rollout: list[dict]) -> dict:
    """Fold k rollouts into one profile.

    Maxima and ORs, not means, for the discriminating keys: a predicate asks
    whether this task *can* trigger the mechanism, so one rollout reaching 12k
    tokens makes the task a candidate even if the other two stayed small.
    """
    return {key: _reduce(key, [r.get(key) for r in per_rollout]) for key in FEATURE_SCHEMA}


def schema_reference_table(observed: dict[str, dict] | None = None) -> str:
    """Markdown table of legal predicate keys, injected into the evolve prompt.

    `observed` is the aggregated profile of every task, and passing it is close
    to mandatory. Given key names alone a model writes thresholds from intuition
    -- the proposal's own worked example is `max_command_output_tokens > 8000`,
    which on the measured Terminal-Bench profiles matches zero of forty-five
    tasks, because the largest observation there is 5,026. An empty activation
    set fails falsification, so every iteration would be rejected before it ran.
    Showing the realised quartiles lets the agent pick a threshold that actually
    partitions the task set, and marks the keys that cannot discriminate at all.
    """
    rows = ["| key | type | observed across tasks | description |",
            "| --- | --- | --- | --- |"]
    for key, meta in FEATURE_SCHEMA.items():
        seen = ""
        if observed:
            values = [t.get(key) for t in observed.values() if t.get(key) is not None]
            if not values:
                seen = "never recorded -- do not use"
            elif meta["type"] == "bool":
                n = sum(1 for v in values if v)
                seen = (f"true for {n}/{len(values)} tasks" if n
                        else "false for every task -- do not use")
            else:
                ordered = sorted(values)
                q = lambda f: ordered[min(len(ordered) - 1, int(len(ordered) * f))]
                seen = (f"p25={q(0.25):,} p50={q(0.50):,} p75={q(0.75):,} "
                        f"max={ordered[-1]:,}")
        rows.append(f"| `{key}` | {meta['type']} | {seen} | {meta['desc']} |")
    return "\n".join(rows)
