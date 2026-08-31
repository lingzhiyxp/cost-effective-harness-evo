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

"""A nexau tracer that appends every LLM call's token usage to a JSONL file.

Why this exists: nexau already captures usage -- InMemoryTracer holds it on each
LLM span's `outputs.usage` -- but nothing in the AHE path gets it onto disk. The
stock closed-source `nexau-harbor` CLI writes only
`nexau_in_memory_tracer.cleaned.json`, and the cleaner
(`extract_trace_data_from_inmemory_dump`) keeps id/timestamp/name/input/output/
latency while dropping the usage payload. `adb ask` declares `tracers: []` and
records nothing at all. So agent spend was unmeasurable for three of the four
roles in an AHE run.

Registering this alongside whatever tracer an agent already has closes that gap
without touching any run loop: `tracers:` is a first-class nexau agent-yaml field
(AgentConfigBuilder.build_tracers accepts any BaseTracer import path plus params),
and multiple entries are fanned out through CompositeTracer.

Design notes:

- **One line per LLM call, written immediately.** Not buffered until `flush()`:
  an E2B sandbox killed by its timeout would lose a buffer, and losing the tail of
  an expensive run is exactly when the number matters. Appending is crash-safe.
- **Calls with no usage are still recorded, with `usage: null`.** A streaming
  chat-completions call whose provider omits the usage chunk must show up as an
  unpriceable call, not vanish -- a missing row reads as "no spend" downstream,
  which is the failure mode this whole exercise exists to remove.
- **No pricing here.** This records what was used; `step_evolve_runner/pricing.py`
  stays the single place that knows what things cost.
- **Never raises into the agent.** Any tracer error disables further writes and
  leaves the run alone; an accounting side-channel must not be able to fail a task.

`agent_debugger_core/runtime/usage_tracer.py` is a copy of this file. ADB runs as
its own console script and resolves imports against its installed package, so it
cannot import this module from the repo root. Keep the two in sync.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexau.archs.tracer.core import BaseTracer, Span, SpanType

#: Overrides the `path` param when set. Host-side agents (evolve_agent,
#: explore_agent) are launched in-process by evolve.py, which knows the
#: per-iteration output directory only at runtime and sets this; sandbox-side
#: code_agent has no such launcher and passes an absolute `path` param in its
#: yaml instead.
PATH_ENV_VAR = "NEXAU_USAGE_LOG"


class UsageTracer(BaseTracer):
    """Append `{ts, role, span, model, usage}` to a JSONL file per LLM span.

    Args:
        path: Output file. Ignored when PATH_ENV_VAR is set. Relative paths are
            resolved by nexau against the declaring yaml's directory, so prefer
            an absolute path.
        role: Free-form label recorded on every line (``code_agent``,
            ``evolve_agent``, ...). Lets one file hold several roles, and lets a
            reader attribute spend without inferring it from the file's location.
    """

    def __init__(self, path: str | None = None, role: str | None = None) -> None:
        target = os.environ.get(PATH_ENV_VAR) or path or "nexau_usage.jsonl"
        self._path = Path(target)
        self._role = role
        # Keyed by the span id this tracer hands back as `vendor_obj`, because
        # that is the only identity that survives CompositeTracer: it calls
        # end_span with a wrapper Span carrying id/name/type/vendor_obj and *not*
        # inputs, so the model name has to be stashed at start_span time. (It
        # also skips any tracer whose vendor_obj came back None -- hence setting
        # it unconditionally below.)
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._disabled = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._fail(f"cannot create {self._path.parent}: {exc}")

    # -- BaseTracer -------------------------------------------------------

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        span_id = str(uuid.uuid4())
        if span_type == SpanType.LLM and not self._disabled:
            model = None
            if isinstance(inputs, dict):
                raw_model = inputs.get("model")
                model = raw_model if isinstance(raw_model, str) else None
            with self._lock:
                self._pending[span_id] = {"name": name, "model": model}

        return Span(
            id=span_id,
            name=name,
            type=span_type,
            inputs=inputs or {},
            attributes=attributes or {},
            vendor_obj=span_id,
        )

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self._disabled or span.type != SpanType.LLM:
            return

        key = str(span.vendor_obj) if span.vendor_obj is not None else span.id
        with self._lock:
            started = self._pending.pop(key, None)

        usage = None
        model = started.get("model") if started else None
        if isinstance(outputs, dict):
            raw_usage = outputs.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
            # openai_responses puts the model on the response; chat-completions
            # streaming has the aggregator attach it. Either beats the request.
            raw_model = outputs.get("model")
            if isinstance(raw_model, str):
                model = raw_model

        self._write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "role": self._role,
                "span": (started or {}).get("name") or span.name,
                "model": model,
                "usage": usage,
                "error": str(error) if error is not None else None,
            }
        )

    def flush(self) -> None:
        return  # every record is already on disk

    def shutdown(self) -> None:
        return

    # -- internals --------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - must never reach the agent
            self._fail(f"write to {self._path} failed: {exc}")

    def _fail(self, message: str) -> None:
        """Disable recording and say so once. Losing the accounting side-channel
        is worth reporting, but never worth failing the run it is measuring."""
        if not self._disabled:
            self._disabled = True
            sys.stderr.write(f"[usage-tracer] disabled -- {message}\n")
