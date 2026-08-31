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

"""harbor adapter that makes the classic AHE loop's code_agent spend measurable.

The problem: code_agent already traces its own token usage -- code_agent.yaml
registers nexau's InMemoryTracer, and every LLM span carries `outputs.usage`. But
the only thing that reaches the host is `nexau_in_memory_tracer.cleaned.json`,
written by the closed-source `nexau-harbor` CLI via
`extract_trace_data_from_inmemory_dump()`, which keeps id/timestamp/name/input/
output/latency and drops the usage payload. So the largest agent in an AHE run
was also the one whose cost could not be recovered afterwards.

This subclass changes nothing about how the agent runs -- `create_run_agent_commands`
is inherited verbatim, so the same `nexau-harbor run` command executes with the same
environment. It only adds, at setup time:

  1. `usage_tracer.py` uploaded next to the workspace, and
  2. a copy of `code_agent.yaml` with `usage_tracer:UsageTracer` appended to its
     `tracers:` list, uploaded over the one `NexAU.setup()` just pushed.

`tracers:` is a first-class nexau agent-yaml field (AgentConfigBuilder.build_tracers
accepts any BaseTracer import path plus params) and multiple entries fan out through
CompositeTracer, so the existing InMemoryTracer keeps working exactly as before and
the CLI keeps writing the cleaned trace it always did. The new tracer independently
appends one line per LLM call to /logs/agent/usage.jsonl, which harbor copies back
with the rest of the agent directory.

Why patch the yaml here instead of committing the `tracers:` line into
agents/code_agent_simple/code_agent.yaml:

  - **The workspace is the artifact AHE evolves.** Measurement apparatus does not
    belong in it: evolve_agent would see the entry, could rewrite or delete it, and
    it would show up in every harness diff and archived snapshot as though it were
    part of the harness under study.
  - **It keeps `--agent nexau` working.** A committed `tracers:` entry pointing at
    a module that only exists because some adapter uploaded it is a dangling
    reference -- exactly the failure that cost the previous AHE run an entire
    iteration (a middleware registered in code_agent.yaml importing a module that
    did not exist, which failed config loading before any agent started and scored
    all 40 tasks 0). Uploading the module and the line that references it in one
    place means they cannot disagree.

The patched yaml is also written into the trial's own log directory, so what
actually ran is recoverable from the run's output rather than only inferable.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from harbor.agents.installed.nexau import NexAU
from harbor.environments.base import BaseEnvironment

_USAGE_TRACER_PATH = Path(__file__).resolve().parent / "usage_tracer.py"

#: Sandbox path for the recorded usage. Under /logs/agent so harbor copies it out
#: with nexau.txt and the cleaned trace; no extra plumbing needed to retrieve it.
_SANDBOX_USAGE_LOG = "/logs/agent/usage.jsonl"

_TRACER_ENTRY = {
    "import": "usage_tracer:UsageTracer",
    "params": {"path": _SANDBOX_USAGE_LOG, "role": "code_agent"},
}


class TracedNexAU(NexAU):
    """Stock NexAU plus an on-disk record of every LLM call's token usage."""

    @staticmethod
    def name() -> str:
        # Cosmetic (harbor's own logs/results). Selected via --agent-import-path,
        # never harbor's --agent name registry, so it stays distinct from "nexau".
        return "traced-nexau"

    def _write_patched_config(self) -> Path:
        """Return a local copy of code_agent.yaml with the usage tracer registered.

        Appends rather than replaces: code_agent.yaml ships an InMemoryTracer, and
        the CLI's cleaned trace comes from it. Dropping it to make room here would
        trade one blind spot for another.
        """
        config_name = Path(self._sandbox_config_path).name
        source = self._config_dir / config_name
        config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}

        tracers = config.get("tracers")
        config["tracers"] = ([*tracers] if isinstance(tracers, list) else []) + [_TRACER_ENTRY]

        target = self.logs_dir / config_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return target

    async def setup(self, environment: BaseEnvironment) -> None:
        # Uploads the workspace to /nexau-workspace and runs install-nexau.sh.j2.
        await super().setup(environment)

        # Alongside the workspace, which is what nexau resolves `tracers:` imports
        # against -- the same path that makes `tools.shell_tools:run_shell_command`
        # in code_agent.yaml resolve.
        await environment.upload_file(
            source_path=_USAGE_TRACER_PATH,
            target_path=f"{self._nexau_workspace_path}/usage_tracer.py",
        )
        # Must land after upload_dir, since it overwrites what that just wrote.
        await environment.upload_file(
            source_path=self._write_patched_config(),
            target_path=self._sandbox_config_path,
        )
