# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.apps import App, ResumabilityConfig

from app.harness.scope import RunScope
from app.harness.trace import TracePlugin
from app.pipeline import pipeline

# Prompt caching: each role's prompt and tools are cached once per role, in the
# background, by app/harness/prompt_cache.py (a model callback on every agent).
# ADK's ContextCacheConfig is not used: it creates the cache inside the request,
# so the agent waits for it.

# The pipeline Workflow is the deployed agent. Its name must match
# agents-cli-manifest.yaml (root_agent_name), which telemetry reports as gen_ai.agent.name.
root_agent = pipeline

app = App(
    root_agent=root_agent,
    name="app",
    # Lets the pipeline pause at a human review and resume from the answer.
    resumability_config=ResumabilityConfig(is_resumable=True),
    # RunScope first: every step binds its session's run folder before it is traced.
    # The trace logs every agent, tool call and file written to the run's logs/.
    plugins=[RunScope(), TracePlugin()],
)
