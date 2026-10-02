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

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App, ResumabilityConfig

from app.harness.scope import RunScope
from app.harness.trace import TracePlugin
from app.pipeline import pipeline

# Each agent's role prompt is a static instruction: the same head on every request.
# Gemini caches it (and the turn so far) server-side; through LiteLLM, providers that
# cache a marked prefix get cache_control marks and the others ignore them; Ollama
# reuses its KV cache for an unchanged prefix on its own. Gemini 3 caches only past
# 4096 tokens, so the first calls of a turn are never cached.
CACHE = ContextCacheConfig(cache_intervals=20, ttl_seconds=1800, min_tokens=4096)

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
    context_cache_config=CACHE,
)
