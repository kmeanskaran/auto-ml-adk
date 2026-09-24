# Google ADK concept guide

A simple map of **Agent Development Kit (ADK)**: what the name means, what it does, how it compares to LangChain, every major feature in plain language, and what each step of `adk_tutorial.ipynb` is teaching.

Official docs: [adk.dev](https://adk.dev/) · [Technical overview](https://adk.dev/get-started/about/) · [Agents](https://adk.dev/agents/) · [Ollama](https://adk.dev/agents/models/ollama/)

This notebook version runs **locally with Ollama** via LiteLLM. No Gemini / OpenAI / Anthropic keys.

---

## 1. What “ADK” means

**ADK = Agent Development Kit.**

It is Google’s **open-source framework for building production agents**, not just chatbots.

- **Agent** = a worker that can read a goal, think with an LLM, call tools, and talk to other agents.
- **Development Kit** = the pieces you need around that worker: sessions, events, callbacks, evaluation, a runner, a web UI, deploy paths.

Google’s own one-liner ([adk.dev](https://adk.dev/)):

> Build production agents, not prototypes.

Languages: Python, TypeScript, Go, Java, Kotlin.

This project uses **ADK Python**.

---

## 2. What ADK does (simplest picture)

You write:

1. An **agent** (name + instruction + model + optional tools).
2. Optional **sub-agents**, **state**, **callbacks**.
3. A **Runner** that executes the agent.

ADK then runs this loop for you:

```text
User message
    → Runner
        → Agent (LLM reads instruction + history + tools)
            → maybe call a tool
            → maybe transfer to a sub-agent
            → produce a reply
    → Event stream (you print the final response)
```

You do **not** write “if the model asked for a function, run it, send the result back.” That glue is the runtime.

In the notebook that loop is `call_agent_async` + `runner.run_async(...)`.

---

## 3. The building blocks (simple meanings)

From [ADK technical overview](https://adk.dev/get-started/about/). These are the words ADK actually uses.

| Word | Simple meaning | In this notebook |
|---|---|---|
| **Agent / LlmAgent** | One worker with a model, instruction, optional tools | `weather_agent`, `greeting_agent`, root team |
| **Tool** | A Python function the model can call | `get_weather`, `say_hello`, `say_goodbye` |
| **Model** | The LLM brain | `LiteLlm(model="ollama_chat/qwen3:4b")` |
| **Session** | One conversation thread | `SESSION_ID_STATEFUL` |
| **State** | Scratchpad for that session (preferences, last city) | `user_preference_temperature_unit` |
| **Event** | One thing that happened (user text, tool call, final answer) | `event.is_final_response()` |
| **Runner** | The engine that runs the loop | `Runner(agent=..., session_service=...)` |
| **SessionService** | Where sessions/state live | `InMemorySessionService` |
| **Callback** | Your code at a checkpoint (before model, before tool, …) | `before_model_callback`, `before_tool_callback` |
| **Memory** | Recall across *many* sessions (long-term) | Not used here (state is enough) |
| **Artifact** | Saved files (PDFs, images, models) | Not used here |
| **Workflow agent** | Code that *orders* other agents (sequence / loop / parallel) | Not used; this tutorial uses LLM delegation instead |

Two kinds of agents:

- **LlmAgent** — the model decides the next step (used throughout the notebook).
- **Workflow agents** — your code decides the order: `SequentialAgent`, `ParallelAgent`, `LoopAgent`. Docs: [workflow agents](https://adk.dev/agents/workflow-agents/). ADK 2.0 also has [graph workflows](https://adk.dev/graphs/).

---

## 4. How a turn actually runs

```text
1. You send text as types.Content(role="user", ...)
2. Runner loads the Session (history + state)
3. Optional before_agent_callback
4. Optional before_model_callback   ← Step 5 blocks "BLOCK"
5. Model thinks (Ollama via LiteLLM)
6. If the model wants a tool:
      optional before_tool_callback  ← Step 6 blocks Paris
      run the Python function
      optional after_tool_callback
      send the tool result back to the model
7. Maybe transfer to a sub-agent     ← Step 3 greetings / farewells
8. Final text event
9. If output_key is set, save that text into session.state
```

Each of those steps can emit an **Event**. `call_agent_async` walks the stream and prints the **final** one. Drain the whole stream; do not `break` early (Jupyter + OpenTelemetry otherwise logs `Failed to detach context`).

---

## 5. ADK vs LangChain

Same job: LLM + tools + multi-step work. Different product.

| | **Google ADK** | **LangChain** (and LangGraph) |
|---|---|---|
| What it is | An **agent runtime** | A **library of LLM parts** |
| Main object | `Agent` + `Runner` | Chains, then graphs, then agents |
| Who runs the tool loop | ADK (`Runner`) | You, or a LangGraph / agent executor |
| Conversation | First-class `Session` + `State` + `Events` | Memory / checkpointer you attach |
| Multi-agent | Built in (`sub_agents`, workflow agents, graphs) | LangGraph nodes, or agents-as-tools |
| Guardrails | Callbacks / plugins at model and tool time | Middleware, validators, extra graph nodes |
| Local run | `adk web`, `adk run`, `adk api_server` | Your app (FastAPI, LangServe, LangGraph Platform) |
| Default cloud | Gemini / Vertex Agent Platform | Any vendor |
| Other models | `LiteLlm` (Ollama, OpenAI, Anthropic, …) | Native from day one |
| RAG / loaders | Not the focus | Very large catalog |
| Eval | Built-in trajectory evaluation | LangSmith and others |
| Voice / live | `run_live()` + Gemini Live | Separate integrations |

**LangChain** = Lego bricks (prompt, retriever, parser, tool). You snap them together.

**ADK** = a small operating system for agents. You describe the agent; the Runner is the OS.

LangGraph is the closest LangChain piece to ADK workflows: both are graphs of steps. ADK still gives you sessions, events, `adk web`, and callbacks without assembling them.

**Use ADK** when the product *is* an agent (tools, team, state, guardrails, later Vertex).

**Use LangChain** when you need RAG loaders, a huge integration list, or a graph you fully own.

They can coexist (LangChain for retrieval, ADK as the runner). This notebook does not need LangChain.

---

## 6. All ADK features (website), in simple words

Grouped the way [adk.dev](https://adk.dev/) and the [overview](https://adk.dev/get-started/about/) describe them. “In notebook?” means this weather tutorial.

### 6.1 Agents

| Feature | Simple meaning | In notebook? |
|---|---|---|
| **LlmAgent / Agent** | Model + instruction + tools | Yes |
| **name** | Unique id (needed so other agents can transfer to it) | Yes |
| **description** | One-line “what I do” — the root model uses this to delegate | Yes (Step 3) |
| **instruction** | The job description | Yes |
| **tools** | Functions it may call | Yes |
| **sub_agents** | Specialists it can hand work to | Yes (Step 3) |
| **output_key** | Auto-save the final text into `session.state[key]` | Yes (Step 4) |
| **mode** (`chat` / `task` / `single_turn`) | How long the agent stays in control | Not set (defaults) |
| **Managed agents** | Google-hosted specialists (search, code) | No |

Docs: [Simple agents](https://adk.dev/agents/llm-agents/), [Managed agents](https://adk.dev/agents/managed-agents/).

### 6.2 Models

| Feature | Simple meaning | In notebook? |
|---|---|---|
| Gemini string | `model="gemini-2.5-flash"` via Google | No (local) |
| **LiteLlm** | Adapter to 100+ providers | Yes |
| **Ollama** | Local models. Use **`ollama_chat/<tag>`**, not `ollama/` | Yes |
| vLLM / LiteRT-LM / Apigee | Other hosts | No |
| Model routing | Failover between models | No |

**Ollama rules** ([docs](https://adk.dev/agents/models/ollama/)):

1. `export OLLAMA_API_BASE="http://localhost:11434"`
2. `LiteLlm(model="ollama_chat/qwen3:4b")`
3. Model must list **tools** in `ollama show <tag>`
4. Do **not** use provider `ollama` — it can loop on tools and drop context

This notebook also shows the alternate OpenAI-compatible path: `openai/<tag>` against `http://localhost:11434/v1`.

### 6.3 Tools

| Feature | Simple meaning | In notebook? |
|---|---|---|
| **Function tool** | A Python function + docstring becomes a tool | Yes |
| Return a dict | `{"status": "success", "report": "..."}` so the model can branch | Yes |
| **ToolContext** | Extra argument: session state, actions, artifacts | Yes (Step 4) |
| **AgentTool** | Wrap another agent as a tool | No |
| Built-in (search, code) | Google-provided tools | No |
| OpenAPI / MCP | Tools generated from an API spec or MCP server | No |
| Long-running / streaming tools | Tools that take time or yield progress | No |

Docs: [Custom tools](https://adk.dev/tools-custom/).

### 6.4 Sessions, state, memory, artifacts

| Feature | Simple meaning | In notebook? |
|---|---|---|
| **Session** | One chat thread (`app_name` + `user_id` + `session_id`) | Yes |
| **Events** | Full history of that thread | Yes (via Runner) |
| **State** | Key-value scratchpad | Yes |
| No prefix | Lives in this session only | `last_weather_report` |
| `user:` prefix | Follows the user across sessions | `user_preference_temperature_unit` |
| `app:` prefix | Shared for the whole app | Not used |
| `temp:` prefix | Dies at the end of this turn | Not used |
| **InMemorySessionService** | RAM only; gone on restart | Yes |
| **DatabaseSessionService** | SQLite / SQL persistence | No |
| **Memory** | Searchable long-term recall | No |
| **Artifacts** | Versioned files | No |

Docs: [State](https://adk.dev/sessions/state/).

Update state the ADK way:

- In a tool: `tool_context.state["key"] = value`
- On the agent: `output_key="last_weather_report"`
- Not by mutating a copied `session.state` from `get_session()` (that copy is not saved)

### 6.5 Callbacks (guardrails)

Checkpoints. Return `None` = continue. Return an object = skip/replace that step.

| Callback | When | Return to skip | In notebook? |
|---|---|---|---|
| `before_agent_callback` | Before the agent runs | `Content` | No |
| `after_agent_callback` | After the agent finishes | extra `Content` | No |
| **`before_model_callback`** | Before the LLM call | `LlmResponse` | **Step 5** |
| `after_model_callback` | After the LLM replies | replacement `LlmResponse` | No |
| **`before_tool_callback`** | Before a tool runs | `dict` (fake tool result) | **Step 6** |
| `after_tool_callback` | After a tool runs | replacement `dict` | No |

Docs: [Callbacks](https://adk.dev/callbacks/). For reusable policy packs, ADK also has [Plugins](https://adk.dev/plugins/).

### 6.6 Workflows (beyond this notebook)

| Feature | Simple meaning |
|---|---|
| **SequentialAgent** | Run A, then B, then C. Code decides, not the LLM |
| **ParallelAgent** | Run A and B at the same time |
| **LoopAgent** | Repeat until `max_iterations` or `escalate=True` |
| **Graph workflows** (ADK 2.0) | Nodes + edges + branches |
| **Dynamic workflows** | Branching in Python |
| **Collaborative workflows** | Coordinator + specialists (this notebook’s `sub_agents` is the simple form) |

Docs: [Workflows](https://adk.dev/workflows/).

### 6.7 Runtime, eval, deploy (beyond this notebook)

| Feature | Simple meaning |
|---|---|
| `adk web` | Local chat UI + event inspector |
| `adk run` | CLI chat |
| `adk api_server` | HTTP API |
| Evaluation | Score whole **trajectories** (tools + order), not just the last sentence |
| Live / voice | Bidirectional audio/text (`run_live`) |
| Planning | ReAct-style plan-then-act |
| Skills | Packaged extra instructions that fit in context |
| Identity / auth | Credentials for tools |
| Deploy | Vertex / Gemini Enterprise Agent Platform |

---

## 7. Notebook walkthrough (`adk_tutorial.ipynb`)

The notebook is Google’s **Weather Bot team** tutorial, rewritten to use **local Ollama**. Each step adds one ADK idea.

```text
Step 0  Install + Ollama model
Step 1  One agent + one tool + Runner
Step 2  Same agent, two LiteLLM paths to Ollama
Step 3  Team: root + greeting + farewell (delegation)
Step 4  Session state + ToolContext + output_key
Step 5  before_model_callback  (block the word BLOCK)
Step 6  before_tool_callback   (block weather for Paris)
```

### Step 0 — Setup (cells: install, imports, Ollama config, model)

**Goal:** Python packages and a local model. No cloud keys.

| Cell | What it does |
|---|---|
| Install | `pip install google-adk "litellm>=1.84"` |
| Imports | `Agent`, `LiteLlm`, `Runner`, `InMemorySessionService`, `types` |
| Ollama config | `OLLAMA_API_BASE=http://localhost:11434`, tag `qwen3:4b`, ping `/api/tags` |
| Model constants | `AGENT_MODEL = LiteLlm(model="ollama_chat/qwen3:4b", ...)` |

Also sets `OTEL_SDK_DISABLED=true` so Jupyter does not spam OpenTelemetry errors.

Need: Ollama running, `ollama pull qwen3:4b`, and `ollama show qwen3:4b` listing **tools**.

---

### Step 1 — First agent (cells: `get_weather`, define agent, session/runner, `call_agent_async`, run)

**Goal:** Prove the ADK loop with one tool.

**Tool `get_weather(city)`**

- Mock dict: New York, London, Tokyo succeed; anything else errors.
- Returns `{"status": "success", "report": "..."}` or `{"status": "error", "error_message": "..."}`.
- The docstring is the schema the model sees.

**Agent `weather_agent_v1`**

```text
name + AGENT_MODEL + instruction + tools=[get_weather]
```

Instruction tells it: city weather → call the tool; if error, be polite.

**Session + Runner**

```text
InMemorySessionService
  → create_session(app_name, user_id, session_id)
Runner(agent, app_name, session_service)
```

Three IDs: **app**, **user**, **session**. Same three IDs later = same memory.

**`call_agent_async`**

1. Wrap the query as `types.Content`.
2. `async for event in runner.run_async(...)`.
3. Keep the last `event.is_final_response()` text.
4. **Do not `break`** — drain the generator.

**Test queries:** New York (hit), Paris (tool error). You should see `--- Tool: get_weather called ---` then a final sentence.

**Concept:** Agent + tool + session + runner. This is the whole ADK core.

---

### Step 2 — Local Ollama through LiteLLM

**Goal:** The agent is model-agnostic. Only the `LiteLlm(...)` string changes.

Two official paths ([Ollama docs](https://adk.dev/agents/models/ollama/)):

| Path | Model string | Base URL |
|---|---|---|
| Recommended | `ollama_chat/qwen3:4b` | `http://localhost:11434` |
| OpenAI-compatible | `openai/qwen3:4b` | `http://localhost:11434/v1` |

Each path gets its **own** session + runner, same `get_weather` tool.

Tokyo / London tests should still call the tool. Wording of the final sentence may differ; the mock data does not.

**Concept:** Swap brains without rewriting tools or the loop. Always `ollama_chat`, never `ollama`.

---

### Step 3 — Agent team (delegation)

**Goal:** One coordinator, two specialists. ADK routes automatically.

```text
                 user
                   │
            weather_agent_v2  (root)
           /        |         \
   greeting     get_weather    farewell
    agent         (tool)        agent
   say_hello                  say_goodbye
```

**New tools**

- `say_hello(name)` — greeting string.
- `say_goodbye()` — goodbye string.

**Sub-agents**

- `greeting_agent` — *only* hellos, tool `say_hello`. **`description` is required** so the root knows when to transfer.
- `farewell_agent` — *only* goodbyes, tool `say_goodbye`.

**Root `weather_agent_v2`**

- Own tool: `get_weather`.
- `sub_agents=[greeting_agent, farewell_agent]`.
- Instruction: greetings → greeting_agent; farewells → farewell_agent; weather → `get_weather`.

**Auto flow:** the root LLM reads each sub-agent’s **description** and may emit an internal **transfer**. That sub-agent then runs with its own tools.

**Tests:** “Hello”, “weather in New York”, “Thanks, bye”.

**Concept:** Multi-agent by hierarchy + descriptions, not a hand-written router.

---

### Step 4 — Session state (memory in this conversation)

**Goal:** Remember preference and last result across turns.

**Initial state** when creating the session:

```python
{"user_preference_temperature_unit": "Celsius"}
```

`user:` prefix = tied to the user, not only this one chat (still lost on restart with InMemory).

**New tool `get_weather_stateful(city, tool_context)`**

- Same mock weather, but reads `tool_context.state["user_preference_temperature_unit"]`.
- Converts to °F if the preference is Fahrenheit.
- Writes `last_city_checked_stateful`.

**Root also has `output_key="last_weather_report"`** — each final reply is copied into state.

**Demo flow**

1. Weather in New York → Celsius (initial preference).
2. Manually set preference to Fahrenheit (tutorial shortcut; production would use a tool).
3. Weather in New York again → Fahrenheit.
4. “Hello” still delegates to greeting_agent.
5. Print final state: unit, last city, `last_weather_report`.

**Concept:** `ToolContext.state` for tools; `output_key` for the agent’s last sentence. Session is the memory of *this* thread.

---

### Step 5 — Input guardrail (`before_model_callback`)

**Goal:** Stop some user text **before** it reaches the LLM.

Callback `block_keyword_guardrail(callback_context, llm_request)`:

- Looks at the last user message.
- If it contains **`BLOCK`**: return an `LlmResponse` with a refusal → **no model call, no tools**.
- Else: return `None` → continue.

Wired as `before_model_callback=block_keyword_guardrail` on the **root** agent.

**Tests**

- “weather in London” → normal.
- “BLOCK the request for weather in Tokyo” → blocked. Tokyo tool should **not** run.

**Concept:** `return None` = allow. `return LlmResponse(...)` = skip the model. Safety without changing the model.

---

### Step 6 — Tool argument guardrail (`before_tool_callback`)

**Goal:** Stop a **specific tool call** even if the model asked for it.

Callback `block_paris_tool_guardrail(tool, args, tool_context)`:

- If tool is `get_weather_stateful` and city is **Paris**: return a fake error dict → **tool function never runs**.
- Else: return `None` → run the tool.

Root now has **both** callbacks.

**Tests**

- Weather in New York → allowed.
- Weather in Paris → blocked by callback (not by the tool’s own “city not found”).
- Message containing BLOCK → still blocked by Step 5 (model never called).

**Concept:** Two layers. Model guardrail = “don’t even think.” Tool guardrail = “you may think, but you may not call that function with those args.”

---

## 8. What you have built by the last cell

A **weather agent team** that:

1. Runs on **local Ollama** (`ollama_chat/qwen3:4b`).
2. Looks up mock weather with a **tool**.
3. **Delegates** hi/bye to specialists.
4. **Remembers** temperature unit and last city in **session state**.
5. **Blocks** the word BLOCK before the model.
6. **Blocks** Paris before the weather tool.

That is the ADK core: **Agent + Tools + Runner + Session + Callbacks + Sub-agents**.

---

## 9. How to run this project

1. Install [Ollama](https://ollama.com/) and start it (`ollama serve`).
2. `ollama pull qwen3:4b` (or another tag with **tools**).
3. Python 3.10+ with:

   ```bash
   pip install "google-adk" "litellm>=1.84"
   ```

4. Open `adk_tutorial.ipynb`, pick that kernel, run from the top.

Optional, from the parent folder of an agent package:

```bash
export OLLAMA_API_BASE="http://localhost:11434"
adk web --port 8000
```

`adk web` is for development only ([runtime docs](https://adk.dev/runtime/web-interface/)).

---

## 10. Mini glossary

| Term | One line |
|---|---|
| **Agent** | Worker: model + instruction + tools |
| **Tool** | Python function the model can call |
| **Runner** | Engine that executes one user turn |
| **Session** | One conversation |
| **State** | Scratchpad on that conversation |
| **Event** | One recorded step (message, tool, reply) |
| **Callback** | Your hook before/after model or tool |
| **Sub-agent** | Specialist the root can transfer to |
| **LiteLlm** | Bridge from ADK to Ollama and other APIs |
| **ollama_chat** | Required LiteLLM provider prefix for Ollama + tools |
| **output_key** | Save the agent’s final text into state |
| **ToolContext** | State + actions inside a tool |
| **Delegation / auto flow** | Root LLM hands a turn to a sub-agent |
| **Guardrail** | Callback that can skip model or tool |

---

## 11. Official links

| Topic | URL |
|---|---|
| Home | https://adk.dev/ |
| Technical overview | https://adk.dev/get-started/about/ |
| Python quickstart | https://adk.dev/get-started/python/ |
| Agents | https://adk.dev/agents/ |
| LLM agents | https://adk.dev/agents/llm-agents/ |
| Models | https://adk.dev/agents/models/ |
| Ollama | https://adk.dev/agents/models/ollama/ |
| LiteLLM | https://adk.dev/agents/models/litellm/ |
| Custom tools | https://adk.dev/tools-custom/ |
| State | https://adk.dev/sessions/state/ |
| Callbacks | https://adk.dev/callbacks/ |
| Workflows | https://adk.dev/workflows/ |
| Runtime / `adk web` | https://adk.dev/runtime/ |
| Multi-tool tutorial | https://adk.dev/tutorials/multi-tool-agent/ |
