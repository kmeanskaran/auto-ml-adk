# The agents

This page explains who is on the team, what each agent does, how they hand work to each
other, and how Google's Agent Development Kit (ADK) runs them.

## The idea

A real ML team doesn't let one person build a model, grade it and ship it. Someone
builds, someone else looks for holes, and a person who owns the business decides. This
project copies that structure with LLM agents:

- **Agents decide what to do.** Which features to build, which models to try, what looks
  wrong. Nothing about the dataset is written into their prompts.
- **The harness decides how it is checked.** The order of the stages, what a complete
  hand-over must contain, how models are scored, and when a person must be asked. That
  part is plain Python, so it gives the same result every time.
- **You decide what ships.** By default the run stops for you at the features and at
  go-live.

The agents never decide from what they "see" in the data. They write and run Python and
decide from what that code prints. Every number they report comes from a script in the
run folder, so anyone can rerun it.

## The team

| Agent | ADK name | Job | Hands over |
|---|---|---|---|
| **Analyst** (first look) | `analyst_profile` | Measures what matters in the data before anyone models it: where the outcome rate moves, missing values, skew. | `submit_summary`: a headline, up to five findings, up to three charts |
| **Engineer**, features stage | `engineer_features` | Writes `src/data.py` and `src/features.py`: cleaning, the population, the split, and features available at the prediction moment. Proposes a training plan. | `propose_plan`, then `submit_receipt` |
| **Engineer**, model stage | `engineer_model` | Writes `src/train.py` and trains every model in the approved plan side by side. | `submit_receipt` |
| **Skeptic** | `skeptic_features`, `skeptic_model` | Gives each stage one quick review. Looks for leakage, a split unlike real traffic, features built from protected attributes, implausible results, and models no better than the baseline. Never edits the engineer's files. | `submit_review`: pass or concerns, findings, recommendations |
| **Analyst** (chat) | `analyst` | Answers your questions in the console about the data, features, models and production traffic, with charts. Read-only, and it declines off-topic questions. | Markdown answer and charts |

Each agent is an ADK `LlmAgent` built in `app/agents.py`. The role prompt is fixed,
while everything specific to a run (the business settings, the measured profile, your
feedback, past runs' lessons) travels in the request message.

### Tools

Every agent works inside its run folder with the same file tools, plus one or two
hand-over tools (`app/harness/tools.py`):

| Tool | What it does |
|---|---|
| `list_files`, `search`, `read_file` | Look around the run folder. |
| `write_file`, `write_and_run`, `run_python` | Write a script and run it in a separate process. |
| `check_stage` | Asks the harness whether the hand-over is complete before submitting. |
| `submit_summary`, `submit_receipt`, `submit_review`, `propose_plan` | Hand-overs. The pipeline reads these files, not the chat. |
| `show_chart` | Shows a chart in the console (chat analyst only). |

Scripts can import `kit.py`, a set of tested helpers for statistics the agents would
otherwise rewrite each time: outcome rate by group, whether a gap is real, drift between
training and production data, the live model's scores and feature importance.

### Where agent code runs

`app/harness/environment.py` runs every script in a child process with a clean
environment, so agent-written code can't read the server's API keys. A source screen
also rejects obvious mistakes such as network imports, subprocesses, `eval` and deleting
folders, and every script is stopped after `limits.script_seconds`. The screen is a
tripwire, not a sandbox. Real isolation comes from the container that the server runs in.

## The pipeline

The order is fixed. The agents fill in each step.

```
START → profile → analyst → check
      → engineer (features) → check ─fix→ engineer
      → skeptic → REVIEW FEATURES ─feedback→ engineer
      → TRAINING PLAN
      → engineer (model) → check ─fix→ engineer       harness evaluates every candidate
      → skeptic → PROMOTE ─retrain→ engineer (model)
                          ─replan→ training plan
                          ─features→ engineer (features)
                          promote / keep → model registry
```

Capitalised steps are **gates**. A gate listed in `autonomy.ask_human` always waits for
you. A gate not listed is settled by the team only when the skeptic passed the stage and
the harness raised no warning. Anything else comes to you, with the reason shown.

| Step | Code | What happens |
|---|---|---|
| `start` | `pipeline.start` | Creates the run folder and measures every column's type, missing share and solo signal (`reports/profile.json`). If the last run used the same data and you gave no feedback, it reuses that run's first look and tells the team to explore new ideas. |
| `check_*` | `pipeline.check` | Checks the hand-over against a contract (`app/harness/contracts.py`). If it is incomplete, the agent is sent back once (`limits.fix_rounds`), and never twice for the same problems. |
| `review_features` | `pipeline.review_features` | Continue (the feature view is frozen into the feature store), send selected recommendations or your own words back to the engineer, or discard. |
| `plan_gate` | `pipeline.plan_gate` | The models to train and the metric that picks the winner: the engineer's proposal or yours. |
| `promote_gate` | `pipeline.promote_gate` | Promote, keep as a candidate, retrain, rework the features, replan, or discard. A fair-lending warning always comes to a person. |
| `finish` | `pipeline.finish` | Records the run in `runs/index.jsonl`, where the next run reads its lessons. |

### What the harness checks, not the agents

- **Scoring.** `app/harness/evaluation.py` re-scores every candidate itself. It picks the
  best on validation, using cross-validation when there are few rows, then tests it once
  on the held-out split. It also computes the do-nothing baseline, calibration, and the
  threshold that minimises the cost of mistakes from `config.costs`.
- **Fair lending.** For each attribute in `config.fairness`, it compares approval rates
  and the share of eligible applicants approved across groups. A ratio below `min_ratio`
  is a warning.
- **Going live.** A model the team promotes must beat the baseline and the current
  production model on the plan's metric.

### Runs build on each other

A new run copies the last finished run's code to `prior/`. The engineer reads it, keeps
what still holds on the data, and changes what your feedback or new measurements say.
Without feedback, the run explores: it keeps what worked and tries at least one new
feature and model family. **Clear runs** in the console starts cold.

## How ADK runs it

| ADK concept | Here |
|---|---|
| `LlmAgent` | Each team member (`app/agents.py`): model, static instruction, tools, callbacks. |
| `Workflow` with `@node` functions and routed edges | The pipeline graph (`app/pipeline.py`). Each node yields `Event`s that carry output, state changes and the route to take. |
| `RequestInput` | How a gate pauses for a person. The console posts the answer and the workflow resumes at that gate. |
| `App` with `ResumabilityConfig(is_resumable=True)` | `app/agent.py`. A paused run survives a server restart and resumes from the answer. |
| Session state | Holds the run name, fix-round counters and gate rounds, so a resume on another server finds the same run. |
| Plugins | `RunScope` binds each step to its run folder. `TracePlugin` logs every agent turn, tool call and file written to `runs/<run>/logs/`. |
| Callbacks | `before_model_callback`: skip the call once handed over, apply the console's model choice, trim long history, use the prompt cache. `before_tool_callback`: enforce the tool budget. `on_tool_error_callback`: return tool errors to the agent instead of crashing. |
| `get_fast_api_app` | `app/fast_api_app.py` serves the ADK routes and dev UI, A2A at `/a2a/app`, the console API at `/api`, and `/predict`. |

### Models

Gemini is the default, through Google AI Studio (`GEMINI_API_KEY`) or Vertex AI
(`GOOGLE_GENAI_USE_VERTEXAI=true`, which the deployment uses). `config.models.team`
sets the team's model and `config.models.analyst` the chat analyst's. `ML_MODEL`
overrides the team's model with any LiteLLM model, such as Ollama. The console can
switch the model and thinking level during a run.

`app/agents.py` wraps Gemini so a long run survives an expired login token (it rebuilds
the client and retries once) and a rejected prompt cache (it resends without the cache).

### Keeping the LLM bill down

- **Briefing, not re-reading.** Each request already contains the data at a glance, the
  analyst's findings, the bar to beat and past lessons. The skeptic gets the stage's
  code in its request.
- **Tool budget.** `limits.tool_budget` calls per turn (10 for the skeptic), with a
  wrap-up reminder at two thirds.
- **Compaction.** Old tool output is summarised once a turn's history passes
  `limits.compact_above_chars`.
- **Prompt caching.** `app/harness/prompt_cache.py` stores each role's fixed prompt and
  tools on Gemini once, in the background, and every call reuses it.
- **One pass.** One review per stage, no agent loops, and at most one fix round.

## What a run leaves behind

```
runs/<run>/
  src/              the engineer's code: data.py, features.py, train.py
  checks/           the analyst's and skeptic's scripts
  reports/          profile, summary, features, plan, evaluation (JSON)
  receipts/         each stage's hand-over
  reviews/          the skeptic's reviews
  notes/            the engineer's notes
  decisions.jsonl   every gate decision, and whether the team or a person made it
  logs/             the full trace
  prior/            the previous run's code this run builds on
feature_store/<view>/<version>/   frozen feature definitions and tables
registry/                         versioned models, and which one is in production
```

Locally these folders are on your disk. When deployed, `app/harness/state_sync.py`
mirrors them to a Cloud Storage bucket (see [gcp-deployment.md](gcp-deployment.md)).
