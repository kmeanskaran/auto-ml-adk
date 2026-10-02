# ML Team

A team of AI agents that works like a small ML team, built on Google's
[Agent Development Kit (ADK)](https://adk.dev/). The agents write and run their own code,
review each other's work, and settle their own disagreements; you are asked only where
a human decision matters. Every step they take is traced.

The example problem, from `notebooks/Loan-status-prediction.ipynb`: **decide, when a
home-loan application is submitted, whether it will be approved**.

## How it works

The pipeline is an ADK Workflow with five stages.

| # | Stage | Who | What happens |
|---|-------|-----|--------------|
| 1 | Understand the data | Analyst | The harness profiles every column; the analyst summarises what matters. Full report under `›`. |
| 2 | Engineer features | Engineer, Skeptic | The engineer writes `data.py` and a row-wise `features.py`; the skeptic tries to break them and recommends changes. |
| 3 | Training plan | Team (or you) | The engineer proposes the models to compare and the metric that decides the winner. |
| 4 | Train and evaluate | Engineer, Skeptic | The engineer trains; the harness scores every candidate on every metric. |
| 5 | Go live ✋ | You | Promote a candidate, keep it, retrain with feedback, change the features, or replan. |

**How little it asks you** is set in `config/config.yml` under `autonomy`:

- While the skeptic has concerns, the engineer gets every recommendation and tries
  again, for up to `self_review_rounds` rounds, without asking you.
- A review listed in `ask_human` (by default only `promote`) always waits for you.
  The others the team settles: sound features continue, the proposed plan trains,
  and a sound model goes live if it beats production and doing nothing.
- If the team cannot settle a concern within its rounds, the review goes to you,
  with the reason shown above the question.

Every decision, the team's or yours, lands in the run's `decisions.jsonl`.

Along the way:

- **Tracing**: an ADK plugin (`app/harness/trace.py`) records every agent, model
  reply, tool call and decision. Per run, in `runs/<run>/logs/`:
  `activity.log` (one plain line per step, also shown in the console's Activity panel
  and on the server's terminal), `trace.jsonl` (the same steps as records: arguments,
  results, seconds, tokens), and `code/` (a numbered copy of every file an agent
  wrote, so each version of each script traces back to who wrote it and when).
- **Feature store**: approved features are saved as a version (definition, transform
  code, train/valid/test tables). Training and serving use the same transform.
- **Model registry**: every kept model is a version with its metrics and lineage.
  The model library compares versions, rolls back, or removes one.
- **One run per session**: the run's folder name is stored in the ADK session state
  (`run`), and every step binds to it (`app/harness/scope.py`), so sessions running side
  by side, or a run resumed by another server, each write to their own folder.
- **Run history**: `runs/index.jsonl` records what every run tried, scored and decided
  (`GET /api/runs`). The next run's engineer and skeptic get the production model's
  score to beat and the lessons of the last runs on the same data, as hypotheses to
  test, never as findings.
- **Fair lending**: the harness compares each candidate's approval rate, and its share
  of eligible applicants approved, across the groups of `fairness.attributes`. A gap
  below `min_ratio` (the four-fifths rule) is a warning, and such a model never goes
  live without a human, whatever `autonomy` says. A feature built from a protected
  attribute is pointed out to the skeptic.
- **Few tool calls, small prompts**: each request carries a briefing (the data at a
  glance, the analyst's findings, past runs) so agents do not re-read reports;
  `write_and_run` and `search` replace common pairs of calls; a file an agent already
  has is not sent again while unchanged; overlong review lines are all named at once;
  a turn has a tool budget (a reminder at 30 calls, only hand-over tools after 45);
  and once a turn's history passes 120k characters, copies a later read, write or run
  superseded are elided from what the model is sent (the session keeps everything).
- **Prompt caching**: each role prompt is a `static_instruction`, the stable head of
  every request, and the App sets `ContextCacheConfig`: Gemini caches it server-side,
  providers that cache a marked prefix get the marks through LiteLLM, and Ollama reuses
  its KV cache for the unchanged prefix.
- **Analyst chat**: answers questions about the data, the features, the models and
  production traffic, with stats and a chart; ask for "a report on …" to get sections,
  tables and up to three diagrams. It starts from a briefing of what the team has now
  (live model, feature store, latest run) and uses `kit.py` (`app/harness/analyst_kit.py`),
  tested helpers for rates with 95% intervals, whether a gap is real (chi-square,
  Cramér's V), drift between the training data and production (PSI), and the live model's scores
  and permutation importance. It answers nothing else.
- **Skeptic**: at most three recommendations, ranked by impact, and only ones that would
  change the result. The team applies them itself; when you are asked, they come
  pre-ticked, so improving is one click.
- **Console**: an overview (live model, feature store, past runs, when you are asked),
  the pipeline next to the analyst, and tabs for the activity trace, the feature store
  (every version and its features, which models use it, protected-attribute flags), the
  model library and past runs.

Metrics are always computed by the harness, never reported by the agents themselves.

## Quick start

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/)
(for the data script) and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
as `GEMINI_API_KEY` in `.env` (or set `ML_MODEL` to another LiteLLM model, e.g.
`ollama_chat/gpt-oss:120b-cloud` with [Ollama](https://ollama.com/)). `./run.sh local`
needs Node.js instead of Docker.

```bash
cp .env.example .env                           # local settings (run.sh does this too)
```

The data comes with the repo: the [Loan Prediction](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/)
`train.csv` and `test.csv`, in `data/lending-loan/`. The team learns from `train.csv`
(614 labelled applications); `test.csv` (367 unlabelled ones) plays production
traffic. `dataset` and `traffic` in `config/config.yml` name the two files; point them
at your own CSVs in `data/` to give the team another problem.

Start the team. `run.sh` builds two containers with `docker-compose.yml`, the
backend and the console, waits until both are healthy, then follows the agents' log:

```bash
./run.sh                 # console http://localhost:3000, backend http://localhost:8000
./run.sh logs            # follow the agents' log again (Ctrl+C stops following only)
./run.sh status          # what is running
./run.sh down            # stop both
./run.sh local           # no Docker: uv backend + Next.js dev server on this machine
```

`run.sh` checks for the data and for the model's key (or Ollama) first. `BACKEND_PORT` and `FRONTEND_PORT`
pick other ports; `ML_MODEL` picks another LLM. Runs, models, features and sessions
live on your machine (`runs/`, `registry/`, `feature_store/`, `.adk/`), mounted into
the backend container, so a rebuild keeps them.

Open the console, press **Run pipeline** and watch the Activity panel.

## Architecture

```
browser ──► frontend (Next.js, :3000) ──/api/*──► backend (FastAPI + ADK, :8000)
                                                    ├─ /api      console API (JSON)
                                                    ├─ /predict  served models
                                                    └─ agents ── Ollama on the host
```

- **backend** (`app/`, root `Dockerfile`): the ADK workflow and agents, the harness,
  the analyst, the JSON API the console uses (`/api/view`, `/api/pipeline/*`,
  `/api/chat`, `/api/registry/*`), and `/predict`. It serves no pages.
- **frontend** (`frontend/`, Next.js + React + TypeScript): the console. Its
  `/api/*` route forwards to the backend at `BACKEND_URL`, read at runtime, so the
  browser only ever talks to the frontend and no CORS setup is needed.

## Scoring applications

Once a model is promoted, `POST /predict` scores raw application records with the
production version. `POST /predict/<version>` scores with any version in the library.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"Loan_ID": "LP001015", "Gender": "Male", "Married": "Yes",
       "Dependents": "0", "Education": "Graduate", "Self_Employed": "No",
       "ApplicantIncome": 5720, "CoapplicantIncome": 0, "LoanAmount": 110,
       "Loan_Amount_Term": 360, "Credit_History": 1, "Property_Area": "Urban"}]}'
```

```json
{"version": "v1", "predictions": [{"id": "LP001015", "score": 0.874551, "flag": 1}]}
```

`score` is the chance the application is approved; `flag` is 1 when the score reaches
the version's threshold, chosen to minimise the business cost in `config/config.yml`.
A record missing a needed column returns 422; an unknown version returns 404.

## Configuration

`config/config.yml` holds the dataset, the target column and its positive value
(`Y`), the feature view name, the prediction moment, the business costs of a missed
approval and a wrong approval, the protected attributes the fair-lending check
compares (`fairness`), and how much the team decides alone (`autonomy`).

| Environment variable | Default | Purpose |
|---|---|---|
| `ML_MODEL` | (unset) | Overrides `models.team` in `config/config.yml`, e.g. `ollama_chat/gpt-oss:120b-cloud` |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama server |
| `ML_RUNS_ROOT` | `runs/` | Where pipeline runs are written |
| `ML_REGISTRY_ROOT` | `registry/` | Model registry |
| `ML_FEATURE_STORE_ROOT` | `feature_store/` | Feature store |

## Project layout

```
app/
  agent.py           root agent: the pipeline workflow
  pipeline.py        stages, team and human reviews, routing
  agents.py          analyst, engineer and skeptic prompts
  harness/           tools, stage contracts, profiling, feature store,
                     evaluation, model registry, sandboxed code runner,
                     tracing plugin (trace.py)
  ui/                console API (/api) and serving (/predict) routes
  fast_api_app.py    FastAPI server
frontend/            the console: Next.js app (components/, lib/, app/), Dockerfile
docker-compose.yml   backend + frontend containers; ./run.sh drives it
config/config.yml    the one config: business settings, models, limits
data/lending-loan/   train.csv and test.csv, the example data (also in the image)
notebooks/           the loan notebook the pipeline automates
runs/                one folder per pipeline run (logs/ holds the trace);
                     runs/analysis for the chat
feature_store/       versioned feature views
registry/            versioned models; production.json is what /predict serves
tests/               unit, integration and eval tests
archive/             first implementation and the earlier hotel example
```

## Development

```bash
uv run pytest tests/unit                       # harness, registry, feature store, workflow
uvx ruff check app tests                       # lint
(cd frontend && npm run typecheck && npm run build)   # console
```

The project was scaffolded with [Agents CLI](https://google.github.io/agents-cli/);
`agents-cli playground`, `agents-cli eval` and `agents-cli deploy` work from the repo
root.

## Status

- Runs on Gemini (API key) by default, or on Ollama. Deployment on Agent Runtime is next.
- Agent-written code runs in a separate process with no access to the server's
  secrets. Network isolation comes from the deploy target's container.
