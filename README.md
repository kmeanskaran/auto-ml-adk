# ML Team

A team of AI agents that builds a machine-learning model with you, built on Google's
[Agent Development Kit (ADK)](https://adk.dev/). The agents write and run their own code;
you review the work at the points where a human decision matters.

The example problem: **predict which hotel bookings will be cancelled**, at the moment
the booking is made.

## How it works

The pipeline is an ADK Workflow with five stages. You act at the three marked ✋.

| # | Stage | Who | What happens |
|---|-------|-----|--------------|
| 1 | Understand the data | Analyst | The harness profiles every column; the analyst summarises what matters. Full report under `›`. |
| 2 | Engineer features ✋ | Engineer, Skeptic | The engineer builds features; the skeptic recommends what to add, improve or remove. You send feedback or continue. |
| 3 | Training plan ✋ | You | Pick the models to compare and the metric that decides the winner. |
| 4 | Train and evaluate | Engineer, Skeptic | The engineer trains; the harness scores every candidate on every metric. |
| 5 | Promote ✋ | You | Pick a candidate: promote it, keep it as a candidate, retrain with feedback, or change the plan. |

Along the way:

- **Feature store**: approved features are saved as a version (definition, transform
  code, train/valid/test tables). Training and serving use the same transform.
- **Model registry**: every kept model is a version with its metrics and lineage.
  The model library compares versions, rolls back, or removes one.
- **Analyst chat**: answers questions about the data and the pipeline's results,
  with a chart when it helps. It answers nothing else.

Metrics are always computed by the harness, never reported by the agents themselves.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/) with the
`gpt-oss:120b-cloud` model (or set `ML_MODEL` to another LiteLLM model).

```bash
uv sync                                        # install dependencies into .venv
cp .env.example .env                           # local settings
```

Download the [Hotel Booking Demand](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11)
dataset (Antonio, Almeida & Nunes 2019, CC BY 4.0) and split it into what the team
sees and later production traffic:

```bash
mkdir -p data/datasets/hotels
curl -sSfL -o data/datasets/hotels/hotels.csv \
  "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
uv run python scripts/make_environment.py      # writes data/lake/ and data/traffic/
```

Start the server and open the console:

```bash
uv run python -m app.fast_api_app              # http://localhost:8000/team
```

Press **Run pipeline** and follow the stages.

## Scoring bookings

Once a model is promoted, `POST /predict` scores raw booking records with the
production version. `POST /predict/<version>` scores with any version in the library.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"booking_id": "B009776", "hotel": "Resort Hotel", "lead_time": 74,
       "arrival_date_year": 2017, "arrival_date_month": "January",
       "arrival_date_week_number": 1, "arrival_date_day_of_month": 1,
       "stays_in_weekend_nights": 1, "stays_in_week_nights": 0, "adults": 2,
       "children": 0, "babies": 0, "meal": "BB", "country": "PRT",
       "market_segment": "Online TA", "distribution_channel": "TA/TO",
       "is_repeated_guest": 0, "previous_cancellations": 0,
       "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
       "booking_changes": 0, "deposit_type": "No Deposit", "agent": 5,
       "company": null, "days_in_waiting_list": 0, "customer_type": "Transient",
       "adr": 57.6, "required_car_parking_spaces": 0, "total_of_special_requests": 0}]}'
```

```json
{"version": "v2", "predictions": [{"id": "B009776", "score": 0.492036, "flag": 0}]}
```

`score` is the chance the booking cancels; `flag` is 1 when the score reaches the
version's threshold, chosen to minimise the business cost in `config/pipeline.yml`.
A record missing a needed column returns 422; an unknown version returns 404.

## Configuration

`config/pipeline.yml` holds the dataset, target column, feature view name, the
prediction moment and the business costs of a missed cancellation and a false alarm.
Models and the metric are chosen per run in the console.

| Environment variable | Default | Purpose |
|---|---|---|
| `ML_MODEL` | `ollama_chat/gpt-oss:120b-cloud` | LLM the agents use (any LiteLLM model) |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama server |
| `ML_RUNS_ROOT` | `runs/` | Where pipeline runs are written |
| `ML_REGISTRY_ROOT` | `registry/` | Model registry |
| `ML_FEATURE_STORE_ROOT` | `feature_store/` | Feature store |

## Project layout

```
app/
  agent.py           root agent: the pipeline workflow
  pipeline.py        stages, human reviews and routing
  agents.py          analyst, engineer and skeptic prompts
  harness/           tools, stage contracts, profiling, feature store,
                     evaluation, model registry, sandboxed code runner
  ui/                console (/team) and serving (/predict) routes
  fast_api_app.py    FastAPI server
config/pipeline.yml  business settings
data/                lake and traffic (gitignored)
runs/                one folder per pipeline run; runs/analysis for the chat
feature_store/       versioned feature views
registry/            versioned models; production.json is what /predict serves
tests/               unit, integration and eval tests
docs/                design notes and setup log
archive/             first implementation, kept for reference
```

## Development

```bash
uv run pytest tests/unit                       # harness, registry, feature store, workflow
uvx ruff check app tests                       # lint
```

The project was scaffolded with [Agents CLI](https://google.github.io/agents-cli/);
`agents-cli playground`, `agents-cli eval` and `agents-cli deploy` work from the repo
root. See `docs/setup.md` for the full setup log.

## Status

- Runs locally on Ollama. Gemini and deployment on Agent Runtime are next.
- Agent-written code runs in a separate process with no access to the server's
  secrets. Network isolation comes from the deploy target's container.
