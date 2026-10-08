# ML Team

A team of AI agents that works like a small machine learning team. Give it a CSV and a
goal: an **analyst**, an **engineer** and a **skeptic** explore the data, build features,
train and compare models, review each other's work, and ask you only when a decision
matters. The winning model is versioned and served on `/predict`.

Built with Google's [Agent Development Kit (ADK)](https://adk.dev/) and Gemini,
scaffolded with [Agents CLI](https://google.github.io/agents-cli/), and deployable to
Agent Runtime and Cloud Run with Terraform and GitHub Actions.

The example problem that ships with the repo: **decide, when a home-loan application is
submitted, whether it will be approved.** Nothing about loans is written into the agents.
Only `config/config.yml` and the data are loan-specific. Point them at your own CSV and
the same team works on your problem.

## What it does

The agents don't decide by reading the data themselves. They write and run Python that
reads it, and decide from what that code prints. The harness fixes the order of the
steps and the rules. The agents fill in the work.

| Stage | Who | What happens |
|---|---|---|
| Understand the data | Analyst | Profiles every column and summarises what matters. |
| Engineer features ✋ | Engineer, Skeptic, You | The engineer writes the features, the skeptic reviews them once, and you continue or send fixes back. |
| Choose models | Engineer | Proposes the models and the metric that picks the winner. |
| Train and test | Engineer, Skeptic | The engineer trains, the harness scores every model, and the skeptic reviews once. |
| Go live ✋ | You | Put the winner live, keep it, send it back with fixes, or discard the run. |

✋ = waits for you by default (`autonomy.ask_human` in `config/config.yml`).

**Rules the harness enforces:**

- **No agent scores its own model.** The harness retrains every candidate, picks the
  winner on validation, then tests it once on rows no model has seen.
- **Mistakes have a price.** Models are ranked by the business cost of their mistakes
  and must beat doing nothing.
- **Fair lending.** Approval rates are compared across protected attributes. A gap below
  the four-fifths rule is a warning, and that model never goes live without a person.
- **One review, then you.** The skeptic reviews each stage once, and you are the second
  reviewer. Every decision is logged with who made it.
- **Agent code is contained.** Scripts run in a separate process without the server's
  environment variables, so they can't read API keys.

Each run keeps a full trace, versioned features, versioned models, and a history the
next run builds on. See [docs/agents.md](docs/agents.md) for how the agents work.

## What you need

| | Required for | Notes |
|---|---|---|
| **Gemini access**, one of: | running the agents | |
| · a [Google AI Studio API key](https://aistudio.google.com/apikey) | local runs | Easiest. No Google Cloud project needed. |
| · a Google Cloud project with Vertex AI | local runs or deployment | Uses your `gcloud` login. No key is stored. |
| **Docker** | `./run.sh` | Or [uv](https://docs.astral.sh/uv/) and Node.js 24 for `./run.sh local`. |
| `gcloud`, Terraform ≥ 1.11, `gh`, Agents CLI | deploying to Google Cloud only | See [docs/gcp-deployment.md](docs/gcp-deployment.md). |

You can also run the agents on a local or cloud model through Ollama (see
[Configuration](#configuration)).

## Run it locally

**1. Clone**

```bash
git clone https://github.com/kmeanskaran/auto-ml-adk.git
cd auto-ml-adk
cp .env.example .env
```

**2. Give it Gemini.** Edit `.env` and choose one option.

*Option A: Google AI Studio key.* Comment out the three Vertex lines and set the key:

```dotenv
# GOOGLE_GENAI_USE_VERTEXAI=true
# GOOGLE_CLOUD_PROJECT=your-gcp-project-id
# GOOGLE_CLOUD_LOCATION=global
GEMINI_API_KEY=your-api-key
```

*Option B: Vertex AI in your Google Cloud project.* Keep the Vertex lines, set your
project ID, and log in once:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project your-gcp-project-id
gcloud services enable aiplatform.googleapis.com --project your-gcp-project-id
```

`.env` is gitignored, so your key or project ID is never committed.

**3. Start it**

```bash
./run.sh          # builds and starts both containers, then follows the agents' log
```

| | URL |
|---|---|
| Console | http://localhost:3000 |
| Backend API, `/predict` | http://localhost:8000 |
| ADK dev UI | http://localhost:8000/dev-ui |

Open the console and press **Run pipeline**. The run pauses at the features review and
the go-live decision for you to answer.

```bash
./run.sh logs     # follow the agents' log again
./run.sh status   # what is running
./run.sh down     # stop
./run.sh local    # no Docker: backend with uv, console with npm
```

Runs, models and features are written to `runs/`, `registry/` and `feature_store/` on
your disk (all gitignored).

## Score applications

Once a model is live, `POST /predict` scores raw records with it, and
`POST /predict/<version>` uses any version in the model library.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"Loan_ID": "LP001015", "Gender": "Male", "Married": "Yes",
       "Dependents": "0", "Education": "Graduate", "Self_Employed": "No",
       "ApplicantIncome": 5720, "CoapplicantIncome": 0, "LoanAmount": 110,
       "Loan_Amount_Term": 360, "Credit_History": 1, "Property_Area": "Urban"}]}'
```

```json
{"version": "v1", "predictions": [{"id": "LP001015", "score": 0.842117, "flag": 1}]}
```

`score` is the probability of approval. `flag` is 1 when the score reaches the threshold
the harness chose to minimise the cost of mistakes.

## Configuration

`config/config.yml` is the only config file. Changes apply from the next run, with no
rebuild needed.

| Key | What it sets |
|---|---|
| `goal`, `dataset`, `traffic`, `target`, `positive_value` | The problem, the training CSV and the production CSV in `data/`, and the outcome column. |
| `prediction_moment` | What may be known when a prediction is made. The skeptic checks features against it. |
| `costs` | The price of each kind of mistake, used to rank models and set the threshold. |
| `fairness` | Protected attributes and the minimum approval-rate ratio. |
| `autonomy.ask_human` | Which decisions wait for you: `features`, `plan`, `promote`. |
| `models` | The Gemini model per role, the thinking level, and prompt caching. |
| `limits` | Tool calls per turn, fix rounds, CV folds, and the script timeout. |

To use a model other than Gemini, set `ML_MODEL` to any LiteLLM model, for example
`ML_MODEL=ollama_chat/gpt-oss:120b-cloud ./run.sh`.

**Your own data:** put `train.csv` (with the outcome column) and `test.csv` (same
columns, no outcome) in `data/<name>/`, then update `dataset`, `traffic`, `target`,
`positive_value` and `fairness` in `config/config.yml`.

## Deploy to Google Cloud

| Part | Runs on |
|---|---|
| Backend: ADK agents, harness, `/predict` | **Agent Runtime** (Vertex AI) |
| Console: Next.js | **Cloud Run** |
| Runs, models, features | Cloud Storage bucket |
| Infrastructure | Terraform, in `deployment/terraform/` |
| CI/CD | GitHub Actions, signed in with Workload Identity Federation (no stored keys) |
| Tracing | Cloud Trace and BigQuery |

[docs/gcp-deployment.md](docs/gcp-deployment.md) explains how it fits together and how
to deploy your own copy. [SETUP.md](SETUP.md) is the command-by-command log of how this
repository was built and deployed.

## Project layout

```
app/
  agent.py           the ADK App: the pipeline workflow and plugins
  pipeline.py        the stages, checks, reviews and routing (an ADK Workflow)
  agents.py          the analyst, engineer and skeptic (ADK LlmAgents)
  harness/           tools, evaluation, feature store, model registry, tracing
  ui/                console API (/api) and serving (/predict)
  fast_api_app.py    the server: ADK routes, A2A, console API, /predict
frontend/            the console (Next.js)
config/config.yml    the one config file
data/lending-loan/   example data: train.csv and test.csv
deployment/          Terraform and the project ID
scripts/             rebuild, tear down, and GitHub variables
tests/               unit, integration and eval tests
docs/                how the agents work, how to deploy
```

## Development

```bash
uv sync                                                # install
uv run pytest tests/unit                               # unit tests
uvx ruff check app tests                               # lint
uvx pre-commit install                                 # the same checks on every commit
(cd frontend && npm ci && npm run typecheck && npm run build)
```

## Documentation

- [docs/agents.md](docs/agents.md): the agents, what each one does, and how ADK runs them
- [docs/gcp-deployment.md](docs/gcp-deployment.md): Agents CLI, Terraform, CI/CD, and deploying your own copy
- [SETUP.md](SETUP.md): the exact commands used to build this deployment
