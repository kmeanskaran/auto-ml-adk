# ML Team: Automated ML Engineering with AI Agents

**An analyst, an engineer and a skeptic, all AI agents, that turn a CSV into a reviewed,
production-ready model. You make the final call.**

Give it a CSV and a goal. An analyst, an engineer and a skeptic explore the data, build
features, train and compare models, and review each other's work. They ask you only
when a decision matters. The winning model is versioned and served on `/predict`.

Built with Google's [Agent Development Kit (ADK)](https://adk.dev/) and Gemini,
scaffolded with [Agents CLI](https://google.github.io/agents-cli/), and deployable to
Google Cloud with Terraform and GitHub Actions.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Development](#development)
- [Documentation](#documentation)
- [Author](#author)
- [License](#license)

## Features

- **A full ML team.** An analyst profiles the data, an engineer builds features and
  trains models, and a skeptic looks for leakage, unfair features and weak models.
- **Code behind every decision.** Agents don't decide from what they "see" in the data.
  They write and run Python and decide from what it prints, so every number can be
  rerun.
- **Independent scoring.** No agent scores its own model. The harness fits a fresh copy
  of every candidate on the training rows and scores them all on the same unseen test
  rows.
- **Business-aware model choice.** By default, models are ranked by what their mistakes
  cost per 1,000 records. A model goes live only if it beats doing nothing and the live
  model.
- **Fairness gate.** Approval rates are compared across protected attributes. A gap
  below the four-fifths rule is a warning, and that model never goes live without a
  person.
- **Human in the loop.** Each stage gets one skeptic review, and you decide at the gates
  you choose. Every decision is logged with who made it.
- **Contained agent code.** Scripts run in a separate process without the server's
  environment variables, so they can't read API keys.
- **Runs build on each other.** Each run keeps a full trace, versioned features and
  versioned models, and the next run starts from the last run's code and your feedback.
- **Use case agnostic.** Nothing about the example problem is written into the agents.
  Only `config/config.yml` and the data are problem-specific.

The repository ships with one example: **decide, when a home-loan application is
submitted, whether it will be approved.**

## How it works

The harness fixes the order of the steps and the rules. The agents fill in the work.

```text
data ─► Analyst ─► Engineer ─► Skeptic ─► ✋ review features ─► feature store
                   (features)
     ─► training plan ─► Engineer ─► harness scores ─► Skeptic ─► ✋ go live ─► model registry
                         (models)    every model
```

| Stage | Who | What happens |
| --- | --- | --- |
| Understand the data | Analyst | Profiles every column and summarises what matters. |
| Engineer features ✋ | Engineer, Skeptic, you | The engineer writes the features, the skeptic reviews them once, and you continue or send fixes back. |
| Choose models | Engineer | Proposes the models and the metric that picks the winner. |
| Train and test | Engineer, Skeptic | The engineer trains, the harness scores every model, and the skeptic reviews once. |
| Go live ✋ | You | Put the winner live, keep it, send it back with fixes, or discard the run. |

✋ waits for you by default. Set which gates wait in `autonomy.ask_human`.

See [docs/agents.md](docs/agents.md) for the agents, their tools and how ADK runs them.

## Quick start

### Prerequisites

- **Gemini access**, one of:
  - a [Google AI Studio API key](https://aistudio.google.com/apikey). This is the
    easiest option and needs no Google Cloud project.
  - a Google Cloud project with Vertex AI. It uses your `gcloud` login, so no key is
    stored.
- **Docker**, or [uv](https://docs.astral.sh/uv/) and Node.js 24 to run without Docker.

You can also use a local or cloud model through Ollama. See
[Configuration](#configuration).

### 1. Clone

```bash
git clone https://github.com/kmeanskaran/auto-ml-adk.git
cd auto-ml-adk
cp .env.example .env
```

### 2. Connect Gemini

Edit `.env` and choose one option.

**Option A: Google AI Studio key.** Comment out the three Vertex lines and set the key:

```dotenv
# GOOGLE_GENAI_USE_VERTEXAI=true
# GOOGLE_CLOUD_PROJECT=your-gcp-project-id
# GOOGLE_CLOUD_LOCATION=global
GEMINI_API_KEY=your-api-key
```

**Option B: Vertex AI.** Keep the Vertex lines, set your project ID, and log in once:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project your-gcp-project-id
gcloud services enable aiplatform.googleapis.com --project your-gcp-project-id
```

`.env` is gitignored, so your key or project ID is never committed.

### 3. Run

```bash
./run.sh
```

This builds and starts the backend and the console, then follows the agents' log.

| Service | URL |
| --- | --- |
| Console | <http://localhost:3000> |
| Backend API and `/predict` | <http://localhost:8000> |
| ADK dev UI | <http://localhost:8000/dev-ui> |

Open the console and press **Run pipeline**. The run pauses at the features review and
at the go-live decision for you to answer.

## Usage

### Commands

```bash
./run.sh          # build and start, then follow the agents' log
./run.sh logs     # follow the log again
./run.sh status   # show what is running
./run.sh down     # stop
./run.sh local    # run without Docker: backend with uv, console with npm
```

Runs, models and features are written to `runs/`, `registry/` and `feature_store/` on
your disk. All three are gitignored.

### Score records

Once a model is live, `POST /predict` scores raw records with it.
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

| Field | Meaning |
| --- | --- |
| `score` | The probability of the outcome (here, approval). |
| `flag` | 1 when `score` reaches the cut-off set by the costs: `false_alarm / (false_alarm + missed_positive)`, which is 2 / (2 + 1) = 0.67 in the example. |
| `ignored` | Fields you sent that the model doesn't use, such as `Gender` and `Married` once the skeptic has had them removed. |

## Configuration

`config/config.yml` is the only config file. Changes apply from the next run, with no
rebuild.

| Key | What it sets |
| --- | --- |
| `goal`, `dataset`, `traffic`, `target`, `positive_value` | The problem, the training CSV and the production CSV in `data/`, and the outcome column. |
| `prediction_moment` | What may be known when a prediction is made. The skeptic checks features against it. |
| `costs` | The price of each kind of mistake, used to rank models and set the cut-off. |
| `fairness` | Protected attributes and the minimum approval-rate ratio. |
| `autonomy.ask_human` | Which decisions wait for you: `features`, `plan`, `promote`. |
| `models` | One Gemini model for every agent, the thinking level, and prompt caching. The console's model menu switches it for all of them. |
| `limits` | Tool calls per turn (fewer for the skeptic), fix rounds, and the script timeout. |

### Use your own data

1. Put `train.csv` (with the outcome column) and `test.csv` (same columns, no outcome)
   in `data/<name>/`.
2. Update `dataset`, `traffic`, `target`, `positive_value` and `fairness` in
   `config/config.yml`.

### Use another model

Set `ML_MODEL` to any [LiteLLM](https://docs.litellm.ai/) model:

```bash
ML_MODEL=ollama_chat/gpt-oss:120b-cloud ./run.sh
```

## Deployment

| Part | Runs on |
| --- | --- |
| Backend: ADK agents, harness, `/predict` | Agent Runtime (Vertex AI) |
| Console: Next.js | Cloud Run |
| Runs, models, features | Cloud Storage |
| Infrastructure | Terraform, in `deployment/terraform/` |
| CI/CD | GitHub Actions, signed in with Workload Identity Federation (no stored keys) |
| Tracing | Cloud Trace and BigQuery |

Deploying needs `gcloud`, Terraform ≥ 1.11, `gh` and Agents CLI. The whole setup is one
command:

```bash
REPO=<you>/<repo> scripts/restart_gcp.sh <your-project-id>
```

- [docs/gcp-deployment.md](docs/gcp-deployment.md) explains the architecture and how to
  deploy your own copy, step by step.
- [SETUP.md](SETUP.md) is the command-by-command log of how this repository was built
  and deployed.

## Project structure

```text
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
uv sync                          # install
uv run pytest tests/unit         # unit tests
uvx ruff check app tests         # lint
uvx pre-commit install           # run the same checks on every commit

cd frontend && npm ci && npm run typecheck && npm run build   # console
```

CI runs the same pre-commit checks and the console build on every pull request.

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/agents.md](docs/agents.md) | The agents, their tools, the pipeline, and how ADK runs them |
| [docs/gcp-deployment.md](docs/gcp-deployment.md) | Agents CLI, Terraform, CI/CD, and deploying your own copy |
| [SETUP.md](SETUP.md) | The exact commands used to build this deployment |

## Author

Built by **Karan Shingde**. Questions, ideas and feedback are welcome.

- Email: [karanshingde@gmail.com](mailto:karanshingde@gmail.com)
- X: [@kmeanskaran](https://x.com/kmeanskaran)

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Karan Shingde.
