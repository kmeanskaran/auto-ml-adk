# ML Team

A team of AI agents that works like a small machine learning team. Give it a CSV and a
goal; an analyst, an engineer and a skeptic explore the data, build features, train and
compare models, review each other's work, and ask you only when a decision matters. The
winning model is versioned and served on `/predict`.

Built on Google's [Agent Development Kit (ADK)](https://adk.dev/) with Gemini, scaffolded
with [Agents CLI](https://google.github.io/agents-cli/), and deployed to Agent Runtime
and Cloud Run with Terraform and GitHub Actions.

The example problem: **decide, when a home-loan application is submitted, whether it
will be approved.** Nothing about loans is in the agents. Only `config/config.yml` and
the data are loan-specific.

## How it works

The agents don't decide by reading the data themselves. They write and run code that
reads it, and decide from what that code finds. The harness fixes the order of the
steps and the rules; the agents fill in the work.

| Stage | Who | What happens |
|---|---|---|
| Understand the data | Analyst | Profiles every column and summarises what matters. |
| Engineer features ✋ | Engineer, Skeptic, You | The engineer writes the features; the skeptic reviews once; you continue or send fixes back. |
| Choose models | Team | The engineer proposes the models and the metric that picks the winner. |
| Train and test | Engineer, Skeptic | The engineer trains; the harness scores every model; the skeptic reviews once. |
| Go live ✋ | You | Put the winner live, keep it, send it back with fixes, or discard the run. |

✋ = waits for you by default (`autonomy.ask_human` in `config/config.yml`).

**The rules the harness enforces:**

- **No agent scores its own model.** The harness retrains every candidate, picks the
  winner on cross-validation, then tests it once on rows no model has seen.
- **Mistakes have a price.** Models are ranked by the cost of their mistakes per 1,000
  applications (a wrong approval costs twice a missed one) and must beat doing nothing.
- **Fair lending.** Approval rates are compared across `Gender` and `Married`. A gap
  below the four-fifths rule is a warning, and that model never goes live without a person.
- **One review, then you.** The skeptic reviews each stage once; you are the second
  reviewer. Every decision, the team's or yours, is logged with who made it.
- **Agent code is contained.** Scripts run in a separate process with no access to the
  server's secrets.

**What every run keeps:** a full trace (`runs/<run>/logs/`), versioned features
(`feature_store/`), versioned models (`registry/`), and a run history the next run
builds on.

**Keeping the LLM bill down:** agents get a briefing instead of re-reading reports, a
tool budget per turn, trimmed history on long turns, and Gemini context caching for each
role's fixed instructions.

## Quick start

Needs Docker (or [uv](https://docs.astral.sh/uv/) and Node for `./run.sh local`).

```bash
cp .env.example .env    # your Google Cloud project, or a GEMINI_API_KEY
./run.sh                # backend on :8000, console on http://localhost:3000
```

Open the console and press **Run pipeline**.

```bash
./run.sh logs           # follow the agents' log
./run.sh status         # what is running
./run.sh down           # stop
```

## Scoring applications

Once a model is live, `POST /predict` scores raw applications with it.
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

`score` is the chance of approval; `flag` is 1 when it reaches the threshold the harness
chose to minimise the cost of mistakes.

## Deploying to Google Cloud

- **Backend** (ADK agents, harness, `/predict`) runs on **Agent Runtime**.
- **Console** (Next.js) runs on **Cloud Run**.
- **Infrastructure** is Terraform in `deployment/terraform/`; runs, models and features
  persist in a Cloud Storage bucket.
- **CI/CD**: `.github/workflows/deploy.yml` runs checks, builds, and deploys on every push
  to `main`, signing in with Workload Identity (no stored keys).
- **Tracing**: every run lands in Cloud Trace; prompts and responses go to BigQuery.

The full setup, from an empty Google Cloud project to a live deploy, is in
[SETUP.md](SETUP.md).

## Configuration

`config/config.yml` is the one config file: the dataset and target, the business costs of
mistakes, the protected attributes, which decisions wait for you, the Gemini models and
the per-turn limits. Changes apply from the next run.

## Project layout

```
app/
  agent.py           the ADK App: workflow, plugins, context caching
  pipeline.py        the stages, reviews and routing
  agents.py          analyst, engineer and skeptic
  harness/           evaluation, feature store, model registry, tools, tracing
  ui/                console API (/api) and serving (/predict)
frontend/            the console (Next.js)
config/config.yml    the one config
data/lending-loan/   example data: train.csv and test.csv
deployment/          Terraform and the project ID
tests/               unit, integration and eval tests
```

## Development

```bash
uv run pytest tests/unit                               # unit tests
uvx ruff check app tests                               # lint
(cd frontend && npm run typecheck && npm run build)    # console
```
