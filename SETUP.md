# Setup

Everything from an empty machine to the ML team running on Gemini in Google Cloud, in
the order it was done.

| Part | What | Status |
|------|------|--------|
| 1 | Tools | ✅ |
| 2 | The project: scaffolded with Agents CLI | ✅ |
| 3 | Run it locally | ✅ |
| 4 | Google Cloud project | ✅ |
| 5 | Gemini on Vertex AI, locally | ✅ |
| 6 | Deployment code: Agent Runtime scaffold, Terraform, CI/CD, pre-commit | ✅ written and checked |
| 7 | Cloud infrastructure: `terraform apply` | ✅ agent `ml-team` created |
| 8 | GitHub: repository variables | ✅ |
| 9 | First deploy and checks | ✅ deployed (run 37033235793); checks in progress |

The commands use these values. To set up your own copy, change them once here and use
the same names below:

```bash
export PROJECT_ID=ml-team-adk          # globally unique: pick your own
export REGION=us-east1                 # where the agent and the buckets live
export STATE_BUCKET=ml-team-adk-state  # runs, model registry, feature store
export REPO=kmeanskaran/auto-ml-adk    # GitHub owner/name
```

---

## 1. Tools

| Tool | Why | Install (macOS) |
|------|-----|-----------------|
| [uv](https://docs.astral.sh/uv/) | Python and its packages | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Agents CLI](https://google.github.io/agents-cli/) | scaffold, run, eval and deploy ADK agents | `uvx google-agents-cli setup` |
| Docker Desktop | the backend and the console as containers | [docker.com](https://docs.docker.com/get-docker/) |
| Node.js 24 | the console, for `./run.sh local` | `brew install node` |
| gcloud | Google Cloud | `brew install --cask gcloud-cli` |
| gh | GitHub | `brew install gh` |
| Terraform ≥ 1.11 | the cloud setup as code | `brew install hashicorp/tap/terraform` |
| [Ollama](https://ollama.com/) (optional) | run the agents on a non-Google model | `brew install ollama` |

Homebrew's own `terraform` stopped at 1.5.7; the scaffolded Terraform needs 1.11 or
later, hence HashiCorp's tap (`brew uninstall terraform` first if the old one is there).

`uvx google-agents-cli setup` installs `agents-cli` (1.7.0 here) and gives your coding
agent (Claude Code, Gemini CLI, Antigravity, Cursor…) the Agents CLI skills: how to
write ADK code, scaffold, evaluate, deploy, observe and publish.

```bash
agents-cli --version
ls ~/.agents/skills        # google-agents-cli-adk-code, -deploy, -eval, -observability, …
```

---

## 2. The project

You don't need this part if you cloned the repo; it records how the project began.

```bash
agents-cli create ml-team -i
agents-cli install            # uv sync into .venv
```

| Choice | Value | Why |
|--------|-------|-----|
| base template | `adk` | a plain ADK (Python) agent |
| deployment target | `none` | added in part 6 |
| session type | `in_memory` | local first; the code uses SQLite locally, Agent Platform Sessions deployed |
| A2A | on | the agent also speaks A2A at `/a2a/app` |
| guidance file | `GEMINI.md` | instructions for coding agents |

The scaffold gave the server (`app/fast_api_app.py`), the session and artifact services
(`app/app_utils/services.py`), the Dockerfile, the tests and the eval config. The ML
team, an ADK 2.8 `Workflow` with an analyst, an engineer and a skeptic, is in `app/`;
see the README.

---

## 3. Run it locally

```bash
git clone https://github.com/$REPO.git && cd auto-ml-adk
cp .env.example .env
```

The data comes with the repo: `data/lending-loan/train.csv` (614 labelled loan
applications) and `test.csv` (367 unlabelled ones, playing production traffic).

Pick the model the agents use, in `.env`:

| Option | `.env` | Needs |
|--------|--------|-------|
| **Gemini on Vertex AI** (this project) | `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT=…`, `GOOGLE_CLOUD_LOCATION=global` | a Google Cloud project: parts 4 and 5 |
| Gemini through AI Studio | `GEMINI_API_KEY=…`, the three Vertex lines commented out | a free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Ollama | `ML_MODEL=ollama_chat/gpt-oss:120b-cloud` | Ollama running, the model pulled |

The models are set in `config/config.yml` (`models:`). `ML_MODEL`, in the shell or
`.env`, overrides the team's.

```bash
./run.sh                 # console http://localhost:3000, backend http://localhost:8000
./run.sh logs            # follow the agents' log again
./run.sh down            # stop
./run.sh local           # no Docker: uv backend + Next.js dev server
```

Open the console, press **Run pipeline**, watch the Activity panel. Every run writes
`runs/<run>/logs/`: `activity.log` (one line per step), `trace.jsonl` (the same steps
with arguments, results, seconds and tokens) and `code/` (every file an agent wrote).

### Tests and pre-commit

The checks live in `.pre-commit-config.yaml`, and CI runs the very same ones, so a
commit that passes here passes on GitHub:

```bash
uvx pre-commit install              # once: the checks run on every git commit
uvx pre-commit run --all-files      # run them all now
```

| Hook | Runs when | What |
|------|-----------|------|
| ruff check, ruff format | Python changed | lint (fixes what it can) and format |
| detect-private-key | always | no private keys committed |
| unit tests (`uv run pytest tests/unit`) | Python changed | 65 tests: harness, registry, feature store, workflow, the bucket sync, the scripts |
| console typecheck | `frontend/` changed | `tsc --noEmit` |

`git commit --no-verify` skips them in an emergency; CI still runs them.

By hand:

```bash
uv run pytest tests/unit                     # tests
uvx ruff check app tests scripts             # lint
npm --prefix frontend run typecheck          # console
npm --prefix frontend run build
```

---

## 4. Google Cloud project

```bash
gcloud auth login                                       # browser
gcloud projects create $PROJECT_ID --name="ML Team"
gcloud billing accounts list                            # note the ACCOUNT_ID
gcloud billing projects link $PROJECT_ID --billing-account=<ACCOUNT_ID>
gcloud config set project $PROJECT_ID

gcloud auth application-default login                   # browser: credentials for local code
gcloud auth application-default set-quota-project $PROJECT_ID
agents-cli login -i                                     # choose Google Cloud
agents-cli login --status

gcloud services enable \
  aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com secretmanager.googleapis.com logging.googleapis.com \
  cloudtrace.googleapis.com telemetry.googleapis.com run.googleapis.com \
  cloudresourcemanager.googleapis.com iam.googleapis.com

gcloud storage buckets create gs://$STATE_BUCKET --location $REGION \
  --uniform-bucket-level-access --public-access-prevention
```

A budget alert, in the console: **Billing → Budgets & alerts → Create budget**, scoped
to the project, $50, alerts at 50%, 90%, 100%. It emails you; it does not stop spending.

---

## 5. Gemini on Vertex AI, locally

`.env`:

```dotenv
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=ml-team-adk
GOOGLE_CLOUD_LOCATION=global
# GEMINI_API_KEY=…          off: Vertex wins when both are set
OTEL_TO_CLOUD=false
```

No key: Vertex uses the application-default login from part 4. The Docker backend gets
it read-only through the `~/.config/gcloud` mount in `docker-compose.yml`.

```bash
uv run python -c "
from google import genai
print(genai.Client().models.generate_content(model='gemini-3.8-flash', contents='Reply: Vertex OK').text)"
./run.sh
```

A full run on `gemini-3.8-flash` (`run-20261002-141111`):

| Agent | Time | Tool calls | Tokens | Cached |
|-------|------|------------|--------|--------|
| analyst | 168 s | 7 | 56,178 | 46% |
| feature engineer, round 1 | 84 s | 16 | 144,156 | 48% |
| skeptic: sends back "remove `Gender`" | 100 s | 6 | 40,675 | 44% |
| feature engineer, round 2 | 106 s | 14 | 125,136 | 41% |
| skeptic: continue | 100 s | 8 | 52,335 | 47% |
| model engineer | 343 s | 16 | 271,700 | 21% |
| model skeptic | 81 s | 12 | 172,540 | 27% |
| **total** | **~16 min** | **79** | **862,720** | |

The team settled the skeptic's concern itself and asked the human only at "Go live?".

---

## 6. Deployment code

```
 anyone ──► console (Cloud Run, open to all) ──Google token──► backend (Agent Runtime, 1 instance)
                                                     ├─ Gemini on Vertex AI
                                                     ├─ Agent Platform Sessions
                                                     ├─ gs://$STATE_BUCKET  (runs, registry, features)
                                                     └─ Cloud Trace, Cloud Logging, prompt logs
 git push main ──► GitHub Actions: checks ──► deploy backend, then console
```

One environment: a push to `main` that passes the checks is production. Your local
`./run.sh` is the staging. (A human approval before production, with a GitHub
environment and a required reviewer, is the next step up; the team's own "Go live?"
gate already guards the models.)

### 6.1 Agent Runtime scaffold

```bash
agents-cli scaffold enhance . -d agent_runtime --session-type agent_platform_sessions --cicd-runner skip -y
agents-cli install            # updates uv.lock: the image builds with uv sync --frozen
```

It added the Terraform base (`deployment/terraform/`), the Agent Runtime adapter
(`app/app_utils/reasoning_engine_adapter.py`: the Cloud Console playground and Gemini
Enterprise call the agent through it), the deployment target and two dependencies.
It kept our `app/fast_api_app.py`, so the adapter is connected there by hand
(`attach_reasoning_engine_routes(app)`).

### 6.2 What changed in the app

| File | Why |
|------|-----|
| `app/harness/state_sync.py` | A container's disk is lost on redeploy. With `ML_STATE_BUCKET` set, the server pulls `runs/`, `registry/`, `feature_store/` from the bucket at start, pushes changes every 10 s and at shutdown. Hence **one instance**. |
| `Dockerfile` | ships `data/lending-loan` (68 KB) in the image |
| `frontend/app/api/[...path]/route.ts` | when `BACKEND_URL` is Agent Runtime, adds a Google token: the console's service account on Cloud Run, your gcloud login on a laptop |

### 6.3 Terraform (`deployment/terraform/ml-team-adk/`)

| File | Creates |
|------|---------|
| scaffolded: `apis.tf`, `iam.tf`, `service.tf`, `storage.tf`, `telemetry.tf` | the APIs; the agent's service account `ml-team-app`; **the Agent Runtime agent itself** with placeholder code, which every `agents-cli deploy` replaces (Terraform never reverts it); a logs bucket and BigQuery tables for prompt-response logging |
| `service.tf` (edited) | size set once at creation: **2 vCPU, 4 GiB, 0 to 1 instance** (scales to zero when idle); `ML_STATE_BUCKET`, `OTEL_TO_CLOUD=true` |
| `ml_team.tf` (ours) | adopts the state bucket (versioned, protected from `destroy`) and lets the agent use it; the console's service account (may only call the agent); the deployer service account for GitHub Actions, allowed to deploy as the agent's and the console's accounts and to build as the default compute account, and a Workload Identity pool that trusts **only `$REPO`'s `main` branch**: no keys |
| `backend.tf` | Terraform's state in `gs://$PROJECT_ID-tfstate` |
| `vars/env.tfvars` | project, region, repository, state bucket |
| `console.auto.tfvars` (gitignored) | `console_users = ["user:you@gmail.com"]`: who may open the console |

### 6.4 CI/CD (`.github/workflows/deploy.yml`)

One workflow, three connected jobs; each starts only when the one before succeeded:

```
push to main ──► checks ──► build ──► deploy
pull request ──► checks   (never builds or deploys)
```

| Job | Does | Google Cloud |
|-----|------|--------------|
| `checks` | the pre-commit hooks (part 3), then the console's production build | no access |
| `build` | console image → Artifact Registry as `ml-team-console:<commit>`; backend image built to check the Dockerfile, not pushed (Agent Runtime takes no prebuilt image) | Workload Identity |
| `deploy` | backend → Agent Runtime (`agents-cli deploy`, Google builds it from source); console → Cloud Run from build's image, open to anyone, 1 instance; checks the backend answers | Workload Identity |

Run it by hand on `main`: `gh workflow run deploy`. GitHub shows the three jobs as one
connected graph under **Actions → deploy**.
---

## 7. Cloud infrastructure

```bash
gcloud storage buckets create gs://$PROJECT_ID-tfstate --location $REGION \
  --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://$PROJECT_ID-tfstate --versioning

cd deployment/terraform/ml-team-adk
echo 'console_users = ["user:you@gmail.com"]' > console.auto.tfvars
terraform init -backend-config="bucket=$PROJECT_ID-tfstate"
terraform plan -var-file=vars/env.tfvars -out=tfplan     # read it
terraform apply tfplan                                    # ~10 min: creating the agent is slow
terraform output github_variables                         # for part 8
```


The plan for this project: 59 to create, 1 to import (the state bucket), 1 change
(versioning on it), 0 to destroy.

What `terraform output` printed after the apply:

```hcl
agent_runtime_resource_name = "7461691625878585344"
app_service_account_email = "ml-team-app@ml-team-adk.iam.gserviceaccount.com"
github_variables = {
  "AGENT_ENGINE_ID" = "7461691625878585344"
  "APP_SA" = "ml-team-app@ml-team-adk.iam.gserviceaccount.com"
  "CONSOLE_SA" = "ml-team-console@ml-team-adk.iam.gserviceaccount.com"
  "DEPLOYER_SA" = "ml-team-deployer@ml-team-adk.iam.gserviceaccount.com"
  "GCP_PROJECT_ID" = "ml-team-adk"
  "GCP_REGION" = "us-east1"
  "STATE_BUCKET" = "ml-team-adk-state"
  "WIF_PROVIDER" = "projects/956236550460/locations/global/workloadIdentityPools/github/providers/github"
}
logs_bucket_name = "ml-team-adk-ml-team-logs"
telemetry_bigquery_connection_id = "ml-team-genai-telemetry"
telemetry_dataset_id = "ml_team_telemetry"
```

| Output | What it is |
|--------|------------|
| `agent_runtime_resource_name` | the agent on Agent Runtime, still running Terraform's placeholder code until the first deploy (part 9) |
| `app_service_account_email` | the identity the agent runs as: Gemini, the state bucket, logs, traces |
| `github_variables` | everything GitHub Actions needs, set as repository variables in part 8 |
| `logs_bucket_name` | prompts and responses (prompt-response logging), as JSONL under `completions/` |
| `telemetry_bigquery_connection_id`, `telemetry_dataset_id` | BigQuery over those logs, to query the agents' model calls with SQL |

These are identifiers, not secrets: none of them gives access without a Google login
that IAM allows. Your own copy prints its own values.

---

## 8. GitHub

```bash
gh auth login
```

Repository variables, straight from `terraform output github_variables`. They are IDs,
not secrets: with Workload Identity Federation there are no keys. Set them **before**
the first push of the workflow, or its first deploy fails.

```bash
scripts/set_github_vars.sh --dry-run                    # show them
ML_MODEL=gemini-3.8-flash scripts/set_github_vars.sh    # set them (ML_MODEL optional)
gh variable list --repo $REPO
```

Or by hand: **Settings → Secrets and variables → Actions → Variables**, one per line of
`terraform output github_variables`, plus `ML_MODEL` if you want to pin the team's model
(unset, `config/config.yml` decides).

---

## 9. First deploy and checks

Push to `main` and watch **Actions → deploy** (`gh run watch`). Or deploy the backend by
hand first, for faster feedback:

```bash
agents-cli deploy --service-account ml-team-app@$PROJECT_ID.iam.gserviceaccount.com --no-confirm-project \
  --update-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_LOCATION=global,ML_STATE_BUCKET=$STATE_BUCKET,OTEL_TO_CLOUD=true
```

Then check:
- `agents-cli run --url <agent url> --mode adk "hello"`, and the Agent Runtime
  playground in the Cloud Console
- the console, at its public URL (`gcloud run services describe ml-team-console --region $REGION --format='value(status.url)'`):
  open to anyone, no login. Anyone with the link can start runs and chat, which spends
  your Gemini budget: tear the demo down when it's over
- Cloud Trace shows the agents' spans
- redeploy, and the runs, the model registry and a paused review are all still there

---

## Remove everything

```bash
scripts/nuke_gcp.sh            # dry run: lists what would go, changes nothing
scripts/nuke_gcp.sh --apply    # asks you to type the project ID, then:
```

1. backs up `gs://$STATE_BUCKET` (runs, models, features) to `backups/` (skip with `--no-backup`)
2. disables the deploy workflow in GitHub and deletes its variables, so a push can't recreate anything
3. deletes the Google Cloud project, and with it the agent, its sessions, the console,
   every bucket, service accounts, Workload Identity, BigQuery, logs and traces;
   billing stops
4. removes Terraform's working files here and unsets gcloud's default project

A deleted project can be restored for 30 days (`gcloud projects undelete $PROJECT_ID`),
then it is gone and its ID can't be reused. The script lists what's left for you: the
project in `.env`, your login's quota project, and the budget alert (it lives on the
billing account).

---

## Environment variables at a glance

| Where | Set by | Variables |
|-------|--------|-----------|
| Your machine | `.env` | `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, optional `ML_MODEL`, `OTEL_TO_CLOUD=false`. Never `ML_STATE_BUCKET`. |
| Backend on Agent Runtime | Terraform at creation, `agents-cli deploy --update-env-vars` after | `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_LOCATION=global`, `ML_STATE_BUCKET`, `OTEL_TO_CLOUD=true`, `LOGS_BUCKET_NAME` and the `OTEL_*` telemetry settings, optional `ML_MODEL`. Agent Runtime sets `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_AGENT_ENGINE_ID` itself; sessions then use Agent Platform Sessions. |
| Console on Cloud Run | the workflow | `BACKEND_URL` = `https://$REGION-aiplatform.googleapis.com/reasoningEngines/v1/projects/$PROJECT_ID/locations/$REGION/reasoningEngines/<AGENT_ENGINE_ID>/api` |
| GitHub repository | `gh variable set` | the variables in part 8; no secrets |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `DefaultCredentialsError` | Vertex on (or `OTEL_TO_CLOUD=true`) without Google Cloud credentials | `gcloud auth application-default login`; for Docker, keep the `~/.config/gcloud` mount |
| Model 404 on Vertex | a regional `GOOGLE_CLOUD_LOCATION` | use `global`; don't change the model name |
| Agents on the wrong model | `ML_MODEL` in the shell beats `.env`, which beats `config.yml` | `unset ML_MODEL`, check `.env` |
| CI fails on a missing file that exists on your machine | a `.gitignore` rule hides it (the Python template's `lib/` once hid `frontend/lib/`) | `git check-ignore -v <file>`, then narrow the rule |
| Console: "This page couldn't load" | a JavaScript error in the console, not the server | browser developer console; `curl localhost:3000/api/view` shows whether the backend answers |
| `agents-cli deploy`: "No deployment target configured" | manifest says `deployment_target: none` | part 6.1 |
| Docker build fails at `uv sync --frozen` | `pyproject.toml` changed, `uv.lock` did not | `agents-cli install`, commit `uv.lock` |
| `terraform init`: unsupported Terraform version | Homebrew's terraform 1.5.7 | part 1: HashiCorp's tap |
| Deployed agent forgets runs and models | `ML_STATE_BUCKET` unset, or more than one instance | check the agent's env; it must run as one instance |
| GitHub deploy: "unable to impersonate" / 403 | not run from `main`, or the repository name differs from `github_repository` | `vars/env.tfvars`, then `terraform apply` |
| Console deploy: "caller does not have permission to act as service account" | Cloud Build builds `--source` deploys as the default compute account, and the deployer may not act as it | `deployer_acts_as["build"]` in `ml_team.tf`, then `terraform apply` |
| `--iap` on Cloud Run fails or warns about an organization | IAP on Cloud Run, for a project outside an organization, needs a consent screen and your own OAuth client, made by hand in the Cloud Console (Google has no API for them there) | this project skips IAP: the console is open to anyone |
| Console: "backend unreachable: Too Many Requests" | Agent Runtime's `/api` passthrough lets one caller through about 30 times a minute; each tab polls every 1.5 s | the console's server asks the agent at most every 3 s and shares the answer across tabs (`route.ts`), and runs as one instance |
| `terraform plan` wants to change `class_methods` on the agent | `agents-cli deploy` registers the agent's methods; Terraform would delete them | `spec[0].class_methods` in `ignore_changes` (`service.tf`) |
