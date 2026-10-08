# Deploying to Google Cloud

This page explains how the ML Team runs on Google Cloud, what Agents CLI and Terraform
each do, and how to deploy your own copy. For the exact commands used to build this
repository's deployment, see [SETUP.md](../SETUP.md).

## Architecture

```
                     GitHub Actions (push to main)
                        │ Workload Identity Federation, no keys
                        ▼
 browser ──► Cloud Run: console (Next.js)
                │ Google access token (console service account)
                ▼
             Agent Runtime: backend (ADK app, FastAPI)
                │            │                    │
                ▼            ▼                    ▼
        Gemini on Vertex AI  Cloud Storage        Cloud Trace, Cloud Logging,
                             <project>-state:     BigQuery (prompts and responses)
                             runs/, registry/,
                             feature_store/
```

| Part | Service | Defined in |
|---|---|---|
| Backend: agents, pipeline, `/api`, `/predict` | Agent Runtime (Vertex AI reasoning engine), 1 instance, 2 CPU, 4 GiB, scales to zero | `service.tf`, deployed by `agents-cli deploy` |
| Console | Cloud Run service `ml-team-console`, at most 1 instance | `.github/workflows/deploy.yml` |
| Team's work | Bucket `<project>-state`, versioned, public access prevented | `ml_team.tf` |
| Terraform state | Bucket `<project>-tfstate` | Created by `gcloud` before Terraform, in `backend.tf` |
| Telemetry | Logs bucket, BigQuery dataset and connection | `storage.tf`, `telemetry.tf` |
| Identities | `ml-team-app` (agent), `ml-team-console`, `ml-team-deployer` (CI) | `iam.tf`, `ml_team.tf` |
| CI sign-in | Workload Identity pool `github`, limited to this repo's `main` branch | `ml_team.tf` |

## The three tools

| Tool | Its job here |
|---|---|
| **ADK** | The agent framework. The app (`app/agent.py`) is an ADK `App` whose root agent is the pipeline `Workflow`, and `app/fast_api_app.py` serves it. See [agents.md](agents.md). |
| **Agents CLI** | Scaffolds the project, adds the deployment target and Terraform base, and deploys the code to Agent Runtime. |
| **Terraform** | Creates everything the code runs on: APIs, service accounts, buckets, the agent resource, telemetry, and the GitHub sign-in. It is applied by hand, and CI never runs it. |

### Agents CLI

[Agents CLI](https://google.github.io/agents-cli/) (`google-agents-cli`, version 1.7.0
here) wraps the ADK project lifecycle and installs skills for coding agents.

```bash
uvx google-agents-cli setup          # installs agents-cli and its skills
agents-cli create ml-team -i         # ADK template, A2A on: server, Dockerfile, tests
agents-cli install                   # uv sync into .venv
agents-cli scaffold enhance . -d agent_runtime \
  --session-type agent_platform_sessions --cicd-runner skip -y
                                     # adds Agent Runtime: Terraform base, adapter
agents-cli deploy ...                # uploads the source to Agent Runtime
agents-cli eval run                  # runs tests/eval against the agent
```

`agents-cli-manifest.yaml` records what was scaffolded: the agent directory (`app`), the
root agent name (`ml_team`), the deployment target (`agent_runtime`) and the region.
`agents-cli deploy` reads it.

`--cicd-runner skip` means Agents CLI did not generate a pipeline. The workflow in
`.github/workflows/deploy.yml` and `ml_team.tf` were written on top of the scaffold.

### How the ADK app runs on Agent Runtime

Agent Runtime takes no prebuilt image. `agents-cli deploy` uploads the source and Google
builds it with the `Dockerfile` (the classic builder, so no BuildKit features). The
container runs `uvicorn app.fast_api_app:app` on port 8080, the same server you run
locally.

- `app/app_utils/reasoning_engine_adapter.py` serves Agent Runtime's
  `{class_method, input}` contract, so the Vertex AI console playground and Gemini
  Enterprise can call the agent.
- Agent Runtime forwards `…/reasoningEngines/<id>/api/*` to the server, which is how the
  console reaches `/api` and `/predict`. Callers need `roles/aiplatform.user`.
- A container's disk is lost on a redeploy or scale-down. `app/harness/state_sync.py`
  restores `runs/`, `registry/` and `feature_store/` from `ML_STATE_BUCKET` at start,
  pushes changes regularly, and pushes once more at shutdown. Only one instance may
  write the bucket, so `max_instances = 1`.
- Sessions use Agent Platform sessions, so a run paused at a gate survives a restart.

### Terraform

`deployment/terraform/ml-team-adk/` contains:

| File | Creates |
|---|---|
| `providers.tf`, `backend.tf` | Providers, and state in the `<project>-tfstate` bucket (given at `terraform init`). |
| `apis.tf` | Enables the APIs. Service Usage and Resource Manager are enabled first through a bootstrap provider. |
| `iam.tf` | The agent's service account `ml-team-app` and its roles (`variables.tf`: `app_sa_roles`). |
| `service.tf` | The Agent Runtime resource, with a placeholder source, scaling, and environment variables. |
| `storage.tf`, `telemetry.tf` | The logs bucket, a BigQuery dataset, and a connection for prompt and response logs. |
| `ml_team.tf` | The state bucket (imported, `prevent_destroy`), the console and deployer service accounts, the Artifact Registry repo, Workload Identity for GitHub, and the `github_variables` output. |
| `vars/env.tfvars` | Project ID, region, GitHub repo, state bucket. Committed, with no secrets. |
| `console.auto.tfvars` | `console_users`, your email. **Gitignored**, so you create it yourself. |

**Terraform creates the agent and CI deploys the code.** `service.tf` creates the
reasoning engine with a placeholder source (`shared/dummy_source.b64`) and then ignores
changes to the source, deployment spec and methods (`lifecycle.ignore_changes`). So
`terraform apply` never reverts what `agents-cli deploy` uploaded, and CI never needs
Terraform permissions.

**CI sign-in has no keys.** GitHub Actions exchanges its OIDC token for a short-lived
token for `ml-team-deployer`. The provider's `attribute_condition` accepts only
`assertion.repository == github_repository` on `refs/heads/main`, so forks, pull
requests and other branches can't deploy.

## CI/CD

`.github/workflows/deploy.yml` contains one workflow with three jobs:

```
push to main ──► checks ──► build ──► deploy
pull request ──► checks
```

| Job | Runs | Cloud access |
|---|---|---|
| `checks` | pre-commit (ruff, private-key check, unit tests, console typecheck) and the console's production build | none |
| `build` | Builds the backend image to test the Dockerfile (not pushed). Pushes the console image to Artifact Registry as `ml-team-console:<commit>`. | deployer |
| `deploy` | `agents-cli deploy` (backend), `gcloud run deploy` (console), then waits for the backend's `/api/view` to answer | deployer |

The workflow needs no keys. `scripts/set_github_vars.sh` sets what it reads from
`terraform output github_variables`:

| Kind | Names |
|---|---|
| Repository variables | `GCP_PROJECT_ID`, `GCP_REGION`, `DEPLOYER_SA`, `APP_SA`, `CONSOLE_SA`, `STATE_BUCKET`, and optionally `ML_MODEL` |
| Repository secrets | `WIF_PROVIDER`, `AGENT_ENGINE_ID` |

The two secrets aren't keys either. They are secrets so that GitHub masks them in the
public logs: `WIF_PROVIDER` contains the project number, which the console's Cloud Run
URL is built from, and `AGENT_ENGINE_ID` identifies the agent.

## Deploy your own copy

### 1. Prerequisites

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx google-agents-cli setup
brew install node gh hashicorp/tap/terraform   # Terraform ≥ 1.11
brew install --cask gcloud-cli
```

You also need a Google Cloud billing account and a GitHub repository you can push to,
such as a fork of this one.

### 2. Point the repo at your project and repository

| File | Change |
|---|---|
| `deployment/project.env` | `PROJECT_ID` (a new, globally unique ID) and `REGION` |
| `deployment/terraform/ml-team-adk/vars/env.tfvars` | `github_repository = "<you>/<repo>"`. Also `project_id`, `region` and `state_bucket = "<project>-state"` if you follow the manual path below. |
| `deployment/terraform/ml-team-adk/console.auto.tfvars` | Create it: `console_users = ["user:you@example.com"]` |

`github_repository` controls who can deploy to your project. If you leave it as this
repository, your CI can't sign in.

### 3a. One command

`scripts/restart_gcp.sh` creates the project, links billing, enables the APIs, creates
both buckets, writes the ID into `project.env`, `env.tfvars` and `.env`, runs Terraform,
sets the GitHub variables, and starts the deploy workflow.

```bash
gcloud auth login && gcloud auth application-default login
gh auth login
REPO=<you>/<repo> scripts/restart_gcp.sh <your-project-id>
```

It uses your first open billing account unless you set `BILLING_ACCOUNT=...`. Each step
skips what already exists, so you can rerun it after a failure.

### 3b. Step by step

```bash
export PROJECT_ID=<your-project-id> REGION=us-east1

# Project
gcloud projects create $PROJECT_ID
gcloud billing projects link $PROJECT_ID --billing-account=<ACCOUNT_ID>
gcloud config set project $PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
agents-cli login -i                      # choose Google Cloud

# Buckets Terraform expects to exist
gcloud storage buckets create gs://$PROJECT_ID-state --location $REGION \
  --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets create gs://$PROJECT_ID-tfstate --location $REGION \
  --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://$PROJECT_ID-tfstate --versioning

# Infrastructure (~10 min: creating the agent is the slow part)
cd deployment/terraform/ml-team-adk
terraform init -backend-config="bucket=$PROJECT_ID-tfstate"
terraform plan -var-file=vars/env.tfvars -out=tfplan
terraform apply tfplan
cd -

# GitHub variables, then deploy
scripts/set_github_vars.sh --dry-run
scripts/set_github_vars.sh
git push origin main && gh run watch
```

### 4. Open it

Open the console from Cloud Run (`ml-team-console`) in your Google Cloud project. Don't
post its URL anywhere in the repository: see [Keeping app URLs out of the
repository](#keeping-app-urls-out-of-the-repository).

## Access to the console

Terraform prepares IAP for the console: it creates IAP's service identity and grants
`console_users` the IAP accessor role. However, the deploy step currently runs
`gcloud run deploy ... --allow-unauthenticated --no-iap`. **The deployed console is open
to anyone with its URL**, and anyone who opens it can start runs that use your Gemini
quota. To restrict access, replace those two flags with `--no-allow-unauthenticated
--iap` in `deploy.yml`.

The backend is never public. Agent Runtime only accepts callers with
`roles/aiplatform.user`, and the console calls it as `ml-team-console`.

## Keeping app URLs out of the repository

The repository is public, and so are its GitHub Actions logs and run summaries. No URL
of the deployed app (the console or the agent's endpoint) is committed or printed:

- The project number and the agent's ID are repository secrets, which GitHub masks.
  Each job also masks the project number on its own, since some tools print it.
- The deploy job builds the agent's endpoint inside a step and masks it with
  `::add-mask::`, so it never appears in a log.
- The output of `agents-cli deploy` and `gcloud run deploy` passes through `hide_urls`,
  which replaces every URL with `<url hidden>`.
- Nothing about the deployment is written to the run summary except the commit, and
  the Docker build jobs upload no build-record artifacts.
- Don't paste the console URL into issues, pull requests, docs or commit messages.


| Task | How |
|---|---|
| Follow a deploy | `gh run watch` |
| Agent logs | Cloud Logging, resource type `aiplatform.googleapis.com/ReasoningEngine` |
| Traces | Cloud Trace, service `ml-team` |
| Prompts and responses | BigQuery dataset `ml_team_telemetry`, and `gs://<project>-ml-team-logs/completions` |
| A run's own trace | `runs/<run>/logs/` in `gs://<project>-state` |
| Change the model | `ML_MODEL` repository variable, or `models.team` in `config/config.yml`, then push |
| Test the agent | `agents-cli eval run` (see `tests/eval/datasets/README.md`) |

## Tear down and rebuild

```bash
scripts/nuke_gcp.sh                  # dry run: lists what would be deleted
scripts/nuke_gcp.sh --apply          # backs up the state bucket to backups/, deletes the project
scripts/restart_gcp.sh <new-id> --restore backups/<dir>
```

`nuke_gcp.sh` deletes the whole Google Cloud project, which stops all billing. GitHub is
left as it is. A deleted project can be restored for 30 days with `gcloud projects
undelete`, and its ID can never be reused, so a rebuild needs a new ID.
