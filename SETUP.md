# Setup

The commands and the flow used to build and deploy this project, in order. Claude Code,
with the Agents CLI skills, ran most of them; the human made the choices.

```
setup ──► create ──► run locally ──► Google Cloud project ──► scaffold enhance
      ──► terraform apply ──► GitHub variables ──► git push main ──► CI: checks → build → deploy
```

Values (the project lives in `deployment/project.env`, the one place to change it):

```bash
export PROJECT_ID=ml-team-adk-2
export REGION=us-east1
export STATE_BUCKET=$PROJECT_ID-state   # runs, model registry, feature store
export REPO=kmeanskaran/auto-ml-adk
```

---

## 1. Tools

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv: Python and its packages
uvx google-agents-cli setup                       # agents-cli + its skills for the coding agent
brew install node gh                              # console, GitHub
brew install --cask gcloud-cli                    # Google Cloud
brew install hashicorp/tap/terraform              # ≥ 1.11 (Homebrew's own stops at 1.5.7)
```

Plus Docker Desktop. Check Agents CLI:

```bash
agents-cli --version       # 1.7.0
ls ~/.agents/skills        # google-agents-cli-adk-code, -deploy, -eval, -observability, …
```

---

## 2. Scaffold the project

```bash
agents-cli create ml-team -i     # base template adk, no deployment target yet, A2A on, GEMINI.md
agents-cli install               # uv sync into .venv
```

This gave the server (`app/fast_api_app.py`), the session and artifact services, the
Dockerfile and the tests. The ML team itself is in `app/`.

---

## 3. Run locally

```bash
cp .env.example .env     # Gemini on Vertex AI: see part 4
./run.sh                 # console http://localhost:3000, backend http://localhost:8000
./run.sh logs            # follow the agents' log
./run.sh down            # stop
```

Checks, the same ones CI runs:

```bash
uvx pre-commit install              # once: ruff, private-key check, unit tests, console typecheck on every commit
uvx pre-commit run --all-files
```

---

## 4. Google Cloud project

```bash
gcloud auth login
gcloud projects create $PROJECT_ID --name="ML Team"
gcloud billing projects link $PROJECT_ID --billing-account=<ACCOUNT_ID>
gcloud config set project $PROJECT_ID

gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
agents-cli login -i                 # choose Google Cloud

gcloud services enable \
  aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com logging.googleapis.com cloudtrace.googleapis.com \
  telemetry.googleapis.com run.googleapis.com cloudresourcemanager.googleapis.com \
  iam.googleapis.com serviceusage.googleapis.com

gcloud storage buckets create gs://$STATE_BUCKET --location $REGION \
  --uniform-bucket-level-access --public-access-prevention
```

`.env` for Gemini on Vertex AI (no API key: it uses the login above):

```dotenv
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=ml-team-adk-2
GOOGLE_CLOUD_LOCATION=global
OTEL_TO_CLOUD=false
```

---

## 5. Add the deployment

```bash
agents-cli scaffold enhance . -d agent_runtime --session-type agent_platform_sessions --cicd-runner skip -y
agents-cli install               # updates uv.lock
```

This added Agent Runtime as the deployment target, the Terraform base
(`deployment/terraform/`) and the Agent Runtime adapter. The CI/CD workflow
(`.github/workflows/deploy.yml`) and `ml_team.tf` (state bucket, console and deployer
service accounts, Workload Identity for GitHub) were written on top.

---

## 6. Cloud infrastructure

```bash
gcloud storage buckets create gs://$PROJECT_ID-tfstate --location $REGION \
  --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://$PROJECT_ID-tfstate --versioning

cd deployment/terraform/ml-team-adk
echo 'console_users = ["user:you@gmail.com"]' > console.auto.tfvars
terraform init -backend-config="bucket=$PROJECT_ID-tfstate"
terraform plan -var-file=vars/env.tfvars -out=tfplan
terraform apply tfplan                  # ~10 min: creates the agent on Agent Runtime
terraform output github_variables       # for part 7
```

---

## 7. GitHub variables

No secrets: GitHub signs in to Google Cloud with Workload Identity.

```bash
gh auth login
scripts/set_github_vars.sh --dry-run    # show what it will set
scripts/set_github_vars.sh              # set them from terraform output
gh variable list --repo $REPO
```

---

## 8. Deploy

```bash
git push origin main
gh run watch
```

Every push to `main` runs one workflow:

```
push to main ──► checks ──► build ──► deploy
pull request ──► checks
```

| Job | Runs |
|-----|------|
| `checks` | the pre-commit hooks, the console's production build |
| `build` | console image → Artifact Registry; backend image built only to check the Dockerfile |
| `deploy` | `agents-cli deploy` (backend → Agent Runtime), `gcloud run deploy ml-team-console` (console → Cloud Run), then checks the backend answers |

The deploy command CI runs:

```bash
agents-cli deploy --project $PROJECT_ID --region $REGION --no-confirm-project \
  --service-account ml-team-app@$PROJECT_ID.iam.gserviceaccount.com \
  --update-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_LOCATION=global,ML_STATE_BUCKET=$STATE_BUCKET,OTEL_TO_CLOUD=true
```

The console's URL:

```bash
gcloud run services describe ml-team-console --region $REGION --format='value(status.url)'
```

---

## Tear down and rebuild

The first project, `ml-team-adk`, was built by hand with parts 4–8, then deleted on
2026-10-03. `ml-team-adk-2` was built with the restart script.

```bash
scripts/nuke_gcp.sh                     # dry run: lists what would go
scripts/nuke_gcp.sh --apply             # backs up the state bucket to backups/, deletes the project

scripts/restart_gcp.sh ml-team-adk-2   # add --restore backups/<dir> to bring back runs and models
```

`restart_gcp.sh` does parts 4, 6, 7 and 8 in one go: creates the project and buckets,
writes the ID into `deployment/project.env`, `vars/env.tfvars` and `.env`, runs
Terraform, restores the backup, sets the GitHub variables and starts the deploy
workflow. A deleted project's ID can't be reused, so each rebuild needs a new one.
