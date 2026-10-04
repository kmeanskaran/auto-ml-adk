#!/usr/bin/env bash
# Build everything in Google Cloud again, A to Z: after scripts/nuke_gcp.sh (with a
# new project ID), or after a teardown that kept the project (same ID).
#
#   scripts/restart_gcp.sh                               the project in deployment/project.env
#   scripts/restart_gcp.sh ml-team-adk-2                 a new project ID (written to project.env)
#   scripts/restart_gcp.sh ml-team-adk-2 --restore backups/ml-team-adk-20261003-120000
#                                                        and bring back the saved runs and models
#   ML_MODEL=gemini-3.8-flash scripts/restart_gcp.sh     also pin the team's model on GitHub
#
# In order (each step skips what already exists, so a failed restart can simply be rerun):
#   1. project: create it, link billing, make it gcloud's default, enable the APIs
#   2. buckets: <id>-state (runs, models, features) and <id>-tfstate (Terraform's state)
#   3. the project ID into vars/env.tfvars and your .env
#   4. Terraform: service accounts, GitHub login, the agent, logs, telemetry (~10 min)
#   5. with --restore: the saved runs and models into <id>-state
#   6. GitHub: turn the deploy workflow on, set its variables
#   7. start the deploy workflow (checks → build → deploy); its summary shows the console URL
#
# Needs: gcloud (logged in, plus application-default login), terraform ≥ 1.11, gh
# (logged in), and deployment/terraform/ml-team-adk/console.auto.tfvars.
set -euo pipefail
cd "$(dirname "$0")/.."

TF_DIR=deployment/terraform/ml-team-adk
REPO="${REPO:-kmeanskaran/auto-ml-adk}"
APIS=(aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
  storage.googleapis.com logging.googleapis.com cloudtrace.googleapis.com
  telemetry.googleapis.com run.googleapis.com cloudresourcemanager.googleapis.com
  iam.googleapis.com serviceusage.googleapis.com)

step() { echo; echo "● $*"; }
item() { echo "    $*"; }
fail() { echo "✗ $*" >&2; exit 1; }

# shellcheck source=../deployment/project.env
source deployment/project.env
RESTORE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --restore) RESTORE="${2:-}"; shift 2 ;;
    -h | --help) sed -n '2,23p' "$0"; exit 0 ;;
    -*) fail "Unknown option $1 (see --help)" ;;
    *) PROJECT_ID="$1"; shift ;;
  esac
done
STATE_BUCKET="$PROJECT_ID-state"
TF_STATE_BUCKET="$PROJECT_ID-tfstate"

# --- Before anything changes -----------------------------------------------------------

for tool in gcloud terraform gh; do
  command -v "$tool" >/dev/null || fail "$tool is not installed (see SETUP.md part 1)."
done
gcloud auth print-access-token >/dev/null 2>&1 || fail "gcloud is not logged in: gcloud auth login"
gcloud auth application-default print-access-token >/dev/null 2>&1 ||
  fail "No application-default login: gcloud auth application-default login"
gh auth status >/dev/null 2>&1 || fail "gh is not logged in: gh auth login"
[ -f "$TF_DIR/console.auto.tfvars" ] ||
  fail "Missing $TF_DIR/console.auto.tfvars: echo 'console_users = [\"user:you@gmail.com\"]' > $TF_DIR/console.auto.tfvars"
[ -z "$RESTORE" ] || [ -d "$RESTORE" ] || fail "No backup folder $RESTORE (ls backups/)."

state="$(gcloud projects describe "$PROJECT_ID" --format='value(lifecycleState)' 2>/dev/null || true)"
[ "$state" = DELETE_REQUESTED ] &&
  fail "$PROJECT_ID was deleted and its ID can't be reused. Pick a new one: $0 <new-project-id>"
BILLING="${BILLING_ACCOUNT:-$(gcloud billing accounts list --filter=open=true --format='value(name.basename())' | head -n 1)}"
[ -n "$BILLING" ] || fail "No open billing account (gcloud billing accounts list)."

echo "Restart in Google Cloud:"
item "project        $PROJECT_ID ($([ "$state" = ACTIVE ] && echo exists || echo 'new'))"
item "region         $REGION"
item "billing        $BILLING"
item "buckets        gs://$STATE_BUCKET, gs://$TF_STATE_BUCKET"
item "restore        ${RESTORE:-no, start empty}"
item "GitHub         $REPO"
read -r -p "Type the project ID to go ahead: " answer
[ "$answer" = "$PROJECT_ID" ] || fail "Not confirmed; nothing was changed."

# --- 1. Project ------------------------------------------------------------------------

step "1. Project $PROJECT_ID"
if [ "$state" != ACTIVE ]; then
  gcloud projects create "$PROJECT_ID" --name="ML Team" ||
    fail "Could not create $PROJECT_ID (IDs are global: try another)."
fi
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING" >/dev/null
gcloud config set project "$PROJECT_ID" >/dev/null 2>&1
gcloud auth application-default set-quota-project "$PROJECT_ID" >/dev/null 2>&1
gcloud services enable "${APIS[@]}" --project "$PROJECT_ID"
item "billing linked, default project set, APIs on"

# --- 2. Buckets ------------------------------------------------------------------------

step "2. Buckets"
for bucket in "$STATE_BUCKET" "$TF_STATE_BUCKET"; do
  if gcloud storage buckets describe "gs://$bucket" >/dev/null 2>&1; then
    item "gs://$bucket exists"
  else
    gcloud storage buckets create "gs://$bucket" --project "$PROJECT_ID" --location "$REGION" \
      --uniform-bucket-level-access --public-access-prevention
  fi
done
gcloud storage buckets update "gs://$TF_STATE_BUCKET" --versioning >/dev/null

# --- 3. The project ID into the files that need it --------------------------------------

step "3. Project ID → deployment/project.env, $TF_DIR/vars/env.tfvars, .env"
sed -i.bak "s/^PROJECT_ID=.*/PROJECT_ID=$PROJECT_ID/" deployment/project.env
sed -i.bak -e "s/^project_id = .*/project_id = \"$PROJECT_ID\"/" \
  -e "s/^region = .*/region = \"$REGION\"/" \
  -e "s/^state_bucket *= .*/state_bucket      = \"$STATE_BUCKET\"/" "$TF_DIR/vars/env.tfvars"
[ -f .env ] || cp .env.example .env
if grep -q '^GOOGLE_CLOUD_PROJECT=' .env; then
  sed -i.bak "s/^GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=$PROJECT_ID/" .env
else
  echo "GOOGLE_CLOUD_PROJECT=$PROJECT_ID" >>.env
fi
rm -f deployment/project.env.bak "$TF_DIR/vars/env.tfvars.bak" .env.bak
item "done"

# --- 4. Terraform ----------------------------------------------------------------------

step "4. Terraform (creating the agent takes ~10 min)"
terraform -chdir="$TF_DIR" init -reconfigure -input=false -backend-config="bucket=$TF_STATE_BUCKET" >/dev/null
terraform -chdir="$TF_DIR" plan -input=false -var-file=vars/env.tfvars -out=tfplan | grep -E "^Plan:|Error" || true
terraform -chdir="$TF_DIR" apply -input=false tfplan
rm -f "$TF_DIR/tfplan"

# --- 5. Saved runs and models ----------------------------------------------------------

if [ -n "$RESTORE" ]; then
  step "5. Restoring $RESTORE → gs://$STATE_BUCKET"
  gcloud storage rsync --recursive "$RESTORE" "gs://$STATE_BUCKET"
else
  step "5. No --restore: the team starts empty (a cold start)"
fi

# --- 6. GitHub -------------------------------------------------------------------------

step "6. GitHub $REPO"
if gh workflow enable deploy --repo "$REPO" 2>/dev/null; then
  item "deploy workflow on"
else
  item "deploy workflow already on"
fi
REPO="$REPO" scripts/set_github_vars.sh

# --- 7. Deploy -------------------------------------------------------------------------

step "7. Starting the deploy workflow (checks → build → deploy, ~15–20 min)"
gh workflow run deploy --repo "$REPO" --ref main
sleep 5
item "$(gh run list --repo "$REPO" --workflow deploy --limit 1 --json url --jq '.[0].url')"

cat <<EOF

✓ Restarted $PROJECT_ID. Next:
  - follow the deploy:  gh run watch --repo $REPO
  - its summary shows the console URL; open it and hard-refresh (Cmd+Shift+R)
  - point your budget alert at $PROJECT_ID: Billing → Budgets & alerts
  - commit deployment/project.env and $TF_DIR/vars/env.tfvars if the ID changed
EOF
