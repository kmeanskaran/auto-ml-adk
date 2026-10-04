#!/usr/bin/env bash
# Remove everything the ML team created in Google Cloud, A to Z. GitHub (the deploy
# workflow, its variables) is left alone: scripts/restart_gcp.sh rewrites the variables.
#
#   scripts/nuke_gcp.sh                       dry run: lists what would go, changes nothing
#   scripts/nuke_gcp.sh --apply               backs up the team's work, then removes it all
#   scripts/nuke_gcp.sh --apply --no-backup   the same, without the backup
#
#   PROJECT_ID=… scripts/nuke_gcp.sh          another project
#
# What --apply does, in order (after you type the project ID):
#   1. backs up gs://$STATE_BUCKET (runs, model registry, feature store) to backups/
#   2. deletes the Google Cloud project, and with it everything inside: the agent on
#      Agent Runtime and its sessions, the console on Cloud Run, every bucket (state,
#      logs, Terraform's state), service accounts, Workload Identity, BigQuery tables,
#      images, logs and traces. Billing for the project stops.
#   3. this machine: Terraform's working files, gcloud's default project
#
# A deleted project can be restored for 30 days: gcloud projects undelete $PROJECT_ID.
# After that it is gone for good, and its ID can never be used again.
set -euo pipefail
cd "$(dirname "$0")/.."

# The project comes from deployment/project.env; PROJECT_ID=… in the shell overrides it.
SHELL_PROJECT_ID="${PROJECT_ID:-}"
# shellcheck source=../deployment/project.env
source deployment/project.env
PROJECT_ID="${SHELL_PROJECT_ID:-$PROJECT_ID}"
STATE_BUCKET="${STATE_BUCKET:-$PROJECT_ID-state}"
TF_DIR=deployment/terraform/ml-team-adk

APPLY=false
BACKUP=true
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=true ;;
    --no-backup) BACKUP=false ;;
    -h | --help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown option $arg (see --help)" >&2; exit 1 ;;
  esac
done

step() { echo; echo "● $*"; }
item() { echo "    $*"; }
fail() { echo "✗ $*" >&2; exit 1; }

command -v gcloud >/dev/null || fail "gcloud is not installed."

# --- What is there -------------------------------------------------------------------

state="$(gcloud projects describe "$PROJECT_ID" --format='value(lifecycleState)' 2>/dev/null || true)"
[ -n "$state" ] || fail "No project $PROJECT_ID, or no access to it (gcloud auth login?)."
[ "$state" = ACTIVE ] || fail "Project $PROJECT_ID is already $state."

step "Google Cloud project $PROJECT_ID (everything below goes with it)"
item "billing: $(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' 2>/dev/null || echo '?')"

token="$(gcloud auth print-access-token 2>/dev/null || true)"
engines="$(curl -s -H "Authorization: Bearer $token" \
  "https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/reasoningEngines" |
  python3 -c 'import json,sys; [print(e.get("displayName"), "·", e["name"].rsplit("/",1)[-1]) for e in json.load(sys.stdin).get("reasoningEngines", [])]' 2>/dev/null || true)"
item "Agent Runtime agents: ${engines:-none}"
item "Cloud Run services: $(gcloud run services list --project "$PROJECT_ID" --format='value(metadata.name)' 2>/dev/null | paste -sd' ' - || true)"
item "buckets:"
for bucket in $(gcloud storage buckets list --project "$PROJECT_ID" --format='value(name)' 2>/dev/null); do
  item "  gs://$bucket  $(gcloud storage du -s "gs://$bucket" 2>/dev/null | awk '{print $1" bytes"}')"
done
item "service accounts: $(gcloud iam service-accounts list --project "$PROJECT_ID" --format='value(email)' 2>/dev/null | paste -sd' ' - || true)"
item "Workload Identity pools: $(gcloud iam workload-identity-pools list --project "$PROJECT_ID" --location global --format='value(name.basename())' 2>/dev/null | paste -sd' ' - || true)"
item "BigQuery datasets: $(bq ls --project_id "$PROJECT_ID" --format=csv 2>/dev/null | tail -n +2 | paste -sd' ' - || true)"

step "This machine"
item "$TF_DIR/.terraform, $TF_DIR/tfplan"
item "gcloud default project: $(gcloud config get-value project 2>/dev/null)"

if ! $APPLY; then
  echo
  echo "Dry run: nothing was changed. To remove all of the above: $0 --apply"
  exit 0
fi

# --- Remove it -----------------------------------------------------------------------

echo
echo "This deletes the project $PROJECT_ID and everything in it."
read -r -p "Type the project ID to confirm: " answer
[ "$answer" = "$PROJECT_ID" ] || fail "Not confirmed; nothing was changed."

if $BACKUP && gcloud storage buckets describe "gs://$STATE_BUCKET" >/dev/null 2>&1; then
  backup="backups/$PROJECT_ID-$(date +%Y%m%d-%H%M%S)"
  step "Backing up gs://$STATE_BUCKET to $backup/"
  mkdir -p "$backup"
  gcloud storage rsync --recursive "gs://$STATE_BUCKET" "$backup" ||
    fail "The backup failed, so nothing was deleted. Retry, or use --no-backup."
fi

step "Deleting the Google Cloud project $PROJECT_ID"
gcloud projects delete "$PROJECT_ID" --quiet

step "Cleaning up this machine"
rm -rf "$TF_DIR/.terraform" "$TF_DIR/tfplan"
[ "$(gcloud config get-value project 2>/dev/null)" = "$PROJECT_ID" ] && gcloud config unset project
item "done"

cat <<EOF

✓ Removed. Billing for $PROJECT_ID has stopped; the project can be restored until 30 days
  from now with: gcloud projects undelete $PROJECT_ID

Left for you:
  - .env still names $PROJECT_ID: switch it to another project or to GEMINI_API_KEY
  - your application-default login bills $PROJECT_ID: gcloud auth application-default set-quota-project <other>
  - a budget alert lives on the billing account, not the project: remove it under Billing → Budgets & alerts
  - to start again (the old ID can't be reused): scripts/restart_gcp.sh <new-project-id>
EOF
