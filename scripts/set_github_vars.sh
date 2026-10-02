#!/usr/bin/env bash
# Set the GitHub repository variables the deploy workflow reads, straight from
# `terraform output github_variables` (deployment/terraform/ml-team-adk).
#
#   scripts/set_github_vars.sh                          set them
#   scripts/set_github_vars.sh --dry-run                show what would be set
#   ML_MODEL=gemini-3.8-flash scripts/set_github_vars.sh  also pin the team's model
#                                                        (unset: config/config.yml decides)
#   REPO=owner/name scripts/set_github_vars.sh          another repository
#
# They are identifiers, not secrets: with Workload Identity Federation there are no keys.
set -euo pipefail
cd "$(dirname "$0")/.."

TF_DIR=deployment/terraform/ml-team-adk
REPO="${REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)}"
DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

fail() { echo "✗ $*" >&2; exit 1; }

command -v gh >/dev/null || fail "gh is not installed: brew install gh"
command -v terraform >/dev/null || fail "terraform is not installed: brew install hashicorp/tap/terraform"
gh auth status >/dev/null 2>&1 || fail "gh is not logged in: gh auth login"
[ -n "$REPO" ] || fail "No GitHub repository found here; set REPO=owner/name"

json="$(terraform -chdir="$TF_DIR" output -json github_variables 2>/dev/null)" ||
  fail "No Terraform output in $TF_DIR: run terraform init and terraform apply first (SETUP.md part 7)"

# name<TAB>value per line, from the Terraform output, then ML_MODEL if given
pairs="$(python3 -c 'import json,sys; [print(f"{k}\t{v}") for k, v in sorted(json.load(sys.stdin).items())]' <<<"$json")"
[ -n "${ML_MODEL:-}" ] && pairs+=$'\n'"ML_MODEL"$'\t'"$ML_MODEL"

echo "Repository variables for $REPO:"
while IFS=$'\t' read -r name value; do
  printf '  %-16s %s\n' "$name" "$value"
  $DRY_RUN || gh variable set "$name" --repo "$REPO" --body "$value"
done <<<"$pairs"

if $DRY_RUN; then
  echo "Dry run: nothing was set. Run without --dry-run to set them."
else
  echo "✓ Set. Check: gh variable list --repo $REPO"
fi
