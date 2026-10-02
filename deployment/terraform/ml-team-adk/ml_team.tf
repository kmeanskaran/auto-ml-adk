# The ML team's own infrastructure, on top of the scaffolded agent (service.tf, iam.tf):
#
#   state bucket   runs/, registry/, feature_store/ survive redeploys (state_sync.py)
#   console        its service account, may only call the agent; IAP lets in console_users
#   deployer       the service account GitHub Actions deploys as, through Workload
#                  Identity Federation: no keys, only github_repository's main branch
#
# Applied by hand (terraform apply); CI/CD only deploys code (.github/workflows/deploy.yml).

variable "github_repository" {
  type        = string
  description = "GitHub owner/name allowed to deploy, e.g. kmeanskaran/auto-ml-adk."
}

variable "console_users" {
  type        = list(string)
  description = "Who may open the console through IAP, e.g. [\"user:you@gmail.com\"]."
}

variable "state_bucket" {
  type        = string
  description = "Bucket keeping runs/, registry/ and feature_store/."
}

locals {
  ml_team_services = [
    "iap.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ]
  deployer_roles = [
    "roles/aiplatform.user",          # update the Agent Runtime agent
    "roles/run.admin",                # deploy the console, set its IAP
    "roles/cloudbuild.builds.editor", # gcloud run deploy --source builds
    "roles/artifactregistry.writer",  # push the console image
    "roles/storage.admin",            # upload the sources to build
    "roles/logging.viewer",           # follow build logs
    "roles/serviceusage.serviceUsageConsumer",
  ]
}

resource "google_project_service" "ml_team" {
  for_each           = toset(local.ml_team_services)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
  depends_on         = [google_project_service.bootstrap]
}

# --- State bucket ------------------------------------------------------------------

# Created with gcloud before Terraform; adopted here.
import {
  to = google_storage_bucket.state
  id = var.state_bucket
}

resource "google_storage_bucket" "state" {
  name                        = var.state_bucket
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true # an overwritten model or run file can be recovered
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle {
    prevent_destroy = true # holds the team's models and runs
  }
}

resource "google_storage_bucket_iam_member" "state_app" {
  bucket = google_storage_bucket.state.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.app_sa.email}"
}

# --- Console -----------------------------------------------------------------------

resource "google_service_account" "console" {
  account_id   = "${var.project_name}-console"
  display_name = "${var.project_name} console (Cloud Run)"
  project      = var.project_id
  depends_on   = [google_project_service.services]
}

# Calls the agent through the Agent Runtime /api passthrough; nothing else.
resource "google_project_iam_member" "console_agent" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.console.email}"
}

# IAP in front of the console: who may open it, and IAP's own right to call it.
resource "google_project_service_identity" "iap" {
  provider   = google-beta
  project    = var.project_id
  service    = "iap.googleapis.com"
  depends_on = [google_project_service.ml_team]
}

resource "google_project_iam_member" "iap_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = google_project_service_identity.iap.member
}

resource "google_project_iam_member" "console_users" {
  for_each = toset(var.console_users)
  project  = var.project_id
  role     = "roles/iap.httpsResourceAccessor"
  member   = each.value
}

# gcloud run deploy --source pushes the console image here.
resource "google_artifact_registry_repository" "console" {
  project       = var.project_id
  location      = var.region
  repository_id = "cloud-run-source-deploy"
  format        = "DOCKER"
  description   = "Console images built by gcloud run deploy --source"
  depends_on    = [google_project_service.ml_team]
}

# --- Deployer: GitHub Actions, through Workload Identity Federation ---------------

resource "google_service_account" "deployer" {
  account_id   = "${var.project_name}-deployer"
  display_name = "${var.project_name} deployer (GitHub Actions)"
  project      = var.project_id
  depends_on   = [google_project_service.services]
}

resource "google_project_iam_member" "deployer" {
  for_each = toset(local.deployer_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

# The deployer runs the agent as ml-team-app and the console as ml-team-console.
resource "google_service_account_iam_member" "deployer_acts_as" {
  for_each = {
    app     = google_service_account.app_sa.name
    console = google_service_account.console.name
  }
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.ml_team]
}

# Only workflow runs of this repository on its main branch can get a token: a push to
# main deploys; pull requests and other branches can't.
resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  attribute_condition = "assertion.repository == '${var.github_repository}' && assertion.ref == 'refs/heads/main'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "deployer_wif" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# --- What GitHub Actions needs (repository variables) ------------------------------

output "github_variables" {
  description = "Set these as GitHub repository variables (gh variable set)."
  value = {
    GCP_PROJECT_ID  = var.project_id
    GCP_REGION      = var.region
    WIF_PROVIDER    = google_iam_workload_identity_pool_provider.github.name
    DEPLOYER_SA     = google_service_account.deployer.email
    APP_SA          = google_service_account.app_sa.email
    CONSOLE_SA      = google_service_account.console.email
    STATE_BUCKET    = google_storage_bucket.state.name
    AGENT_ENGINE_ID = element(split("/", google_vertex_ai_reasoning_engine.app.id), length(split("/", google_vertex_ai_reasoning_engine.app.id)) - 1)
  }
}
