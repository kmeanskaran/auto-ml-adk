# Terraform's state, in a bucket so it is not lost with this machine. The bucket is
# given at init, so each copy of the repo can use its own:
#   terraform init -backend-config="bucket=<project>-tfstate"
terraform {
  backend "gcs" {
    prefix = "ml-team-adk"
  }
}
