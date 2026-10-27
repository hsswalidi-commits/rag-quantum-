terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.2.0"
    }
  }

  # Bootstrap this bucket once before CI; details are in the deployment notes.
  backend "gcs" {
    bucket = "leap-coop-rag-terraform-state"
    prefix = "rag-agent/production"
  }
}

variable "project_id" {
  description = "Google Cloud project hosting the RAG service."
  type        = string
  default     = "leap-coop"
}

variable "region" {
  description = "Google Cloud region for Cloud Run and Artifact Registry."
  type        = string
  default     = "me-central1"
}

variable "image" {
  description = "Immutable Artifact Registry image reference to deploy."
  type        = string
}

# Secret payloads are NEVER Terraform variables: Terraform state can retain
# resource inputs even when the CLI marks an input as sensitive. GitHub Actions
# publishes the GitHub secret values directly into Secret Manager. These are
# only the numeric version identifiers, which allow Cloud Run env secrets to
# use pinned versions.
variable "runtime_secret_versions" {
  description = "Pinned Secret Manager version identifiers. Payloads never enter Terraform; this map is marked sensitive to avoid exposing internal secret metadata in plans."
  type = object({
    openrouter = string
    qdrant_key = string
    qdrant_url = string
    api_token  = string
  })
  sensitive = true
  default = {
    openrouter = "1"
    qdrant_key = "1"
    qdrant_url = "1"
    api_token  = "1"
  }
}

variable "max_instances" {
  description = "Maximum Cloud Run instance count."
  type        = number
  default     = 3
}

variable "monitoring_notification_channels" {
  description = "Cloud Monitoring notification channel resource names. Configure email or incident channels in Cloud Monitoring and pass their full resource names here."
  type        = list(string)
  default     = []
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  runtime_secrets = {
    OPENROUTER_API_KEY = {
      secret       = "rag-openrouter-api-key"
      version_name = "openrouter"
    }
    QDRANT_API_KEY = {
      secret       = "rag-qdrant-api-key"
      version_name = "qdrant_key"
    }
    QDRANT_URL = {
      secret       = "rag-qdrant-url"
      version_name = "qdrant_url"
    }
    API_ACCESS_TOKEN = {
      secret       = "rag-api-bearer-token"
      version_name = "api_token"
    }
  }
}

resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "app" {
  project       = var.project_id
  location      = var.region
  repository_id = "rag-agent"
  description   = "Container images for rag-agent-app Cloud Run service"
  format        = "DOCKER"

  depends_on = [google_project_service.required]
}

resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "rag-agent-runtime"
  display_name = "RAG Agent Cloud Run runtime"
}

resource "google_secret_manager_secret" "runtime" {
  for_each  = toset([for secret in values(local.runtime_secrets) : secret.secret])
  project   = var.project_id
  secret_id = each.value

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_iam_member" "runtime_access" {
  for_each  = google_secret_manager_secret.runtime
  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "app" {
  project             = var.project_id
  name                = "rag-agent-app"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = true

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "120s"
    max_instance_request_concurrency = 4

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 2
        period_seconds        = 10
        failure_threshold     = 18
        http_get {
          path = "/readyz"
          port = 8080
        }
      }

      liveness_probe {
        timeout_seconds   = 2
        period_seconds    = 30
        failure_threshold = 3
        http_get {
          path = "/healthz"
          port = 8080
        }
      }

      env {
        name  = "COLLECTION_NAME"
        value = "client"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "EMBEDDING_MODEL_NAME"
        value = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
      }
      env {
        name  = "GENERATION_MODEL_NAME"
        value = "qwen/qwen-2.5-72b-instruct"
      }
      env {
        name  = "QDRANT_TIMEOUT_SECONDS"
        value = "10"
      }
      env {
        name  = "OPENROUTER_TIMEOUT_SECONDS"
        value = "45"
      }
      env {
        name  = "MAX_HISTORY_CHARS"
        value = "12000"
      }
      env {
        name  = "RATE_LIMIT_PER_MINUTE"
        value = "20"
      }
      env {
        name  = "ENABLE_DOCS"
        value = "0"
      }
      env {
        name  = "HYBRID_SEARCH"
        value = "1"
      }

      dynamic "env" {
        for_each = local.runtime_secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value.secret
              version = var.runtime_secret_versions[env.value.version_name]
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_secret_manager_secret_iam_member.runtime_access,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_monitoring_alert_policy" "api_5xx" {
  project               = var.project_id
  display_name          = "RAG API returned a 5xx response"
  combiner              = "OR"
  notification_channels = var.monitoring_notification_channels
  enabled               = true

  conditions {
    display_name = "Cloud Run request returned HTTP 5xx"
    condition_matched_log {
      filter = <<-EOT
        resource.type="cloud_run_revision"
        resource.labels.service_name="rag-agent-app"
        jsonPayload.event="http_request_complete"
        jsonPayload.status_code >= 500
      EOT
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_project_service.required]
}

resource "google_monitoring_alert_policy" "readiness_failure" {
  project               = var.project_id
  display_name          = "RAG API is not ready"
  combiner              = "OR"
  notification_channels = var.monitoring_notification_channels
  enabled               = true

  conditions {
    display_name = "Qdrant readiness check failed"
    condition_matched_log {
      filter = <<-EOT
        resource.type="cloud_run_revision"
        resource.labels.service_name="rag-agent-app"
        (jsonPayload.event="readiness_check_failed" OR jsonPayload.event="readiness_collection_missing")
      EOT
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_project_service.required]
}

output "service_url" {
  description = "Public Cloud Run URL; API bearer-token authentication is still enforced by the application."
  value       = google_cloud_run_v2_service.app.uri
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.app.name
}

