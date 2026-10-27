# Production deployment: Cloud Run + Terraform

This project deploys the FastAPI RAG service to Cloud Run in `leap-coop`,
`me-central1`. The `Production deployment` GitHub Actions workflow validates PRs,
then builds a Linux amd64 image and deploys only from `main` (or a manual run on
`main`). Terraform owns the Cloud Run service, IAM, regional Artifact Registry,
regional Secret Manager resources, and Cloud Monitoring alert policies.

## One-time GCP setup

### Terraform state bucket

Create the private GCS backend bucket before the first `terraform init`. It must
match the bucket in `main.tf`:

```sh
gcloud storage buckets create gs://leap-coop-rag-terraform-state \
  --project=leap-coop --location=me-central1 \
  --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://leap-coop-rag-terraform-state --versioning
```

Grant only the deployment identity object access. Keep the bucket private;
remote state has resource metadata and the lock file records the selected
provider checksums.

### Deployment service account

Create a dedicated CI service account. Grant only the permissions needed to
enable APIs, create/update Cloud Run and Artifact Registry, manage Secret
Manager metadata and IAM, add secret versions, attach the runtime service
account, and read/write the backend bucket. The workflow identity must be able
to push images to Artifact Registry. Do not grant deployment privileges to the
runtime service account. The README's initial role set is broad; reduce it to
custom roles after validating the exact permissions in your organization.

For each secret, Cloud Run's runtime identity gets `roles/secretmanager.secretAccessor`
on that one secret. Secrets use regional replication in `me-central1`.

## GitHub configuration

Add these **repository secrets** in Settings → Secrets and variables → Actions:

- `GCP_SA_KEY`: JSON key for the dedicated deployment service account. This is
  used only by the Google auth action; the workflow never writes it into the
  checkout or Terraform state.
- `OPENROUTER_API_KEY`
- `QDRANT_API_KEY`: create this as a **read-only** Qdrant key.
- `QDRANT_URL`
- `API_BEARER_TOKEN`: a long random token used by `/v1/chat` in Cloud Run.

The ingestion CLI needs a separate write-capable `QDRANT_INGEST_API_KEY` in the
trusted local/CI ingestion environment. Do not add that key to the Cloud Run
service configuration.

Restrict the GitHub `production` environment to `main` and add required
reviewers if desired. Optionally create a Cloud Monitoring notification channel
and define the GitHub environment variable `MONITORING_NOTIFICATION_CHANNELS`
as a JSON list of channel resource names. Example shape:
`["projects/leap-coop/notificationChannels/123456789"]`. Without this variable,
alert policies are created but do not send notifications.

## Deploy flow

1. Pull requests run Ruff, the existing offline setup check, pytest, Terraform
   formatting, and Terraform validation.
2. After merge to `main`, Actions authenticates with `GCP_SA_KEY`, bootstraps
   APIs/registry/secret containers, builds an immutable commit-tagged amd64
   image, and pushes it to Artifact Registry.
3. Action secrets are piped directly into Secret Manager. Their payloads are
   never passed as Terraform variables. Terraform only receives the pinned
   numeric secret-version identifiers (marked sensitive to redact them in
   plans) and the image reference.
4. Terraform deploys `rag-agent-app`, grants `allUsers` Cloud Run invoker, and
   attaches the secret references and runtime service account.
5. Actions checks `/healthz` and `/readyz` after deployment.

The API remains bearer-authenticated even though the Cloud Run service is
publicly invokable. A leaked bearer token can be used by anyone; rotate it in
GitHub Secrets and deploy again if it is exposed.

## Security and operations

- Container Registry writes were shut down in March 2025. This configuration
  uses the supported regional Artifact Registry endpoint
  `me-central1-docker.pkg.dev/leap-coop/rag-agent`.
- The multi-stage Docker image uses Python 3.14, installs the CPU PyTorch build,
  downloads the public multilingual embedding model at build time, runs as a
  non-root UID, and sets Hugging Face offline mode in the runtime stage. It does
  not include local knowledge files, tests, credentials, or Terraform state.
- Cloud Run uses two vCPUs, 4 GiB memory, concurrency four, a 120-second timeout,
  scale-to-zero, at most three instances, startup CPU boost, `/readyz` startup
  probing, and `/healthz` liveness probing.
- JSON logs contain request IDs, Cloud Trace correlation when available, route
  status, latency, refusal status, and retrieval score/rank summaries. They do
  not contain question bodies, tokens, or API keys. Cloud Monitoring policies
  watch for 5xx responses and readiness failures.
- The in-memory rate limiter is per Cloud Run instance, so it is an abuse guard,
  not a global quota. Use a shared limiter or API gateway if account-wide usage
  enforcement is required.
- `terraform validate` checks configuration shape, but `terraform plan/apply`
  and a real Qdrant/OpenRouter request require your GCP account and runtime
  secrets. The local Docker Desktop ARM host may fail amd64 emulation; the GitHub
  hosted runner builds amd64 natively.
- Service-account keys are long-lived credentials. Workload Identity Federation
  is the recommended follow-up migration when you are ready to replace the
  requested `GCP_SA_KEY` secret.
