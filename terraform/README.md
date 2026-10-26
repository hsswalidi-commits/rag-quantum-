# Cloud Run deployment setup

The workflow deploys from pushes to `main` and `workflow_dispatch`. It builds
the Docker image, writes GitHub secret values directly to Secret Manager, and
then deploys the Cloud Run service with Terraform. Secret payloads are never
Terraform inputs; Terraform state contains only Secret Manager resource names
and pinned version identifiers.

## One-time Google Cloud setup

1. Create a GCS bucket for Terraform state before the first `terraform init`.
   The bucket name must match the backend in `main.tf`:

   ```sh
   gcloud storage buckets create gs://leap-coop-rag-terraform-state \
     --project=leap-coop --location=me-central1 \
     --uniform-bucket-level-access --public-access-prevention
   gcloud storage buckets update gs://leap-coop-rag-terraform-state \
     --versioning
   ```

   Keep this bucket private and grant the deploy identity access to its objects.
   Terraform's GCS backend supports state locking. State may contain resource
   metadata, so do not grant broad bucket access.

2. Create a dedicated GitHub deployment service account and grant the minimum
   permissions needed to bootstrap APIs, create the Artifact Registry
   repository, manage Cloud Run and Secret Manager metadata/IAM, publish secret
   versions, and update the GCS Terraform state. A practical initial role set is
   `roles/run.admin`, `roles/artifactregistry.admin`,
   `roles/secretmanager.admin`, `roles/serviceusage.serviceUsageAdmin`, and
   `roles/iam.serviceAccountAdmin`, plus `roles/iam.serviceAccountUser` on the
   runtime service account and object read/write access to the state bucket.
   Reduce these to custom roles after validating the required permissions.
   The workflow also needs Artifact Registry push permission. Do not grant the
   Cloud Run runtime identity deployment permissions.

3. Configure these **GitHub repository secrets** (Settings → Secrets and
   variables → Actions):

   - `GCP_SA_KEY`: JSON key for the dedicated deployment identity. Workload
     Identity Federation is preferable because it avoids a long-lived key; the
     workflow currently uses this secret to match the requested setup.
   - `OPENROUTER_API_KEY`
   - `QDRANT_API_KEY`: use a read-only Qdrant key for the API service.
   - `QDRANT_URL`
   - `API_BEARER_TOKEN`: required by the API in Cloud Run. Use a long random
     value. This secret is separate from the GCP deployment credential.

4. Restrict the GitHub `production` environment to the intended deployment
   branch and add required reviewers if your release process requires approval.

## Deploy

Push a commit to `main`, or run **Build and deploy RAG API** from the Actions
tab. The workflow bootstraps the registry and Secret Manager containers, builds
and pushes a commit-tagged image, adds new Secret Manager versions from the
GitHub repository secrets, pins those version numbers in Cloud Run, and applies
the service configuration.

The Cloud Run service is publicly invokable at the network layer (`allUsers`),
as requested. API bearer authentication must remain enabled in the application.
Anyone who obtains the API token can call the service, so rotate it promptly if
exposed. The runtime service account receives Secret Accessor only on the four
application secrets. Secret Manager values are environment variables pinned to
numeric versions; a new GitHub secret version is attached by the next deploy.

## Registry and state notes

The workflow uses Artifact Registry at
`me-central1-docker.pkg.dev/leap-coop/rag-agent`. Container Registry writes
were shut down on March 18, 2025, so a new deployment should not push to legacy
`gcr.io` Container Registry. Artifact Registry is Google's supported successor.

Terraform's `sensitive = true` redacts values in Terraform output and plans; it
does not keep secret payloads out of the state file. For that reason this
configuration marks only the version identifier map as sensitive and does not
pass API keys or the GCP JSON credential to Terraform. GitHub Actions sends
runtime secret payloads straight to Secret Manager over stdin.

The Cloud Run service uses scale-to-zero, a maximum of three instances, two
vCPUs, 4 GiB memory, concurrency four, and a 120-second request timeout. Review
cost and latency after observing cold starts and model load time in the target
region.
