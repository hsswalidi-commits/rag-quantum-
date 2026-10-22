# Quantum Knowledge Assistant

A stateless FastAPI service that answers questions about quantum physics and quantum computing in Arabic or English using the indexed documents. Responses include source filenames and the service refuses when retrieval finds no relevant context.

I built it to see how well a small RAG setup handles Arabic, and to measure my choices instead of guessing them.

<!-- Add the live demo link here after deploying -->

## The short version

| | |
|---|---|
| Vector database | Qdrant Cloud, cosine distance, 384-dimensional vectors |
| Chunking | 400 characters with 2-character overlap; PDF text is joined before chunking |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 by default; override with EMBEDDING_MODEL_NAME |
| LLM | Qwen2.5-72B-Instruct through OpenRouter |
| Search | Qdrant semantic search; BM25/RRF remains an experiment outside the API path |
| Interface | FastAPI service; optional Streamlit API client |

## How a question gets answered

Question -> rewrite follow-ups -> Qdrant semantic search -> select up to six chunks -> LLM answer with source filenames.

The client sends the last five turns with each request. The API does not keep conversation state.

## Vector database: Qdrant

I used Qdrant Cloud because it is built for exactly this job, has a free tier, and the Python client is simple. Everything lives in one collection with cosine distance. Each point stores the chunk text, source filename, chunk index, and a content hash. PDF page numbers are not retained.

Indexing is incremental. Every file gets a fingerprint made from its text and the chunking settings. Unchanged files are skipped, edited files are re-indexed, and the same content uploaded under a different name is detected and skipped. Point IDs are deterministic, so running it twice never creates duplicates.

## Chunking

The current settings use a 400-character chunk size with a 2-character overlap. PDF pages are extracted and joined before splitting, so responses cite the PDF filename but not a page number.

## Embedding model

The current code defaults to sentence-transformers/all-MiniLM-L6-v2 and accepts an EMBEDDING_MODEL_NAME override. The historical retrieval evaluation below compared that English model with multilingual alternatives. If changing the model for an existing Qdrant collection, re-embed the corpus with the same model used for queries; matching vector dimensions alone does not make different embedding spaces compatible.

## LLM

Qwen2.5-72B-Instruct (`qwen/qwen-2.5-72b-instruct`) through OpenRouter, using the standard `openai` client. Temperature is 0.1 for answers and 0 for rewriting. The prompt tells it to use only the supplied context, answer in the language of the question, and otherwise reply with a fixed "not available" sentence. Both the LLM and the embedding model can be swapped with environment variables.

## Search

The current API uses Qdrant cosine search with a 0.30 similarity threshold. The src/rag/hybrid.py module retains the BM25/RRF experiment, but it is not used by the request path.

## How I tested it

I wrote small question sets with a known right source file and scripted the checks (`eval_rag.py`, `compare_embeddings.py`, `eval_generation.py`).

Historical retrieval results on 16 in-scope Arabic and English questions; they are not a measurement of the current API defaults:

| Embedding model | Right file found | Right file ranked first |
|---|---|---|
| all-MiniLM-L6-v2 (English) | 9 / 16 | 7 / 16 |
| paraphrase-multilingual-MiniLM-L12-v2 | 16 / 16 | 14 / 16 |
| paraphrase-multilingual-mpnet-base-v2 | 16 / 16 | 14 / 16 |

The two multilingual models tied, so I kept the smaller one.

Full pipeline on 20 questions (15 in scope, 5 out of scope): 14 of 15 were answered with the right source, all 5 out-of-scope questions were refused, and every answer was in the right language. The one miss was the split list described above. I raised the retrieval limit from 4 to 6 afterwards but have not re-run the numbers. The sets are small, so treat these as trends, not guarantees.

## Run it

You need uv, a Qdrant Cloud cluster, and an OpenRouter API key.

    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    uv sync
    cp env.example .env
    uv run uvicorn main:app --reload --host 127.0.0.1 --port 8080

The first API start downloads the embedding model and indexes knowledge_base/. Open http://127.0.0.1:8080/docs for the interactive API schema. To run the optional Streamlit client, install uv sync --extra client, set RAG_API_URL=http://127.0.0.1:8080, then run uv run streamlit run chat_app.py. uv run python app.py remains a CLI entrypoint.
## Project layout

    main.py                FastAPI ASGI entrypoint
    src/rag/api.py         API routes, validation, startup lifecycle
    src/rag/core.py        retrieval, indexing, answer generation, CLI
    src/rag/hybrid.py      BM25/RRF experiment, not currently on request path
    chat_app.py            Optional Streamlit client using HTTP API calls
    Dockerfile             Cloud Run container build and server
    knowledge_base/        Documents indexed into Qdrant on startup
## Known limitations

- The API is stateless: conversation turns are sent by the client and are not persisted server-side.
- Scanned PDFs need OCR, which I haven't added.
- A follow-up like "tell me more" retrieves the same chunks as before, so it can repeat itself when the documents have nothing new.
- API access uses one shared bearer token on Cloud Run; add user-specific identity and authorization before serving multiple users.
- The API keeps no conversation history between client sessions.
- The API does not expose a document-upload route; update the checked-in knowledge base and deploy a new revision to re-index it.

## بالعربي

مساعد يجيب عن الأسئلة بالعربية والإنجليزية من ملفاتك فقط، ويذكر مصدر كل إجابة، وإذا لم يجد الجواب يقول ذلك بدل أن يخمّن.

- قاعدة المتجهات: Qdrant Cloud (مقياس cosine، 384 بُعدًا)
- تقسيم النصوص: 400 حرف مع تداخل حرفين؛ يُستخرج نص PDF كاملًا قبل التقسيم
- نموذج الـ Embedding الافتراضي: sentence-transformers/all-MiniLM-L6-v2، ويمكن تغييره عبر EMBEDDING_MODEL_NAME
- نموذج اللغة: Qwen2.5-72B-Instruct عبر OpenRouter
- نتائج التقييم المذكورة أعلاه تاريخية ولا تقيس إعدادات API الحالية
## FastAPI and Cloud Run

The API exposes GET /healthz and POST /v1/chat. The chat request carries up to five previous turns so the server can rewrite follow-up questions without local SQLite storage. FastAPI serves its OpenAPI docs at /docs.

The optional Streamlit app is a client only and is not copied into the API container. Cloud Run requests use PORT (default 8080), and the Docker image starts one Uvicorn worker so each instance loads one embedding model. Set API_ACCESS_TOKEN, OPENROUTER_API_KEY, QDRANT_URL, and QDRANT_API_KEY as Cloud Run secrets. The API requires the bearer token when running on Cloud Run; also keep Cloud Run IAM enabled for the service when callers can use IAM.

Build and run the container locally:

    docker build -t rag-api .
    docker run --rm -p 8080:8080 --env-file .env rag-api

For Cloud Run, deploy the Dockerfile from the repository and configure at least 2 vCPU and 4 GiB memory for the CPU embedding model. Use Secret Manager for all credentials. This repository does not include a project ID, region, or Cloud Run service name, so deployment itself must be configured for your GCP project.

### Cloud Run file audit

Included in the API image: Dockerfile, pyproject.toml, uv.lock, README.md, main.py, src/rag/, and knowledge_base/. requirements.txt is a pip export for alternate builders; this Dockerfile uses the locked uv project.

Not copied into the API image: chat_app.py (optional client), app.py (legacy CLI shim), storage.py (obsolete SQLite persistence), test_qdrant.py, evaluation/debug scripts and result files, .venv/, .env, conversations.db, .git/, and app_backup.py. .dockerignore and .gitignore keep local files out of builds and commits.

### Python version

The project is pinned to Python 3.14 (>=3.14,<3.15) to match .python-version and the 3.14-based Docker image. Cloud Run custom containers must listen on 0.0.0.0 at the injected PORT; the Dockerfile does both. The existing 3.12 image was inconsistent with the project metadata and lockfile.