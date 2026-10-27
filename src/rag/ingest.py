"""Offline/local knowledge-base ingestion CLI; never run from API startup."""

from __future__ import annotations

import argparse
import logging

from rag.core import (
    KNOWLEDGE_BASE_DIR,
    ingest_knowledge_base,
    load_embedding_model,
    load_qdrant_client,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the RAG knowledge base into Qdrant.")
    parser.add_argument("--force", action="store_true", help="Rebuild source points without deleting the Qdrant collection.")
    parser.add_argument("--directory", default=KNOWLEDGE_BASE_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    qdrant_client = load_qdrant_client(for_ingest=True)
    try:
        ingest_knowledge_base(
            qdrant_client,
            load_embedding_model(),
            directory=args.directory,
            force=args.force,
        )
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    main()
