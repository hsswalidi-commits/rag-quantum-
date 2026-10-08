"""
debug_q7.py - prints the EXACT chunks retrieve_context() hands to the model
for one question, using the real threshold/limit (not a diagnostic override).
This lets us see directly whether the answer got split across chunks.

Run:
    uv run python .\\debug_q7.py
"""

from app import (
    COLLECTION_NAME,
    RETRIEVAL_LIMIT,
    SIMILARITY_THRESHOLD,
    load_clients,
    retrieve_context,
)

QUESTION = "لماذا لا تُستخدم الحوسبة الكمومية في كل شيء؟"


def main() -> None:
    _, embedding_model, qdrant_client = load_clients()

    chunks = retrieve_context(
        qdrant_client,
        embedding_model,
        COLLECTION_NAME,
        QUESTION,
        limit=RETRIEVAL_LIMIT,
        threshold=SIMILARITY_THRESHOLD,
    )

    print(f"Chunks actually sent to the model: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"--- chunk {i} | source={chunk.source} | score={chunk.score:.3f} ---")
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()
