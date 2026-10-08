import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    KNOWLEDGE_BASE_DIR,
    RETRIEVAL_LIMIT,
    SIMILARITY_THRESHOLD,
    _load_source_files,
    _stable_point_id,
    load_clients,
    retrieve_context,
)
from eval_rag import TEST_CASES

CLEANUP_AFTER = True

MODELS_TO_COMPARE = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]


def _slug(model_name: str) -> str:
    return "eval_" + model_name.lower().replace("/", "-").replace("_", "-")


def _ingest_into(qdrant_client, embedding_model, collection_name: str) -> int:
    """فهرسة كاملة من الصفر داخل مجموعة مؤقتة (نسخة مبسطة عن ingest_knowledge_base
    بدون منطق الفهرسة التدريجية، لأن هنا نبدأ نظيف كل مرة أصلًا)."""
    source_files = _load_source_files(KNOWLEDGE_BASE_DIR)
    if not source_files:
        raise ValueError(f"No files found in '{KNOWLEDGE_BASE_DIR}/'.")

    vector_size = len(embedding_model.embed_query("dimension probe"))

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total = 0
    for source, content in source_files.items():
        chunks = splitter.split_text(content)
        if not chunks:
            continue

        embeddings = embedding_model.embed_documents(chunks)
        points = [
            PointStruct(
                id=_stable_point_id(source, index, chunk),
                vector=embedding,
                payload={"text": chunk, "source": source, "chunk_index": index},
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)
        total += len(points)

    return total

def _evaluate(qdrant_client, embedding_model, collection_name: str) -> dict:
    positive_hits = positive_top1 = positive_total = 0
    negative_ok = negative_total = 0
    positive_best_scores: list[float] = []
    negative_top_scores: list[float] = []

    for question, expected in TEST_CASES:
        chunks = retrieve_context(
            qdrant_client,
            embedding_model,
            collection_name,
            question,
            limit=RETRIEVAL_LIMIT,
            threshold=0.0,
        )
        top_source = chunks[0].source if chunks else None
        top_score = chunks[0].score if chunks else 0.0

        if expected is None:
            negative_total += 1
            negative_top_scores.append(top_score)
            negative_ok += int(top_score < SIMILARITY_THRESHOLD)
        else:
            positive_total += 1
            best = max((c.score for c in chunks if c.source == expected), default=0.0)
            positive_best_scores.append(best)
            positive_hits += int(best >= SIMILARITY_THRESHOLD)
            positive_top1 += int(top_source == expected)

    lowest_positive = min(positive_best_scores) if positive_best_scores else 0.0
    highest_negative = max(negative_top_scores) if negative_top_scores else 0.0

    return {
        "positive_hits": positive_hits,
        "positive_total": positive_total,
        "positive_top1": positive_top1,
        "negative_ok": negative_ok,
        "negative_total": negative_total,
        "separated": highest_negative < lowest_positive,
    }


def main() -> None:
    _, _, qdrant_client = load_clients()
    results = []

    for model_name in MODELS_TO_COMPARE:
        print(f"\n=== {model_name} ===")
        start = time.time()

        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        collection_name = _slug(model_name)

        point_count = _ingest_into(qdrant_client, embedding_model, collection_name)
        metrics = _evaluate(qdrant_client, embedding_model, collection_name)
        elapsed = time.time() - start

        metrics["model"] = model_name
        metrics["points"] = point_count
        results.append(metrics)

        print(
            f"points={point_count}  "
            f"found={metrics['positive_hits']}/{metrics['positive_total']}  "
            f"top1={metrics['positive_top1']}/{metrics['positive_total']}  "
            f"rejected={metrics['negative_ok']}/{metrics['negative_total']}  "
            f"separated={'yes' if metrics['separated'] else 'no'}  "
            f"time={elapsed:.1f}s"
        )

        if CLEANUP_AFTER:
            qdrant_client.delete_collection(collection_name)

    header = f"{'model':<58} {'found':>7} {'top1':>7} {'reject':>7} {'sep':>5}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        found_str = f"{r['positive_hits']}/{r['positive_total']}"
        top1_str = f"{r['positive_top1']}/{r['positive_total']}"
        reject_str = f"{r['negative_ok']}/{r['negative_total']}"
        sep_str = "yes" if r["separated"] else "no"
        print(f"{r['model']:<58} {found_str:>7} {top1_str:>7} {reject_str:>7} {sep_str:>5}")

    best = max(results, key=lambda r: (r["positive_top1"], r["negative_ok"]))
    print(f"\nBest by top-1 accuracy: {best['model']}")
    print(
        "\nTo use the best model, set in .env:\n"
        f"EMBEDDING_MODEL_NAME={best['model']}\n"
        "then run once with FORCE_REINGEST=1 on the real 'client' collection."
    )


if __name__ == "__main__":
    main()
