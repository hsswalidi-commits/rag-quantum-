"""
البحث بالكلمات المفتاحية (BM25) + دمج النتائج بصيغة RRF.

الفهرس يتبني في الذاكرة من القطع الموجودة في Qdrant، وينعاد بناؤه تلقائيًا
بعد كل فهرسة (عن طريق invalidate()).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

RRF_K = 60

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
# حروف عربية فقط (بدون علامات الترقيم مثل ؟ و ،)
_TOKEN = re.compile(r"[A-Za-z0-9\u0621-\u064A\u0671-\u06D3]+")


def _normalize(text: str) -> str:
    """تطبيع بسيط: أحرف صغيرة، إزالة التشكيل والتطويل، توحيد الألف والياء والتاء المربوطة."""
    text = _DIACRITICS.sub("", text.lower())
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


_RAW_STOPWORDS = {
    # Arabic
    "ما", "ماذا", "هي", "هو", "هذا", "هذه", "وش", "ايش", "في", "من", "على",
    "الى", "عن", "هل", "كيف", "لماذا", "ليش", "اللي", "الذي", "التي", "مع", "او",
    # English
    "what", "is", "the", "a", "an", "of", "in", "to", "and", "or", "how", "does",
    "do", "are", "why", "which", "for", "with", "on", "it", "this", "that",
}
STOPWORDS = {_normalize(word) for word in _RAW_STOPWORDS}


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(_normalize(text)):
        # إزالة "ال" التعريف من بداية الكلمة (قبل ما نفلتر الكلمات الشائعة)
        if len(token) > 3 and token.startswith("ال"):
            token = token[2:]
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


@dataclass
class IndexedChunk:
    id: str
    text: str
    source: str
    page: int | None


class BM25Index:
    def __init__(self, chunks: list[IndexedChunk]):
        self.chunks = chunks
        corpus = [tokenize(chunk.text) or ["_"] for chunk in chunks]
        self._bm25 = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, limit: int) -> list[tuple[IndexedChunk, float]]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]


def rrf_fuse(*ranked_id_lists: list[str], k: int = RRF_K) -> dict[str, float]:
    """RRF: كل قطعة تاخذ 1/(k + الرتبة) من كل قائمة تظهر فيها، والنقاط تنجمع."""
    fused: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def build_index(qdrant_client, collection_name: str) -> BM25Index:
    chunks: list[IndexedChunk] = []
    next_offset = None

    while True:
        records, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=256,
            offset=next_offset,
        )
        for record in records:
            payload = record.payload or {}
            text = payload.get("text")
            if text:
                chunks.append(
                    IndexedChunk(
                        id=str(record.id),
                        text=text,
                        source=payload.get("source", "unknown"),
                        page=payload.get("page"),
                    )
                )
        if next_offset is None:
            break

    return BM25Index(chunks)


_index: BM25Index | None = None


def invalidate() -> None:
    """يُستدعى بعد أي فهرسة، عشان ينعاد بناء فهرس الكلمات عند أول سؤال."""
    global _index
    _index = None


def get_index(qdrant_client, collection_name: str) -> BM25Index:
    global _index
    if _index is None:
        _index = build_index(qdrant_client, collection_name)
    return _index
