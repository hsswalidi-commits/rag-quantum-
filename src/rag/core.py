from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIError, OpenAI
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.hybrid import get_index, invalidate, rrf_fuse
from rag.observability import log_event

load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "client")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
VECTOR_SIZE = 384

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
KNOWLEDGE_FILE_EXTENSIONS = (".md", ".txt", ".pdf")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
CHUNK_FORMAT_VERSION = 1
CHUNK_SEPARATORS = ["\n\n", "\n", "؟ ", ". ", "؛ ", "! ", " ", ""]


SIMILARITY_THRESHOLD = 0.30
RETRIEVAL_LIMIT = 4
SEMANTIC_CANDIDATE_LIMIT = 20
BM25_CANDIDATE_LIMIT = 20
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1").strip().lower() not in {"0", "false", "no"}


ANSWER_MAX_TOKENS = 600
ANSWER_MAX_TOKENS_RETRY = 1000
REWRITE_MAX_TOKENS = 150


MAX_HISTORY_TURNS = 5

REFERENCE_WORDS = {
    "it", "this", "that", "they", "them", "these", "those",
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "ذاته", "نفسه",
}

DEBUG_REWRITE = os.getenv("DEBUG_REWRITE") == "1"
FORCE_REINGEST = os.getenv("FORCE_REINGEST") == "1"


@dataclass
class ConversationTurn:
    question: str
    rewritten_question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: list[ConversationTurn] = field(default_factory=list)

    def add(self, question: str, rewritten_question: str, answer: str) -> None:
        self.turns.append(ConversationTurn(question, rewritten_question, answer))

    def recent(self, max_turns: int = MAX_HISTORY_TURNS) -> list[ConversationTurn]:
        return self.turns[-max_turns:]


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    page: int | None = None
    # حقول التشخيص (تستخدمها واجهة chat_app). لها قيم افتراضية
    # عشان البحث الدلالي العادي يشتغل بدونها.
    via: str = "semantic"
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    bm25_score: float = 0.0
    fusion_score: float = 0.0


def source_label(chunk: RetrievedChunk) -> str:
    """اسم الملف فقط بدون المسار."""
    label = os.path.basename(str(chunk.source))
    return f"{label} (ص {chunk.page})" if chunk.page is not None else label


def _no_context_message(query: str) -> str:
    """رسالة الرفض لما ما يوجد سياق مناسب، بلغة السؤال."""
    if detect_language(query) == "Arabic":
        return "ما لقيت إجابة لهذا السؤال في المعلومات المتوفرة عندي."
    return "The answer is not available in the provided context."


def detect_language(text: str) -> str:
    arabic = len(re.findall(r"[\u0621-\u064A\u066E-\u06D3]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "Arabic" if arabic > latin else "English"


NO_ANSWER_MESSAGES = {
    "Arabic": "ما لقيت إجابة لهذا السؤال في المعلومات المتوفرة عندي.",
    "English": "The answer is not available in the provided context.",
}


def clean_model_answer(content) -> str:
    if content is None:
        return ""

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    text = str(content).strip()

    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL
    )

    if re.search(r"</think\s*>", text, flags=re.IGNORECASE):
        text = re.split(r"</think\s*>", text, flags=re.IGNORECASE)[-1]

    if re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE):
        after_think = re.split(
            r"<think\b[^>]*>", text, maxsplit=1, flags=re.IGNORECASE
        )[-1]

        final_markers = [
            r"(?:final answer|answer)\s*:\s*",
            r"(?:الإجابة النهائية|الإجابة|الجواب)\s*:\s*",
            r"#+\s*(?:final answer|answer|الإجابة النهائية|الإجابة|الجواب)\s*",
        ]

        extracted_answer = None
        for marker in final_markers:
            match = re.search(marker, after_think, flags=re.IGNORECASE)
            if match:
                extracted_answer = after_think[match.end():].strip()
                break

        text = extracted_answer if extracted_answer else after_think

    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"^\s*(?:final answer|answer|الإجابة النهائية|الإجابة|الجواب)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
    
NO_MORE_MESSAGES = {
    "Arabic": "هذا كل ما لدي عن هذا الموضوع في المصادر المتوفرة.",
    "English": "That is everything I have on this topic in the available sources.",
}


def build_prompt(
    context: str, query: str, language: str, previous_answer: str | None = None
) -> str:
    no_answer = NO_ANSWER_MESSAGES[language]
    no_more = NO_MORE_MESSAGES[language]

    # For follow-ups ("tell me more"), show the model what it already said so
    # it adds new details instead of paraphrasing the same answer.
    follow_up_block = ""
    if previous_answer:
        follow_up_block = f"""
This is a follow-up question. The user already received this answer:
<previous_answer>
{previous_answer}
</previous_answer>
Do not repeat what it already says. Add only new details found in the context.
If the context has nothing new, reply with exactly this sentence:
"{no_more}"
"""

    return f"""
Use only the supplied context to answer the question.

Instructions:
- Return only the final answer, nothing else.
- Do not explain your reasoning or the retrieval process.
- Write the answer in {language}, even if the context is in another language.
- If the answer is not available in the context, reply with exactly this sentence:
  "{no_answer}"
- Give a complete answer: at least 2-3 full sentences when the context
  supports it. Avoid single-word or single-phrase answers unless the
  question explicitly asks for a short list or a yes/no.
- The context is split into numbered passages like [1], [2]. After the answer,
  add one last line in exactly this format: USED: 1, 3
  listing only the numbers of the passages you actually used. This line is
  required output, not commentary. Do not mention passage numbers anywhere
  else, and do not add this line when you reply with one of the exact
  sentences above.
{follow_up_block}
CONTEXT:
{context}

QUESTION:
{query}
""".strip()

def looks_like_followup(question: str) -> bool:
    """
    فحص سريع (heuristic) لوجود كلمات مرجعية في السؤال (عربي/إنجليزي).
    هذا لا يحدد النتيجة بشكل نهائي، فقط يساعد على تقليل عدد
    استدعاءات النموذج غير الضرورية.
    """
    words = re.findall(r"[a-zA-Z\u0600-\u06FF']+", question.lower())
    return any(word in REFERENCE_WORDS for word in words)


def format_history_for_rewrite(
    turns: list[ConversationTurn], max_turns: int = MAX_HISTORY_TURNS
) -> str:
    recent_turns = turns[-max_turns:]
    lines = []
    for turn in recent_turns:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)


def rewrite_query(
    llm_client: OpenAI,
    model_name: str,
    current_question: str,
    history: list[ConversationTurn],
) -> str:
    """
    يحاول تحويل السؤال الحالي إلى سؤال مستقل وواضح باستخدام سياق المحادثة.
    """
    if not history:
        return current_question

    if not looks_like_followup(current_question) and len(current_question.split()) > 4:
        return current_question

    history_text = format_history_for_rewrite(history)

    system_prompt = (
        "You rewrite user questions for a search system. "
        "You are given the recent conversation history and the current question. "
        "If the current question depends on the previous conversation "
        "(uses words like it/this/that/they/them/these/those, or is otherwise "
        "incomplete without the previous context), rewrite it into a standalone, "
        "self-contained question that keeps the original meaning and language. "
        "If the current question is already independent and introduces a new "
        "topic, return it exactly as it is, unchanged. "
        "Return only the question text, with no quotes, labels, or explanation."
    )

    user_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Current question:\n{current_question}\n\n"
        "Rewritten standalone question:"
    )

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=REWRITE_MAX_TOKENS,
            temperature=0.0,
        )

        raw_output = response.choices[0].message.content
        rewritten = clean_model_answer(raw_output).strip().strip('"').strip("'").strip()

        return rewritten or current_question

    except APIError:
        return current_question


def retrieve_context(
    qdrant_client: QdrantClient,
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str,
    query: str,
    limit: int = RETRIEVAL_LIMIT,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[RetrievedChunk]:
    query_vector = embedding_model.embed_query(query)

    search_response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=SEMANTIC_CANDIDATE_LIMIT if HYBRID_SEARCH else limit,
        with_payload=True,
    )
    semantic: dict[str, tuple[object, int]] = {
        str(point.id): (point, rank)
        for rank, point in enumerate(search_response.points, start=1)
    }

    if not HYBRID_SEARCH:
        results = []
        for point, rank in semantic.values():
            payload = point.payload or {}
            if payload.get("text") and point.score >= threshold:
                results.append(
                    RetrievedChunk(
                        text=payload["text"],
                        source=payload.get("source", "unknown"),
                        page=payload.get("page"),
                        score=float(point.score),
                        via="semantic",
                        semantic_rank=rank,
                    )
                )
        return results[:limit]

    bm25_hits = get_index(qdrant_client, collection_name).search(
        query, BM25_CANDIDATE_LIMIT
    )
    bm25 = {
        chunk.id: (rank, score)
        for rank, (chunk, score) in enumerate(bm25_hits, start=1)
    }
    semantic_ids = list(semantic)
    bm25_ids = [chunk.id for chunk, _ in bm25_hits]
    fused = rrf_fuse(semantic_ids, bm25_ids, k=60)

    candidates: list[RetrievedChunk] = []
    for point_id in set(semantic) | set(bm25):
        semantic_item = semantic.get(point_id)
        bm25_item = bm25.get(point_id)
        if semantic_item is None:
            # Rescue is deliberately limited to candidates for which Qdrant
            # returned a cosine score; BM25 alone cannot bypass the gate.
            continue
        point, semantic_rank = semantic_item
        payload = point.payload or {}
        text = payload.get("text")
        if not text:
            continue
        cosine = float(point.score)
        bm25_rank, bm25_score = bm25_item if bm25_item else (None, 0.0)
        if cosine < threshold and not (
            bm25_rank is not None
            and bm25_rank <= 3
            and bm25_score >= 1.0
            and cosine >= 0.15
        ):
            continue
        via = "hybrid" if bm25_item else "semantic"
        candidates.append(
            RetrievedChunk(
                text=text,
                source=payload.get("source", "unknown"),
                page=payload.get("page"),
                score=cosine,
                via=via,
                semantic_rank=semantic_rank,
                bm25_rank=bm25_rank,
                bm25_score=bm25_score,
                fusion_score=fused.get(point_id, 0.0),
            )
        )
    candidates.sort(key=lambda item: item.fusion_score, reverse=True)
    return candidates[:limit]


def _call_generation_model(
    llm_client: OpenAI,
    model_name: str,
    prompt: str,
    language: str,
    max_tokens: int,
):
    return llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied context. "
                    f"Answer in {language}. If the context does not support an answer, "
                    f"reply exactly: {NO_ANSWER_MESSAGES[language]} "
                    "Return only the final answer, with no extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )


def generate_answer(
    llm_client: OpenAI,
    model_name: str,
    context: str,
    query: str,
    language: str,
    previous_answer: str | None = None,
) -> str:
    prompt = build_prompt(
        context=context,
        query=query,
        language=language,
        previous_answer=previous_answer,
    )

    response = _call_generation_model(
        llm_client, model_name, prompt, language, max_tokens=ANSWER_MAX_TOKENS
    )
    answer = clean_model_answer(response.choices[0].message.content)

    if not answer:
        retry_response = _call_generation_model(
            llm_client,
            model_name,
            prompt,
            language,
            max_tokens=ANSWER_MAX_TOKENS_RETRY,
        )
        answer = clean_model_answer(retry_response.choices[0].message.content)

    return answer


def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load the configured multilingual embedding model without fallback."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_qdrant_client(*, for_ingest: bool = False) -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    key_name = "QDRANT_INGEST_API_KEY" if for_ingest else "QDRANT_API_KEY"
    api_key = os.getenv(key_name)
    if not url or not api_key:
        missing = [name for name, value in (("QDRANT_URL", url), (key_name, api_key)) if not value]
        raise OSError("Missing required environment variables: " + ", ".join(missing))
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")),
    )


def load_clients() -> tuple[OpenAI, HuggingFaceEmbeddings, QdrantClient]:
    required_environment_variables = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]

    missing_variables = [
        variable for variable in required_environment_variables if not os.getenv(variable)
    ]

    if missing_variables:
        raise OSError(
            "Missing environment variables: " + ", ".join(missing_variables)
        )

    llm_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90")),
        max_retries=0,
    )
    embedding_model = load_embedding_model()
    qdrant_client = load_qdrant_client()

    return llm_client, embedding_model, qdrant_client


def _stable_point_id(source: str, chunk_index: int, chunk: str) -> str:
    """معرف ثابت مبني على المصدر وترتيب القطعة ومحتواها."""
    digest = hashlib.sha256(f"{source}:{chunk_index}:{chunk}".encode()).hexdigest()
    return str(uuid5(NAMESPACE_URL, digest))


def _file_hash(content: str) -> str:
    # A settings change must force a safe re-index even if source bytes stay put.
    versioned = (
        f"format={CHUNK_FORMAT_VERSION};size={CHUNK_SIZE};overlap={CHUNK_OVERLAP};"
        f"separators={CHUNK_SEPARATORS!r};model={EMBEDDING_MODEL_NAME};"
        f"vector_size={VECTOR_SIZE};\n{content}"
    )
    return hashlib.sha256(versioned.encode("utf-8")).hexdigest()


def _load_source_documents(directory: str) -> dict[str, list[tuple[str, int | None]]]:
    """Load source text, retaining PDF page boundaries and one-based page numbers."""
    if not os.path.isdir(directory):
        return {}
    documents: dict[str, list[tuple[str, int | None]]] = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(KNOWLEDGE_FILE_EXTENSIONS):
            continue
        path = os.path.join(directory, filename)
        extension = os.path.splitext(filename)[1].lower()
        try:
            if extension == ".pdf":
                pages = []
                for page_number, page in enumerate(PdfReader(path).pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages.append((text, page_number))
                if not pages:
                    logger.warning("pdf_skipped_no_extractable_text source=%s", filename)
                    continue
                documents[filename] = pages
            else:
                with open(path, "r", encoding="utf-8") as source_file:
                    text = source_file.read().strip()
                if text:
                    documents[filename] = [(text, None)]
        except (OSError, UnicodeError, PdfReadError) as error:
            logger.warning(
                "source_read_failed source=%s error_type=%s",
                filename,
                type(error).__name__,
            )
    return documents


def _load_source_files(directory: str) -> dict[str, str]:
    """Compatibility helper returning concatenated text while preserving pages in ingestion."""
    return {
        filename: "\n\n".join(text for text, _page in pages)
        for filename, pages in _load_source_documents(directory).items()
    }


def _existing_source_hashes(qdrant_client: QdrantClient) -> dict[str, str]:
    """يرجّع {source: file_hash} لكل المصادر المخزَّنة حاليًا في Qdrant."""
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        return {}

    hashes: dict[str, str] = {}
    next_offset = None

    while True:
        records, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["source", "file_hash"],
            limit=256,
            offset=next_offset,
        )

        for record in records:
            payload = record.payload or {}
            source = payload.get("source")
            if source:
                hashes[source] = payload.get("file_hash")

        if next_offset is None:
            break

    return hashes


def _delete_source_points(qdrant_client: QdrantClient, source: str) -> None:
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
        wait=True,
    )


def _ensure_source_index(qdrant_client: QdrantClient) -> None:
    schema = qdrant_client.get_collection(COLLECTION_NAME).payload_schema or {}
    if "source" not in schema:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )


def _detect_vector_size(embedding_model: HuggingFaceEmbeddings) -> int:
    return len(embedding_model.embed_query("dimension probe"))


def ingest_knowledge_base(
    qdrant_client: QdrantClient,
    embedding_model: HuggingFaceEmbeddings,
    directory: str = KNOWLEDGE_BASE_DIR,
    force: bool = FORCE_REINGEST,
) -> None:
    source_files = _load_source_documents(directory)

    if not source_files:
        raise ValueError(
            f"No {'/'.join(KNOWLEDGE_FILE_EXTENSIONS)} files found in '{directory}/'. "
            "Add knowledge files there before running."
        )

    collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)

    if not collection_exists:
        actual_vector_size = _detect_vector_size(embedding_model)
        if actual_vector_size != VECTOR_SIZE:
            raise ValueError(
                f"Embedding size is {actual_vector_size}, but VECTOR_SIZE is {VECTOR_SIZE}."
            )
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=actual_vector_size, distance=Distance.COSINE),
        )
        logger.info("qdrant_collection_created collection=%s", COLLECTION_NAME)
    _ensure_source_index(qdrant_client)

    existing_hashes = _existing_source_hashes(qdrant_client)
    existing_hash_to_source = {h: s for s, h in existing_hashes.items() if h}
    processed_hash_to_source: dict[str, str] = {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    total_new_chunks = 0

    for source, pages in source_files.items():
        content_for_hash = "\n".join(
            f"page={page_number}\n{text}" for text, page_number in pages
        )
        current_hash = _file_hash(content_for_hash)

        if existing_hashes.get(source) == current_hash and not force:
            print(f"'{source}' لم يتغيّر - تم التخطي.")
            continue

        duplicate_source = (
            existing_hash_to_source.get(current_hash)
            or processed_hash_to_source.get(current_hash)
        )
        if duplicate_source and duplicate_source != source:
            # If this path was previously indexed, remove its stale points now
            # that another canonical source owns identical content.
            if source in existing_hashes:
                _delete_source_points(qdrant_client, source)
            print(
                f"'{source}' تم تجاوزه - محتواه مطابق تمامًا لملف مفهرس "
                f"مسبقًا باسم '{duplicate_source}'."
            )
            continue

        processed_hash_to_source[current_hash] = source

        if source in existing_hashes and (force or existing_hashes[source] != current_hash):
            _delete_source_points(qdrant_client, source)
            print(f"'{source}' تغيّر - يُعاد فهرسته.")

        chunk_records: list[tuple[str, int | None]] = []
        for page_text, page_number in pages:
            chunk_records.extend(
                (chunk, page_number) for chunk in text_splitter.split_text(page_text)
            )
        chunks = [chunk for chunk, _page_number in chunk_records]
        if not chunks:
            continue
        embeddings = embedding_model.embed_documents(chunks)
        points = []
        for index, ((chunk, page_number), embedding) in enumerate(
            zip(chunk_records, embeddings, strict=True)
        ):
            payload = {
                "text": chunk,
                "source": source,
                "chunk_index": index,
                "file_hash": current_hash,
            }
            if page_number is not None:
                payload["page"] = page_number
            points.append(
                PointStruct(
                    id=_stable_point_id(source, index, chunk),
                    vector=embedding,
                    payload=payload,
                )
            )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        total_new_chunks += len(points)
        print(f"'{source}': تمت إضافة {len(points)} قطعة.")

    invalidate(COLLECTION_NAME)

    if total_new_chunks == 0:
        print("قاعدة المعرفة محدَّثة بالكامل - لا حاجة لأي تغيير.")
    else:
        print(f"اكتملت الفهرسة: {total_new_chunks} قطعة جديدة/محدَّثة.")


def answer_question(
    llm_client: OpenAI,
    embedding_model: HuggingFaceEmbeddings,
    qdrant_client: QdrantClient,
    query: str,
    memory: ConversationMemory,
) -> tuple[str, list[str]]:
    answer_language = detect_language(query)
    rewritten_query = rewrite_query(
        llm_client=llm_client,
        model_name=GENERATION_MODEL_NAME,
        current_question=query,
        history=memory.recent(),
    )

    if DEBUG_REWRITE and rewritten_query != query:
        print(f"[debug] rewritten query -> {rewritten_query}")

    relevant_chunks = retrieve_context(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        collection_name=COLLECTION_NAME,
        query=rewritten_query,
    )
    log_event(
        "retrieval_complete",
        result_count=len(relevant_chunks),
        results=[
            {
                "cosine": round(chunk.score, 5),
                "semantic_rank": chunk.semantic_rank,
                "bm25_rank": chunk.bm25_rank,
                "via": chunk.via,
            }
            for chunk in relevant_chunks
        ],
    )

    if not relevant_chunks:
        answer = _no_context_message(query)
        memory.add(query, rewritten_query, answer)
        return answer, []

    context = "\n\n---\n\n".join(chunk.text for chunk in relevant_chunks)

    # لو السؤال متابعة (انعاد صياغته)، نعطي النموذج الجواب السابق عشان ما يكرره.
    previous_answer = (
        memory.turns[-1].answer if memory.turns and rewritten_query != query else None
    )

    answer = generate_answer(
        llm_client=llm_client,
        model_name=GENERATION_MODEL_NAME,
        context=context,
        query=rewritten_query,
        language=answer_language,
        previous_answer=previous_answer,
    )

    if not answer or answer.strip() in NO_ANSWER_MESSAGES.values():
        answer = answer.strip() if answer else _no_context_message(query)
        sources: list[str] = []
    else:
        sources = sorted({source_label(chunk) for chunk in relevant_chunks})

    memory.add(query, rewritten_query, answer)
    return answer, sources


def main() -> None:
    llm_client, embedding_model, qdrant_client = load_clients()
    ingest_knowledge_base(qdrant_client, embedding_model)

    memory = ConversationMemory()

    while True:
        try:
            query = input("\nEnter your question (type 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            answer, sources = answer_question(
                llm_client=llm_client,
                embedding_model=embedding_model,
                qdrant_client=qdrant_client,
                query=query,
                memory=memory,
            )
            print("\nAnswer:")
            print(answer)
            if sources:
                print(f"\nSources: {', '.join(sources)}")

        except Exception as error:  # noqa: BLE001 - interactive CLI boundary
            error_text = str(error)
            if "402" in error_text or "Payment Required" in error_text:
                print(
                    "\nAn error occurred: OpenRouter usage credits are "
                    "depleted for this account. Add credits, then try again."
                )
            else:
                print(f"\nAn error occurred: {error_text}")


if __name__ == "__main__":
    main()
