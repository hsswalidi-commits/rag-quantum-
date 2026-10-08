from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

COLLECTION_NAME = "client"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
VECTOR_SIZE = 384

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
KNOWLEDGE_FILE_EXTENSIONS = (".md", ".txt", ".pdf")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 2


SIMILARITY_THRESHOLD = 0.30
RETRIEVAL_LIMIT = 6


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



def build_prompt(context: str, query: str) -> str:
    return f"""
Use only the supplied context to answer the question.

Instructions:
- Return only the final answer, nothing else.
- Do not explain your reasoning or the retrieval process.
- If the answer is not available in the context, say:
  "The answer is not available in the provided context."
- - Answer in the same language as the user's question.
- Give a complete answer: at least 2-3 full sentences when the context
  supports it. Avoid single-word or single-phrase answers unless the
  question explicitly asks for a short list or a yes/no.

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

    except Exception:
   
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
        limit=limit,
        with_payload=True,
    )

    results = []
    for result in search_response.points:
        payload = result.payload or {}
        text = payload.get("text")
        if text and result.score >= threshold:
            results.append(
                RetrievedChunk(
                    text=text,
                    source=payload.get("source", "unknown"),
                    score=result.score,
                )
            )

    return results


def _call_generation_model(
    llm_client: OpenAI, model_name: str, prompt: str, max_tokens: int
):
    return llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied context. "
                    "Return only the final answer, with no extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )


def generate_answer(
    llm_client: OpenAI, model_name: str, context: str, query: str
) -> str:
    prompt = build_prompt(context=context, query=query)

    response = _call_generation_model(
        llm_client, model_name, prompt, max_tokens=ANSWER_MAX_TOKENS
    )
    answer = clean_model_answer(response.choices[0].message.content)

    
    if not answer:
        retry_response = _call_generation_model(
            llm_client, model_name, prompt, max_tokens=ANSWER_MAX_TOKENS_RETRY
        )
        answer = clean_model_answer(retry_response.choices[0].message.content)

    return answer


def load_clients() -> tuple[OpenAI, HuggingFaceEmbeddings, QdrantClient]:
    required_environment_variables = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]

    missing_variables = [
        variable for variable in required_environment_variables if not os.getenv(variable)
    ]

    if missing_variables:
        raise EnvironmentError(
            "Missing environment variables: " + ", ".join(missing_variables)
        )

    
    llm_client = OpenAI(
        base_url=OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY")
    )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )

    return llm_client, embedding_model, qdrant_client

def _stable_point_id(source: str, chunk_index: int, chunk: str) -> str:
    """معرف ثابت مبني على المصدر وترتيب القطعة ومحتواها."""
    digest = hashlib.sha256(f"{source}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()
    return str(uuid5(NAMESPACE_URL, digest))


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def _load_source_files(directory: str) -> dict[str, str]:
    """تحميل محتوى ملفات .md و .txt و .pdf من مجلد قاعدة المعرفة."""
    if not os.path.isdir(directory):
        return {}

    files: dict[str, str] = {}

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(KNOWLEDGE_FILE_EXTENSIONS):
            continue

        path = os.path.join(directory, filename)
        extension = os.path.splitext(filename)[1].lower()

        try:
            if extension in {".md", ".txt"}:
                with open(path, "r", encoding="utf-8") as file_handle:
                    content = file_handle.read()

            elif extension == ".pdf":
                reader = PdfReader(path)
                pages = []

                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(page_text)

                content = "\n\n".join(pages)

            else:
                continue

            if content.strip():
                files[filename] = content

        except Exception as error:
            print(f"Could not read '{filename}': {error}")

    return files


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
    )


def _detect_vector_size(embedding_model: HuggingFaceEmbeddings) -> int:
    return len(embedding_model.embed_query("dimension probe"))


def ingest_knowledge_base(
    qdrant_client: QdrantClient,
    embedding_model: HuggingFaceEmbeddings,
    directory: str = KNOWLEDGE_BASE_DIR,
    force: bool = FORCE_REINGEST,
) -> None:
    source_files = _load_source_files(directory)

    if not source_files:
        raise ValueError(
            f"No {'/'.join(KNOWLEDGE_FILE_EXTENSIONS)} files found in '{directory}/'. "
            "Add knowledge files there before running."
        )

    collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)

    if force and collection_exists:
        qdrant_client.delete_collection(COLLECTION_NAME)
        collection_exists = False

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
        print("Collection created.")

    existing_hashes = {} if force else _existing_source_hashes(qdrant_client)
    existing_hash_to_source = {h: s for s, h in existing_hashes.items() if h}
    processed_hash_to_source: dict[str, str] = {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_new_chunks = 0

    for source, content in source_files.items():

        current_hash = _file_hash(content)

        if existing_hashes.get(source) == current_hash:
            print(f"'{source}' لم يتغيّر - تم التخطي.")
            continue

        duplicate_source = (
            existing_hash_to_source.get(current_hash)
            or processed_hash_to_source.get(current_hash)
        )
        if duplicate_source and duplicate_source != source:
            print(
                f"'{source}' تم تجاوزه - محتواه مطابق تمامًا لملف مفهرس "
                f"مسبقًا باسم '{duplicate_source}'."
            )
            continue

        processed_hash_to_source[current_hash] = source

        if source in existing_hashes:
            _delete_source_points(qdrant_client, source)
            print(f"'{source}' تغيّر - يُعاد فهرسته.")

        chunks = text_splitter.split_text(content)
        if not chunks:
            continue

        embeddings = embedding_model.embed_documents(chunks)

        points = [
            PointStruct(
                id=_stable_point_id(source, index, chunk),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_index": index,
                    "file_hash": current_hash,
                },
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        total_new_chunks += len(points)
        print(f"'{source}': تمت إضافة {len(points)} قطعة.")

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

    if not relevant_chunks:
        answer = _no_context_message(query)
        memory.add(query, rewritten_query, answer)
        return answer, []

    context = "\n\n---\n\n".join(chunk.text for chunk in relevant_chunks)

    answer = generate_answer(
        llm_client=llm_client,
        model_name=GENERATION_MODEL_NAME,
        context=context,
        query=rewritten_query,
    )

    if not answer:
        answer = (
            "The model did not return a usable final answer. "
            "Try another non-reasoning instruction model."
        )

    sources = sorted({chunk.source for chunk in relevant_chunks})
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

        except Exception as error:
            error_text = str(error)
            if "402" in error_text or "Payment Required" in error_text:
                print(""" """  """ """
                    "\nAn error occurred: OpenRouter usage credits are "
                    "depleted for this account. Add credits, then try again."
                )
            else:
                print(f"\nAn error occurred: {error_text}")


if __name__ == "__main__":
    main()