import streamlit as st

import storage
from app import (
    ConversationMemory,
    answer_question,
    ingest_knowledge_base,
    load_clients,
    retrieve_context,
    save_uploaded_file,
    source_label,
    COLLECTION_NAME,
    RETRIEVAL_LIMIT,
    SIMILARITY_THRESHOLD,
)

st.set_page_config(page_title="Quantum Knowledge Assistant", page_icon="🧠", layout="wide")

# Simple right-to-left styling for the Arabic content areas.
st.markdown(
    """
    <style>
    .stChatMessage, .stMarkdown, .stCaption { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Setting up (loading embedding model, indexing files)...")
def get_clients():
    llm_client, embedding_model, qdrant_client = load_clients()
    ingest_knowledge_base(qdrant_client, embedding_model)
    storage.init_db()
    return llm_client, embedding_model, qdrant_client


llm_client, embedding_model, qdrant_client = get_clients()

# --- Conversation state -----------------------------------------------------

if "conversation_id" not in st.session_state:
    # Reuse the most recent conversation if one exists, otherwise start fresh.
    existing = storage.list_conversations()
    if existing:
        st.session_state.conversation_id = existing[0].id
    else:
        st.session_state.conversation_id = storage.create_conversation()


def _load_conversation_into_session(conversation_id: str) -> None:
    """Rebuild ConversationMemory (for query rewriting) and the display
    history from what's stored on disk for this conversation."""
    st.session_state.conversation_id = conversation_id
    stored_messages = storage.load_messages(conversation_id)

    memory = ConversationMemory()
    display_history = []

    pending_question = None
    for message in stored_messages:
        if message.role == "user":
            pending_question = message.content
            display_history.append(("user", message.content, None))
        else:
            if pending_question is not None:
                memory.add(pending_question, pending_question, message.content)
                pending_question = None
            display_history.append(("assistant", message.content, message.sources))

    st.session_state.memory = memory
    st.session_state.display_history = display_history
    st.session_state.last_retrieved = []


if "memory" not in st.session_state:
    _load_conversation_into_session(st.session_state.conversation_id)

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

# --- Sidebar: conversation history + settings -------------------------------

with st.sidebar:
    st.subheader("Conversations")

    if st.button("+ New conversation", use_container_width=True):
        new_id = storage.create_conversation()
        _load_conversation_into_session(new_id)
        st.rerun()

    st.divider()

    for conversation in storage.list_conversations():
        is_current = conversation.id == st.session_state.conversation_id
        label = ("▶ " if is_current else "") + conversation.title
        if st.button(label, key=f"conv_{conversation.id}", use_container_width=True):
            if not is_current:
                _load_conversation_into_session(conversation.id)
                st.rerun()

    st.divider()
    st.subheader("Knowledge base")

    uploaded_files = st.file_uploader(
        "Add files (PDF / MD / TXT)",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Add to knowledge base", use_container_width=True):
        with st.spinner("Saving and indexing (large PDFs can take a while)..."):
            try:
                for uploaded in uploaded_files:
                    save_uploaded_file(uploaded.name, uploaded.getvalue())
                # Unchanged files are skipped automatically by their hash.
                ingest_knowledge_base(qdrant_client, embedding_model)
                st.success(f"Added {len(uploaded_files)} file(s).")
            except Exception as error:
                st.error(f"Upload failed: {error}")

    if st.button("Re-index knowledge base", use_container_width=True):
        with st.spinner("Re-indexing..."):
            ingest_knowledge_base(qdrant_client, embedding_model, force=True)
        st.success("Done.")

# --- Main chat area ----------------------------------------------------------

st.title("🧠 Quantum Knowledge Assistant")

for role, text, sources in st.session_state.display_history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            st.caption("المصادر: " + "، ".join(sources))

query = st.chat_input("اكتب سؤالك هنا...")

if query:
    is_first_message = len(st.session_state.display_history) == 0

    st.session_state.display_history.append(("user", query, None))
    with st.chat_message("user"):
        st.markdown(query)
    storage.save_message(st.session_state.conversation_id, "user", query, [])

    if is_first_message:
        storage.set_title_from_first_question(st.session_state.conversation_id, query)

    with st.chat_message("assistant"):
        with st.spinner("أفكر..."):
            try:
                answer, sources = answer_question(
                    llm_client=llm_client,
                    embedding_model=embedding_model,
                    qdrant_client=qdrant_client,
                    query=query,
                    memory=st.session_state.memory,
                )
                # Diagnostics: use the same (possibly rewritten) query that
                # answer_question searched with, and show scores even for
                # chunks below the threshold (threshold=0.0 is diagnostic-only).
                diagnostic_query = st.session_state.memory.turns[-1].rewritten_question
                diagnostic_chunks = retrieve_context(
                    qdrant_client, embedding_model, COLLECTION_NAME, diagnostic_query,
                    limit=RETRIEVAL_LIMIT, threshold=0.0,
                )
            except Exception as error:
                answer = f"صار خطأ: {error}"
                sources = []
                diagnostic_chunks = []

        st.markdown(answer)
        if sources:
            st.caption("المصادر: " + "، ".join(sources))

        with st.expander("🔍 القطع المسترجعة (تشخيص)"):
            if not diagnostic_chunks:
                st.caption("ولا قطعة رجعت من البحث.")
            for chunk in diagnostic_chunks:
                if chunk.score >= SIMILARITY_THRESHOLD:
                    marker = "✅"
                elif chunk.via == "keyword":
                    marker = "🔑"
                else:
                    marker = "❌"
                semantic_rank = f"#{chunk.semantic_rank}" if chunk.semantic_rank else "-"
                bm25_rank = (
                    f"#{chunk.bm25_rank} ({chunk.bm25_score:.1f})" if chunk.bm25_rank else "-"
                )
                st.markdown(
                    f"{marker} **{source_label(chunk)}** — cosine: `{chunk.score:.3f}` "
                    f"| دلالي: `{semantic_rank}` | كلمات: `{bm25_rank}`"
                )
                st.caption(chunk.text[:220] + ("..." if len(chunk.text) > 220 else ""))

    storage.save_message(st.session_state.conversation_id, "assistant", answer, sources)
    st.session_state.display_history.append(("assistant", answer, sources))
