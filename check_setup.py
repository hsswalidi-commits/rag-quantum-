"""Offline smoke check for the FastAPI package and retrieval helpers."""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

failures = 0


def report(ok: bool, description: str) -> None:
    global failures
    failures += int(not ok)
    print(("PASS " if ok else "FAIL ") + description)


print("[1] Source syntax")
for relative in ("main.py", "src/rag/api.py", "src/rag/core.py", "src/rag/hybrid.py", "src/rag/ingest.py"):
    path = ROOT / relative
    try:
        py_compile.compile(str(path), doraise=True)
        report(True, relative)
    except py_compile.PyCompileError as error:
        report(False, f"{relative}: {error.msg}")

print("[2] Retrieval behavior (no network)")
try:
    from rag import core, hybrid

    report(core.detect_language("ما هي بوابة CNOT؟") == "Arabic", "Arabic question language")
    report(core.detect_language("What is a qubit?") == "English", "English question language")
    report("cnot" in hybrid.tokenize("ما هي بوابة CNOT؟"), "tokenization")
    fused = hybrid.rrf_fuse(["a", "b"], ["b", "c"])
    report(max(fused, key=fused.get) == "b", "reciprocal rank fusion")
    report("multilingual" in core.EMBEDDING_MODEL_NAME.lower(), "multilingual embedding default")
except ImportError as error:
    report(False, f"backend dependencies unavailable: {type(error).__name__}")

print("[3] Deployment invariants")
core_text = (ROOT / "src/rag/core.py").read_text(encoding="utf-8")
api_text = (ROOT / "src/rag/api.py").read_text(encoding="utf-8")
report("delete_collection(" not in core_text, "ingestion does not delete Qdrant collections")
report("load_clients" in api_text and "ingest_knowledge_base" not in api_text, "API starts without ingestion")

print("PASS all checks" if failures == 0 else f"FAIL {failures} check(s)")
raise SystemExit(1 if failures else 0)
