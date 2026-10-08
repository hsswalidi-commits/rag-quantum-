"""
شغّله من نفس مجلد المشروع:   python check_setup.py
يتأكد إن الملفات الجديدة انتقلت صح، والمكتبات منصّبة، والمنطق الأساسي يشتغل.
ما يتصل بـQdrant ولا OpenRouter ولا يغيّر أي شي.
"""
import importlib
import os
import py_compile
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

failures = 0


def report(ok: bool, message: str) -> None:
    global failures
    if not ok:
        failures += 1
    print(("✅ " if ok else "❌ ") + message)


# 1) الملفات موجودة وما فيها أخطاء صياغة (مثل مشكلة الـIndentation)
print("\n[1] الملفات")
for filename in ["app.py", "chat_app.py", "hybrid.py", "storage.py"]:
    path = os.path.join(PROJECT_DIR, filename)
    if not os.path.exists(path):
        report(False, f"{filename} غير موجود")
        continue
    try:
        py_compile.compile(path, doraise=True)
        report(True, f"{filename} موجود وصياغته سليمة")
    except py_compile.PyCompileError as error:
        report(False, f"{filename} فيه خطأ صياغة: {error.msg}")

# 2) علامات تثبت إن النسخة الجديدة هي اللي موجودة
print("\n[2] هل هذي النسخ الجديدة؟")
markers = {
    "app.py": [
        "import hybrid",
        "CHUNK_OVERLAP = 60",
        "def _no_context_message",
        "def detect_language",
        "def save_uploaded_file",
        "HYBRID_SEARCH =",
        "duplicate_source",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ],
    "chat_app.py": ["file_uploader", "save_uploaded_file", "source_label", "SIMILARITY_THRESHOLD"],
    "hybrid.py": ["def rrf_fuse", "def tokenize", "def invalidate"],
}
for filename, needed in markers.items():
    path = os.path.join(PROJECT_DIR, filename)
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as file_handle:
        content = file_handle.read()
    for marker in needed:
        report(marker in content, f"{filename}: «{marker}»")

# 3) المكتبات
print("\n[3] المكتبات")
for package in [
    "streamlit",
    "pypdf",
    "rank_bm25",
    "qdrant_client",
    "langchain_text_splitters",
    "langchain_huggingface",
    "openai",
    "dotenv",
]:
    try:
        importlib.import_module(package)
        report(True, package)
    except ImportError:
        install_name = {"rank_bm25": "rank-bm25", "dotenv": "python-dotenv"}.get(package, package)
        report(False, f"{package} غير منصّبة  →  pip install {install_name}")

# 4) اختبار المنطق (بدون إنترنت)
print("\n[4] اختبار المنطق")
try:
    import hybrid

    tokens = hybrid.tokenize("ما هي بوابة CNOT؟")
    report("cnot" in tokens and not any("؟" in t for t in tokens), f"tokenize → {tokens}")

    fused = hybrid.rrf_fuse(["a", "b"], ["b", "c"])
    report(max(fused, key=fused.get) == "b", "rrf_fuse: القطعة اللي بالقائمتين تفوز")
except Exception as error:
    report(False, f"hybrid.py: {error}")

try:
    import app

    report(app.detect_language("ما هي بوابة CNOT؟") == "Arabic", "detect_language: عربي")
    report(app.detect_language("What is a qubit?") == "English", "detect_language: إنجليزي")
    report(app._no_context_message("وش هو الكيوبت؟") != app._no_context_message("What is a qubit?"),
           "رسالة الرفض تتغير حسب لغة السؤال")
    report(app.CHUNK_OVERLAP >= 20, f"CHUNK_OVERLAP = {app.CHUNK_OVERLAP}")
    report("multilingual" in app.EMBEDDING_MODEL_NAME.lower(),
           f"موديل الـembeddings = {app.EMBEDDING_MODEL_NAME}")
except Exception as error:
    report(False, f"استيراد app.py فشل: {error}")

print("\n" + ("كل شي تمام ✅" if failures == 0 else f"فيه {failures} مشكلة ❌ - راجع السطور اللي فيها ❌ فوق"))
