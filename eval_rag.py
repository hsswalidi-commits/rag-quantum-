"""
uv run python run_eval.py              # hybrid
uv run python run_eval.py --no-hybrid  # بحث دلالي فقط (للمقارنة)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# in_scope = لازم يجاوب | out_of_scope = لازم يرفض
# "new_conversation": False = سؤال متابعة للسؤال اللي قبله
# "expected_source": "file.md" = لازم يظهر ضمن المصادر (اختياري)
# "keywords": كلمات لازم تظهر بالجواب (اختياري)
QUESTIONS = [
    {"q": "ما هو مبدأ عدم اليقين لهايزنبرغ؟", "type": "in_scope", "expected_source": "01_quantum_physics_foundations.md"},
    {"q": "ما هي تجربة قطة شرودنغر؟", "type": "in_scope", "expected_source": "01_quantum_physics_foundations.md"},
    {"q": "ما هو التشابك الكمومي؟", "type": "in_scope", "expected_source": "01_quantum_physics_foundations.md"},
    {"q": "ما الفرق بين البت والكيوبت؟", "type": "in_scope", "expected_source": "02_quantum_computing_basics.md"},
    {"q": "ما وظيفة بوابة هادامارد؟", "type": "in_scope", "expected_source": "02_quantum_computing_basics.md"},
    {"q": "ما هي بوابة CNOT؟", "type": "in_scope", "expected_source": "02_quantum_computing_basics.md", "keywords": ["cnot"]},
    {"q": "لماذا لا تُستخدم الحوسبة الكمومية في كل شيء؟", "type": "in_scope", "expected_source": "02_quantum_computing_basics.md"},
    {"q": "ما هي خوارزمية شور؟", "type": "in_scope", "expected_source": "03_quantum_algorithms.md"},
    {"q": "ماذا تفعل خوارزمية غروفر؟", "type": "in_scope", "expected_source": "03_quantum_algorithms.md"},
    {"q": "ما هي خوارزمية VQE؟", "type": "in_scope", "expected_source": "03_quantum_algorithms.md", "keywords": ["vqe"]},
    {"q": "How does Grover's algorithm speed up search?", "type": "in_scope", "expected_source": "03_quantum_algorithms.md"},
    {"q": "كيف تعمل الكيوبتات فائقة التوصيل؟", "type": "in_scope", "expected_source": "04_quantum_hardware_and_challenges.md"},
    {"q": "ما هو تصحيح الأخطاء الكمومية؟", "type": "in_scope", "expected_source": "04_quantum_hardware_and_challenges.md"},
    {"q": "ما هو عصر NISQ؟", "type": "in_scope", "expected_source": "04_quantum_hardware_and_challenges.md", "keywords": ["nisq"]},
    {"q": "What are trapped ion qubits?", "type": "in_scope", "expected_source": "04_quantum_hardware_and_challenges.md"},
    {"q": "من هو رئيس الولايات المتحدة؟", "type": "out_of_scope"},
    {"q": "ما هي عاصمة فرنسا؟", "type": "out_of_scope"},
    {"q": "كيف أطبخ الكبسة؟", "type": "out_of_scope"},
    {"q": "ما هو أفضل هاتف في العالم؟", "type": "out_of_scope"},
    {"q": "What is the weather today?", "type": "out_of_scope"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--questions", default=None, help="ملف JSON بديل للأسئلة")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import app

    if args.no_hybrid:
        app.HYBRID_SEARCH = False
    mode = "hybrid" if app.HYBRID_SEARCH else "semantic_only"

    questions = QUESTIONS
    if args.questions:
        with open(args.questions, "r", encoding="utf-8-sig") as file_handle:
            questions = json.load(file_handle)

    llm_client, embedding_model, qdrant_client = app.load_clients()
    app.ingest_knowledge_base(qdrant_client, embedding_model)

    refusals = set(app.NO_ANSWER_MESSAGES.values())
    memory = app.ConversationMemory()
    results = []
    print(f"\nالوضع: {mode} | عدد الأسئلة: {len(questions)}\n")

    for number, item in enumerate(questions, start=1):
        question = item["q"]
        kind = item.get("type", "in_scope")
        if item.get("new_conversation", True):
            memory = app.ConversationMemory()

        error = None
        try:
            answer, sources = app.answer_question(
                llm_client=llm_client,
                embedding_model=embedding_model,
                qdrant_client=qdrant_client,
                query=question,
                memory=memory,
            )
        except Exception as exc:  # نكمل بقية الأسئلة حتى لو واحد فشل
            answer, sources, error = "", [], str(exc)

        refused = answer.strip() in refusals
        language_ok = bool(answer) and app.detect_language(answer) == app.detect_language(question)

        expected = item.get("expected_source") or []
        if isinstance(expected, str):
            expected = [expected]
        source_ok = not expected or any(e in label for e in expected for label in sources)

        keywords = [k.lower() for k in item.get("keywords", [])]
        keywords_ok = all(k in answer.lower() for k in keywords) if keywords else None

        reasons = []
        if error:
            reasons.append(f"خطأ: {error}")
        if kind == "in_scope":
            if refused:
                reasons.append("رفض بالغلط")
            if not source_ok:
                reasons.append(f"المصدر المتوقع {expected} غير موجود")
            if keywords_ok is False:
                reasons.append("كلمات ناقصة بالجواب")
        elif not refused:
            reasons.append("المفروض يرفض لكنه جاوب")
        passed_main = not reasons
        if not language_ok and not error:
            reasons.append("لغة الجواب تختلف عن لغة السؤال")

        results.append({
            "n": number, "q": question, "type": kind, "answer": answer,
            "sources": sources, "refused": refused, "language_ok": language_ok,
            "keywords_ok": keywords_ok, "passed_main": passed_main, "reasons": reasons,
        })
        print(f"{'✅' if not reasons else '❌'} [{number}] {question}")

    in_scope = [r for r in results if r["type"] == "in_scope"]
    out_scope = [r for r in results if r["type"] != "in_scope"]
    with_keywords = [r for r in results if r["keywords_ok"] is not None]

    print("\n" + "=" * 60)
    print(f"أسئلة موجبة صح (مصدر + بدون رفض غلط) : {sum(r['passed_main'] for r in in_scope)}/{len(in_scope)}")
    print(f"أسئلة خارج النطاق رُفضت صح           : {sum(r['passed_main'] for r in out_scope)}/{len(out_scope)}")
    print(f"إجابات تطابق لغة السؤال              : {sum(r['language_ok'] for r in results)}/{len(results)}")
    if with_keywords:
        print(f"فحص الكلمات المفتاحية                : {sum(bool(r['keywords_ok']) for r in with_keywords)}/{len(with_keywords)}")
    print("=" * 60)

    failures = [r for r in results if r["reasons"]]
    if failures:
        print("\nتفاصيل الفشل:")
        for r in failures:
            print(f"\n[{r['n']}] {r['q']}")
            print("   السبب   :", " | ".join(r["reasons"]))
            print("   المصادر :", r["sources"])
            print("   الجواب  :", (r["answer"] or "-")[:200].replace("\n", " "))

    out_path = args.out or f"eval_results_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, ensure_ascii=False, indent=2)
    print(f"\nانحفظت النتائج الكاملة في: {out_path}")


if __name__ == "__main__":
    main()
